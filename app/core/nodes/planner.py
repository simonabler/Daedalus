"""Planner plan node and plan parsing helpers."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *
from ._intelligence_helpers import *
from ._prompt_enrichment import *

def planner_plan_node(state: GraphState) -> dict:
    """Planner creates the plan and routes items to specialized workers."""
    emit_node_start("planner", "Planning", item_desc=state.user_request[:100])
    emit_status("planner", f"Analyzing request: {state.user_request[:80]}...", **_progress_meta(state, "planning"))

    intent = (state.input_intent or "").strip().lower()
    if intent not in ROUTER_INTENTS and intent not in {"question_only", "resume_workflow", "new_task"}:
        intent = _classify_request_intent(state.user_request)
    if intent == "question_only":
        answer = _answer_question_directly(state)
        emit_status(
            "planner",
            "Answered user question directly (no coding workflow).",
            **_progress_meta(state, "complete"),
        )
        emit_node_end("planner", "Planning", "Question handled by planner directly")
        return {
            "input_intent": intent,
            "planner_response": answer,
            "phase": WorkflowPhase.COMPLETE,
            "todo_items": [],
            "current_item_index": -1,
            "completed_items": 0,
            "stop_reason": "question_answered",
        }

    if intent in {"resume_workflow", "resume"}:
        # Prefer in-memory items (fastest, most up-to-date)
        if state.todo_items:
            pending = [i for i in state.todo_items if i.status != ItemStatus.DONE]
            if pending:
                current_idx = next(
                    (idx for idx, i in enumerate(state.todo_items) if i.status != ItemStatus.DONE),
                    state.current_item_index,
                )
                emit_node_end("planner", "Planning", f"Resumed in-memory plan at item {current_idx + 1}")
                return {
                    "input_intent": intent,
                    "phase": WorkflowPhase.CODING,
                    "current_item_index": current_idx,
                    "stop_reason": "",
                }
        resumed = _resume_from_saved_todo(state)
        if resumed is not None:
            resumed["input_intent"] = intent
            emit_node_end("planner", "Planning", "Resumed workflow from tasks/todo.md")
            return resumed
        emit_status("planner", "No resumable TODO list found. Nothing to resume.", **_progress_meta(state, "complete"))
        emit_node_end("planner", "Planning", "Resume requested but no open items were found")
        return {
            "input_intent": intent,
            "planner_response": "No resumable TODO items found in tasks/todo.md.",
            "phase": WorkflowPhase.COMPLETE,
            "todo_items": [],
            "current_item_index": -1,
            "completed_items": 0,
            "stop_reason": "resume_not_found",
        }

    try:
        ensure_memory_files()
    except Exception as exc:
        logger.warning("Could not ensure memory files: %s", exc)

    stats = get_memory_stats()
    for key, info in stats.items():
        if info["needs_compression"]:
            emit_status(
                "planner",
                f"Compressing memory: {key} ({info['chars']} chars)",
                **_progress_meta(state, "planning"),
            )
            _compress_memory_file(key)

    total_chars = sum(s["chars"] for s in stats.values())
    if total_chars > 0:
        emit_status(
            "planner",
            f"Shared memory loaded: {total_chars} chars across {len(stats)} files",
            **_progress_meta(state, "planning"),
        )

    context_parts = [
        f"User request: {state.user_request}",
        f"Repository root: {state.repo_root}",
        f"Current branch: {state.branch_name or 'not set'}",
        f"Execution platform: {state.execution_platform or platform.platform()}",
        "Available workers: coder_a, coder_b, documenter.",
        "Available quality gates: reviewer_a/reviewer_b and tester.",
        "Routing history:",
        history_summary(state.repo_root),
    ]

    if state.context_loaded:
        context_parts.append("Repository context has been preloaded.")
    if state.repo_facts:
        context_parts.append(_format_repo_context_for_prompt(state.repo_facts))
        context_parts.append("Repository facts summary:\n" + _format_context_summary(state.repo_facts))
    intelligence_summary = _format_intelligence_summary_for_prompt(state)
    if intelligence_summary:
        context_parts.append(intelligence_summary)
    if state.context_listing:
        context_parts.append(
            "Repository listing (depth<=2):\n"
            + _truncate_context_text(state.context_listing, limit=4000)
        )
    if state.agent_instructions:
        context_parts.append(
            "Repository instructions (AGENT/README snippets):\n"
            + _truncate_context_text(state.agent_instructions, limit=5000)
        )

    memory_ctx = load_all_memory()
    if memory_ctx:
        context_parts.append(memory_ctx)

    try:
        lessons = read_file.invoke({"path": "tasks/lessons.md"})
        if not lessons.startswith("ERROR"):
            context_parts.append(f"Lessons learned:\n{lessons}")
    except Exception:
        pass

    prompt = (
        "Analyze the request and create a detailed execution plan.\n\n"
        + "\n\n".join(context_parts)
        + "\n\nReturn STRICT JSON only:\n"
        "{\n"
        '  "plan": [\n'
        "    {\n"
        '      "description": "short action item",\n'
        '      "task_type": "coding|documentation|testing|ops",\n'
        '      "acceptance_criteria": ["..."],\n'
        '      "verification_commands": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "If this is a programming request, provide a step-by-step TODO plan."
    )

    result, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=PLANNER_TOOLS, inject_memory=False, node="planner",
    )
    items = _parse_plan_from_result(result)

    if not items and is_programming_request(state.user_request):
        items = [
            TodoItem(
                id="item-001",
                description=state.user_request.strip()[:140] or "Implement requested programming task",
                task_type="coding",
                acceptance_criteria=["Requested behavior is implemented."],
                verification_commands=["python -m pytest -q"],
                status=ItemStatus.PENDING,
            )
        ]

    for idx, item in enumerate(items):
        if not item.task_type:
            item.task_type = classify_task_type(item.description)
        item.task_type = item.task_type.lower()
        item.id = item.id or f"item-{idx + 1:03d}"

        candidates = _candidate_agents_for_task(item.task_type)
        assigned_agent, _history = select_agent_thompson(state.repo_root, item.task_type, candidates)
        item.assigned_agent = assigned_agent
        item.assigned_reviewer = _reviewer_for_worker(assigned_agent)

    if items:
        plan_text = "## TODO Plan\n"
        for idx, item in enumerate(items, start=1):
            owner = _coder_label(item.assigned_agent or "coder_a")
            plan_text += f"  {idx}. [ ] ({item.task_type}) {item.description} -> {owner}\n"
        emit_plan("planner", plan_text, items_count=len(items))
        emit_status(
            "planner",
            f"Plan created with {len(items)} items",
            items_count=len(items),
            **_progress_meta(state, "planning"),
        )
        _write_todo_file(items, state.user_request)
    else:
        emit_status("planner", "Could not parse plan items from planner output", **_progress_meta(state, "planning"))

    branch = state.branch_name
    if not branch or branch in ("main", "master"):
        slug = state.user_request[:30].lower().replace(" ", "-")
        slug = "".join(ch for ch in slug if ch.isalnum() or ch == "-")
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        branch = f"feature/{stamp}-{slug}"
        git_create_branch.invoke({"branch_name": branch})
        emit_status("planner", f"Created branch: {branch}", **_progress_meta(state, "planning"))

    if items:
        first = items[0]
        active_coder = first.assigned_agent or _assign_coder_pair(0)[0]
        active_reviewer = first.assigned_reviewer or _reviewer_for_worker(active_coder)
    else:
        active_coder, active_reviewer = _assign_coder_pair(0)

    updates = {
        "input_intent": intent,
        "todo_items": items if items else state.todo_items,
        "current_item_index": 0 if items else state.current_item_index,
        "branch_name": branch,
        # Plan Approval Gate logic:
        #   First-time plan (revision_count=0) → needs_plan_approval=True, stay in PLANNING
        #     so _route_after_plan sends us to plan_approval_gate.
        #   After a human revision (revision_count>=1) → skip gate, go straight to CODING.
        "phase": WorkflowPhase.CODING if (not items or state.plan_revision_count >= 1) else WorkflowPhase.PLANNING,
        "needs_replan": False,
        "active_coder": active_coder,
        "active_reviewer": active_reviewer,
        "needs_plan_approval": bool(items) and state.plan_revision_count < 1,
        "plan_approved": False,
        "plan_approval_feedback": "",
    }
    _save_checkpoint_snapshot(state, updates, "plan_complete")

    emit_node_end("planner", "Planning", f"Plan ready - starting with item 1 -> {_coder_label(active_coder)}")
    return updates

def _compress_memory_file(key: str) -> None:
    """Use the planner LLM to compress an oversized memory file."""
    prompt = build_compression_prompt(key)
    if not prompt:
        return
    try:
        result, budget_update = _invoke_with_budget(
            state, "planner", [HumanMessage(content=prompt)],
            tools=None, inject_memory=False, node="planner_review",
        )
        # Clean up: strip markdown fences if present
        cleaned = result.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        save_compressed(key, cleaned)
        emit_status("planner", f"Memory compressed: {key}", phase="planning")
    except Exception as e:
        logger.warning("Memory compression failed for %s: %s", key, e)

def _parse_plan_from_result(result: str) -> list[TodoItem]:
    """Extract TODO items from planner output. JSON-first with text fallback."""
    items: list[TodoItem] = []

    parsed = None
    try:
        parsed = json.loads(result.strip())
    except json.JSONDecodeError:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(result[start:end])
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
        for idx, raw in enumerate(parsed["plan"], start=1):
            if not isinstance(raw, dict):
                continue
            desc = str(raw.get("description", "")).strip()
            if not desc:
                continue
            task_type = str(raw.get("task_type", "")).strip().lower() or classify_task_type(desc)
            acceptance = raw.get("acceptance_criteria") or []
            verify = raw.get("verification_commands") or []
            items.append(
                TodoItem(
                    id=f"item-{idx:03d}",
                    description=desc,
                    task_type=task_type,
                    acceptance_criteria=[str(v) for v in acceptance if str(v).strip()],
                    verification_commands=[str(v) for v in verify if str(v).strip()],
                    status=ItemStatus.PENDING,
                )
            )
        if items:
            return items

    lines = result.split("\n")
    current_id = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
            current_id += 1
            desc = stripped.replace("- [ ]", "").replace("- [x]", "").strip()
            if desc and desc[0].isdigit():
                parts = desc.split(":", 1)
                if len(parts) > 1:
                    desc = parts[1].strip()
            items.append(TodoItem(
                id=f"item-{current_id:03d}",
                description=desc or f"Task {current_id}",
                task_type=classify_task_type(desc),
                status=ItemStatus.PENDING,
            ))

    if not items:
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:4]:
                current_id += 1
                desc = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
                items.append(TodoItem(
                    id=f"item-{current_id:03d}",
                    description=desc,
                    task_type=classify_task_type(desc),
                    status=ItemStatus.PENDING,
                ))
    return items


# ---------------------------------------------------------------------------
# NODE: coder  (reads shared memory)
# ---------------------------------------------------------------------------
