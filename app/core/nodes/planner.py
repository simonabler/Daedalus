"""Planner nodes — plan creation, review, decide, and env-fix."""
from __future__ import annotations

import json
import platform
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.config import get_settings
from app.core.events import (
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_plan,
    emit_status,
    emit_verdict,
)
from app.core.logging import get_logger
from app.core.memory import (
    build_compression_prompt,
    ensure_memory_files,
    get_memory_stats,
    load_all_memory,
    save_compressed,
)
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase
from app.core.task_routing import (
    classify_task_type,
    history_summary,
    record_agent_outcome,
    select_agent_thompson,
)
from app.core.token_budget import BudgetExceededException
from app.tools.filesystem import read_file, write_file
from app.tools.git import git_command, git_create_branch

from ._helpers import (
    CODER_TOOLS,
    PLANNER_TOOLS,
    ROUTER_INTENTS,
    _assign_coder_pair,
    _candidate_agents_for_task,
    _coder_label,
    _invoke_with_budget,
    _os_note,
    _progress_meta,
    _reviewer_for_worker,
    _reviewer_label,
    _save_checkpoint_snapshot,
    _write_todo_file,
)
from ._context_format import (
    _format_context_summary,
    _format_intelligence_summary_for_prompt,
    _format_repo_context_for_prompt,
    _truncate_context_text,
)
from .router import _answer_question_directly, _classify_request_intent
from .resume import _resume_from_saved_todo

logger = get_logger("core.nodes.planner")

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

    # ── Guard: don't overwrite an existing plan with pending items ─────────
    # If tasks/todo.md has unchecked items, resume them instead of starting
    # from scratch.  This prevents loss of progress after a server restart
    # where the user sends a new task identical to the one in progress.
    existing = _resume_from_saved_todo(state)
    if existing and existing.get("todo_items"):
        existing_items = existing["todo_items"]
        pending_count = sum(1 for i in existing_items if i.status != ItemStatus.DONE)
        done_count = sum(1 for i in existing_items if i.status == ItemStatus.DONE)
        if pending_count > 0 and done_count > 0:
            # There IS partial progress — resume it instead of replanning.
            emit_status(
                "planner",
                f"Found existing plan with {done_count} done / {pending_count} pending "
                f"items — resuming instead of replanning.",
                **_progress_meta(state, "planning"),
            )
            existing["input_intent"] = "resume_workflow"
            emit_node_end("planner", "Planning", f"Resumed existing plan ({done_count} done, {pending_count} pending)")
            return existing

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
            _compress_memory_file(key, state)

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


def _compress_memory_file(key: str, state: GraphState | None = None) -> None:
    """Use the planner LLM to compress an oversized memory file."""
    prompt = build_compression_prompt(key)
    if not prompt:
        return
    try:
        if state is None:
            # Without state, skip LLM compression (no budget context).
            logger.warning("Memory compression skipped for %s: no state context", key)
            return
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


def planner_review_node(state: GraphState) -> dict:
    """Planner final review gate."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to review")
        return {"error_message": "No item to review", "phase": WorkflowPhase.STOPPED}

    emit_node_start("planner", "Final Review", item_id=item.id, item_desc=item.description)
    emit_status("planner", f"🎯 Planner final review: {item.description}", **_progress_meta(state, "reviewing"))

    impl_label = _coder_label(state.active_coder)
    rev_label = _reviewer_label(state.active_reviewer)

    prompt = (
        f"## Planner Final Review\n\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implemented by**: {impl_label}\n"
        f"**Peer-reviewed by**: {rev_label} — APPROVED\n\n"
        f"**Coder's Report**:\n{state.last_coder_result}\n\n"
        f"**Peer Review Notes**:\n{state.peer_review_notes}\n\n"
        f"Final review:\n"
        f"1. Use `git status` and `git diff` to verify the changes.\n"
        f"2. Confirm the peer review didn't miss anything.\n"
        f"3. Verify: minimal diff, clean architecture, tests present, docs updated.\n"
        f"4. Give your verdict: APPROVE or REWORK.\n"
        f"5. If APPROVE, confirm or adjust the suggested Conventional Commit message."
    )

    result, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=PLANNER_TOOLS, inject_memory=False, node="planner",
    )
    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"

    emit_verdict("planner", verdict, detail=result, item_id=item.id)

    if verdict == "REWORK":
        item.rework_count += 1
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS
        if item.rework_count >= get_settings().max_rework_cycles_per_item:
            emit_status(
                "planner",
                "🔄 Planner forcing objective tester gate after repeated rework cycles",
                **_progress_meta(state, "testing"),
            )
            phase = WorkflowPhase.TESTING
        else:
            emit_status("planner", f"Planner REWORK - sending back to {impl_label}", **_progress_meta(state, "coding"))
            phase = WorkflowPhase.CODING
    else:
        emit_status("planner", "✅ Planner APPROVED - sending to Tester", **_progress_meta(state, "testing"))
        phase = WorkflowPhase.TESTING

    emit_node_end("planner", "Final Review", f"Verdict: {verdict}")

    return {
        "last_review_verdict": verdict,
        "review_notes": result,
        "phase": phase,
    }


def planner_env_fix_node(state: GraphState) -> dict:
    """Create a single env-setup fix item and prepend it to the plan.

    Called when tester_node detects an environment failure (missing package,
    command not found, etc.). This node:
    1. Reads the tester's failure output from last_test_result
    2. Uses an LLM to identify the exact missing package/tool
    3. Creates one TodoItem: install the dependency or update config
    4. Prepends it to todo_items so the coder handles it next
    5. Routes to coder (skipping peer review — it's a mechanical fix)

    Does NOT re-plan the entire task. Only creates the minimal fix item.
    """
    emit_node_start("planner", "EnvFix", item_desc="Diagnosing environment failure")
    emit_status(
        "planner",
        "🔧 Planner: diagnosing missing dependency and creating fix item…",
        **_progress_meta(state, "env_fixing"),
    )

    failure_output = state.last_test_result or "(no output captured)"
    repo_facts_summary = ""
    if state.repo_facts:
        rf = state.repo_facts
        lang = rf.get("tech_stack", {}).get("language", "unknown")
        pm = rf.get("tech_stack", {}).get("package_manager", "unknown")
        repo_facts_summary = f"Language: {lang}, Package manager: {pm}"

    prompt = (
        "## Environment Fix Task\n\n"
        "The test runner failed due to a missing dependency or broken environment. "
        "Your job is to create a single fix TodoItem to install or configure what is missing.\n\n"
        f"**Test runner output (failure)**:\n```\n{failure_output[:3000]}\n```\n\n"
        f"**Repository info**: {repo_facts_summary or 'unknown'}\n\n"
        "Instructions:\n"
        "1. Identify the exact missing package, module, or tool from the output above.\n"
        "2. Determine the correct install command (pip install X, npm install X, "
        "poetry add X, update pyproject.toml, etc.).\n"
        "3. Respond with ONLY a JSON object in this exact format:\n"
        "```json\n"
        '{"description": "Install missing dependency X", '
        '"command": "pip install X", '
        '"reason": "ModuleNotFoundError: No module named X"}\n'
        "```\n"
        "Do not add explanation outside the JSON block."
    )

    raw, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=None, inject_memory=False, node="env_fix",
    )

    # Parse JSON from LLM response
    fix_description = "Install missing test dependency"
    fix_command = ""
    try:
        import json as _json
        # Extract JSON block from response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = _json.loads(raw[json_start:json_end])
            fix_description = parsed.get("description", fix_description)
            fix_command = parsed.get("command", "")
    except Exception:
        logger.warning("planner_env_fix | could not parse LLM JSON response: %s", raw[:200])

    # Create the fix TodoItem with a unique id
    fix_item = TodoItem(
        id=f"env_fix_{state.env_fix_attempts + 1}",
        description=fix_description,
        task_type="ops",
        acceptance_criteria=["Test environment is functional and tests can run"],
        verification_commands=[fix_command] if fix_command else [],
    )

    # Assign the same coder that was active — env fixes skip peer review
    fix_item.assigned_agent = state.active_coder
    fix_item.assigned_reviewer = ""  # no review for mechanical env fixes

    # Prepend fix item before the current item (so coder handles it next)
    current_idx = state.current_item_index
    new_items = (
        list(state.todo_items[:current_idx])
        + [fix_item]
        + list(state.todo_items[current_idx:])
    )

    emit_status(
        "planner",
        f"🔧 Env fix item created: {fix_description}",
        **_progress_meta(state, "env_fixing"),
    )
    emit_node_end("planner", "EnvFix", f"Fix item queued: {fix_description}")

    return {
        "todo_items": new_items,
        "current_item_index": current_idx,  # points to fix_item now
        "env_fix_attempts": state.env_fix_attempts + 1,
        "phase": WorkflowPhase.CODING,
        # Skip peer review for env-fix items — coder goes straight back to tester
        "peer_review_verdict": "APPROVE",
        "peer_review_notes": "Auto-approved: environment fix item, no peer review needed.",
        **budget_update,
    }




def planner_decide_node(state: GraphState) -> dict:
    """Mark item done, prepare commit."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to decide on")
        return {"error_message": "No item to decide on", "phase": WorkflowPhase.STOPPED}

    item_num = state.current_item_index + 1
    total = len(state.todo_items)

    item.status = ItemStatus.DONE
    commit_msg = _extract_commit_message(state.peer_review_notes, state.review_notes, item.description)
    item.commit_message = commit_msg
    next_done = state.completed_items + 1

    emit_status(
        "planner",
        f"✅ Item {item_num}/{total} DONE: {item.description}",
        **_progress_meta(state, "deciding", done_override=next_done),
    )
    record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=True)

    try:
        _write_todo_file(state.todo_items, state.user_request)
    except Exception as e:
        logger.warning("Could not update todo.md: %s", e)

    return {
        "phase": WorkflowPhase.COMMITTING,
        "completed_items": next_done,
    }


