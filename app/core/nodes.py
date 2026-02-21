"""LangGraph node implementations — dual-coder workflow with shared memory.

All coder and reviewer nodes receive shared long-term memory before each call.
After each peer review a learning step extracts insights into memory files.
The planner compresses memory at session start if files are too large.
"""

from __future__ import annotations

import json
import platform
import re
from contextlib import suppress
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.config import get_settings
from app.core.events import (
    emit_agent_result,
    emit_agent_thinking,
    emit_commit,
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_plan,
    emit_status,
    emit_tool_call,
    emit_tool_result,
    emit_verdict,
)
from app.core.logging import get_logger
from app.core.memory import (
    LEARNING_EXTRACTION_PROMPT,
    append_memory,
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
    is_programming_request,
    record_agent_outcome,
    select_agent_thompson,
)
from app.tools.build import run_linter, run_tests
from app.tools.filesystem import list_directory, read_file, write_file
from app.tools.git import git_command, git_commit_and_push, git_create_branch, git_status
from app.tools.terminal import run_terminal

logger = get_logger("core.nodes")

CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\]\s*(?:Item\s+\d+:\s*)?(?P<desc>.+)$")
ROUTER_INTENTS = {"code", "status", "research", "resume"}


# -- Helper: invoke LLM with tools + event emission -----------------------

def _invoke_agent(role: str, messages: list, tools: list | None = None,
                  inject_memory: bool = False) -> str:
    """Invoke an LLM agent, handle tool calls, emit events.

    If inject_memory=True, the shared long-term memory is prepended to the
    system prompt so the agent can use established conventions.
    """
    llm = get_llm(role)
    system_prompt = load_system_prompt(role)

    # Inject shared memory into system prompt for coders/reviewers
    if inject_memory:
        memory_ctx = load_all_memory()
        if memory_ctx:
            system_prompt = system_prompt + "\n\n" + memory_ctx

    all_messages = [SystemMessage(content=system_prompt)] + messages

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    prompt_summary = messages[-1].content[:300] if messages else ""
    emit_agent_thinking(role, prompt_summary)

    max_tool_rounds = 15
    for _round_num in range(max_tool_rounds):
        response = llm_with_tools.invoke(all_messages)
        all_messages.append(response)

        if not response.tool_calls:
            result = response.content if isinstance(response.content, str) else str(response.content)
            emit_agent_result(role, result)
            return result

        tool_map = {t.name: t for t in (tools or [])}
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in tc["args"].items())
            emit_tool_call(role, tc["name"], args_str)

            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                    emit_error(role, f"Tool {tc['name']} failed: {e}")
            else:
                result = f"Unknown tool: {tc['name']}"

            emit_tool_result(role, tc["name"], str(result))
            logger.info("tool_call  | %s(%s) -> %d chars", tc["name"], list(tc["args"].keys()), len(str(result)))
            all_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    emit_error(role, "Exceeded maximum tool call rounds (15)")
    return "ERROR: Exceeded maximum tool call rounds."


# -- Tool sets -------------------------------------------------------------

PLANNER_TOOLS = [read_file, write_file, list_directory, git_status, run_terminal]

CODER_TOOLS = [
    read_file, write_file, list_directory,
    run_terminal, git_status, git_command,
    run_tests, run_linter,
]

REVIEWER_TOOLS = [read_file, list_directory, run_terminal, git_status, git_command, run_tests, run_linter]

TESTER_TOOLS = [read_file, list_directory, run_terminal, run_tests, run_linter, git_status]
DOCUMENTER_TOOLS = [read_file, write_file, list_directory, run_terminal, git_status, git_command]


# -- Helper: coder pair assignment -----------------------------------------

def _assign_coder_pair(item_index: int) -> tuple[str, str]:
    """Even items ? coder_a/reviewer_b. Odd items ? coder_b/reviewer_a."""
    if item_index % 2 == 0:
        return ("coder_a", "reviewer_b")
    else:
        return ("coder_b", "reviewer_a")


def _reviewer_for_worker(worker: str) -> str:
    if worker == "coder_a":
        return "reviewer_b"
    if worker == "coder_b":
        return "reviewer_a"
    return "reviewer_a"


def _coder_label(role: str) -> str:
    return {
        "coder_a": "Coder A (Claude)",
        "coder_b": "Coder B (GPT-5.2)",
        "documenter": "Documenter",
    }.get(role, role)

def _reviewer_label(role: str) -> str:
    return {"reviewer_a": "Reviewer A (Claude)", "reviewer_b": "Reviewer B (GPT-5.2)"}.get(role, role)


def _candidate_agents_for_task(task_type: str) -> list[str]:
    if task_type == "documentation":
        return ["documenter", "coder_a", "coder_b"]
    if task_type == "testing":
        return ["coder_b", "coder_a"]
    return ["coder_a", "coder_b"]


def _progress_meta(state: GraphState, phase: str, done_override: int | None = None) -> dict:
    total = len(state.todo_items)
    current_idx = state.current_item_index + 1 if state.current_item_index >= 0 else 0
    return {
        "phase": phase,
        "items_total": total,
        "items_done": state.completed_items if done_override is None else done_override,
        "current_index": current_idx,
        "branch": state.branch_name,
        "platform": state.execution_platform or platform.platform(),
    }


def _write_todo_file(items: list[TodoItem], user_request: str) -> None:
    lines = [f"## Plan: {user_request}", ""]
    for idx, item in enumerate(items, start=1):
        mark = "x" if item.status == ItemStatus.DONE else " "
        owner = _coder_label(item.assigned_agent or "coder_a")
        lines.append(f"- [{mark}] Item {idx}: {item.description}")
        lines.append(f"  - Type: {item.task_type}")
        lines.append(f"  - Owner: {owner}")
        for ac in item.acceptance_criteria:
            lines.append(f"  - AC: {ac}")
        for cmd in item.verification_commands:
            lines.append(f"  - Verify: `{cmd}`")
        lines.append("")
    write_file.invoke({"path": "tasks/todo.md", "content": "\n".join(lines).strip() + "\n"})


def _classify_request_intent(user_request: str) -> str:
    """Classify user input into planning intents."""
    text = (user_request or "").strip().lower()
    if not text:
        return "new_task"

    resume_markers = (
        "resume",
        "continue workflow",
        "continue task",
        "continue where",
        "fortsetzen",
        "weiterarbeiten",
        "weiter machen",
        "nach neustart",
        "wieder aufnehmen",
    )
    if any(marker in text for marker in resume_markers):
        return "resume_workflow"

    question_starters = (
        "was ",
        "wie ",
        "warum ",
        "wieso ",
        "welche ",
        "welcher ",
        "when ",
        "what ",
        "why ",
        "how ",
    )
    looks_like_question = text.endswith("?") or any(text.startswith(prefix) for prefix in question_starters)

    if looks_like_question and not is_programming_request(text):
        return "question_only"

    return "new_task"


def _heuristic_router_intent(user_request: str) -> str | None:
    """Fast intent heuristic for router gate."""
    text = (user_request or "").strip().lower()
    if not text:
        return "status"

    resume_markers = (
        "resume",
        "continue workflow",
        "continue task",
        "continue where",
        "fortsetzen",
        "weiterarbeiten",
        "weiter machen",
        "nach neustart",
        "wieder aufnehmen",
    )
    if any(marker in text for marker in resume_markers):
        return "resume"

    status_markers = (
        "status",
        "progress",
        "fortschritt",
        "current state",
        "aktueller stand",
        "wo stehen wir",
    )
    if any(marker in text for marker in status_markers):
        return "status"

    research_markers = (
        "research",
        "recherche",
        "investigate",
        "analyse",
        "analyze",
        "compare",
        "warum",
        "why",
    )
    if any(marker in text for marker in research_markers):
        return "research"

    if is_programming_request(text):
        return "code"

    return None


def _parse_router_json(result: str) -> tuple[str | None, float]:
    """Parse strict JSON router output and validate intent."""
    try:
        parsed = json.loads((result or "").strip())
    except json.JSONDecodeError:
        return None, 0.0

    if not isinstance(parsed, dict):
        return None, 0.0

    intent = str(parsed.get("intent", "")).strip().lower()
    if intent not in ROUTER_INTENTS:
        return None, 0.0

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence


def _owner_to_agent(owner: str) -> str:
    owner_text = owner.lower()
    if "documenter" in owner_text:
        return "documenter"
    if "coder a" in owner_text:
        return "coder_a"
    if "coder b" in owner_text:
        return "coder_b"
    return ""


def _parse_todo_for_resume(todo_text: str) -> list[TodoItem]:
    """Parse tasks/todo.md and recover structured todo items."""
    items: list[TodoItem] = []
    current: TodoItem | None = None

    for raw_line in todo_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        match = CHECKBOX_RE.match(stripped)
        if match:
            if current:
                items.append(current)
            mark = match.group("mark").lower()
            desc = match.group("desc").strip()
            status = ItemStatus.DONE if mark == "x" else ItemStatus.PENDING
            current = TodoItem(
                id=f"item-{len(items) + 1:03d}",
                description=desc,
                task_type=classify_task_type(desc),
                status=status,
            )
            continue

        if not current:
            continue

        if stripped.startswith("- Type:"):
            current.task_type = stripped.split(":", 1)[1].strip().lower() or current.task_type
        elif stripped.startswith("- Owner:"):
            current.assigned_agent = _owner_to_agent(stripped.split(":", 1)[1].strip())
            if current.assigned_agent:
                current.assigned_reviewer = _reviewer_for_worker(current.assigned_agent)
        elif stripped.startswith("- AC:"):
            current.acceptance_criteria.append(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- Verify:"):
            verify = stripped.split(":", 1)[1].strip().strip("`")
            if verify:
                current.verification_commands.append(verify)

    if current:
        items.append(current)
    return items


def _resume_from_saved_todo(state: GraphState) -> dict | None:
    """Resume a previously planned workflow from tasks/todo.md."""
    todo_raw = read_file.invoke({"path": "tasks/todo.md"})
    if todo_raw.startswith("ERROR"):
        return None

    items = _parse_todo_for_resume(todo_raw)
    if not items:
        return None

    pending_indexes = [idx for idx, item in enumerate(items) if item.status != ItemStatus.DONE]
    if not pending_indexes:
        return None

    current_index = pending_indexes[0]
    completed = sum(1 for item in items if item.status == ItemStatus.DONE)
    current_item = items[current_index]

    if not current_item.assigned_agent:
        candidates = _candidate_agents_for_task(current_item.task_type)
        assigned_agent, _history = select_agent_thompson(state.repo_root, current_item.task_type, candidates)
        current_item.assigned_agent = assigned_agent
        current_item.assigned_reviewer = _reviewer_for_worker(assigned_agent)

    branch = state.branch_name
    if not branch:
        branch_result = git_command.invoke({"command": "git rev-parse --abbrev-ref HEAD"})
        for line in branch_result.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("[") and "stderr" not in candidate.lower():
                branch = candidate
                break

    emit_status(
        "planner",
        f"Resumed workflow at item {current_index + 1}/{len(items)}: {current_item.description}",
        phase="coding",
        items_total=len(items),
        items_done=completed,
        current_index=current_index + 1,
        branch=branch or "",
        platform=state.execution_platform or platform.platform(),
    )

    return {
        "todo_items": items,
        "current_item_index": current_index,
        "completed_items": completed,
        "branch_name": branch or "",
        "phase": WorkflowPhase.CODING,
        "active_coder": current_item.assigned_agent or "coder_a",
        "active_reviewer": (
            current_item.assigned_reviewer
            or _reviewer_for_worker(current_item.assigned_agent or "coder_a")
        ),
    }


def _answer_question_directly(state: GraphState) -> str:
    """Answer non-task questions directly via planner without coder handoff."""
    prompt = (
        "You are the planner. The user asked a question, not a coding task.\n"
        "Answer directly and clearly. Do not create a plan. Do not modify files.\n\n"
        f"Question:\n{state.user_request}\n"
    )
    return _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)


# ---------------------------------------------------------------------------
# NODE: router
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> dict:
    """Intent gate before any planning/coding workflow starts."""
    emit_node_start("planner", "Router", item_desc=state.user_request[:100])
    emit_status("planner", "Classifying request intent", **_progress_meta(state, "planning"))

    heuristic_intent = _heuristic_router_intent(state.user_request)
    if heuristic_intent in ROUTER_INTENTS:
        emit_node_end("planner", "Router", f"Heuristic intent: {heuristic_intent}")
        return {"input_intent": heuristic_intent}

    router_prompt = (
        "Classify the user request into ONE intent only.\n"
        "Allowed intents: code, status, research, resume.\n"
        "Return STRICT JSON only, no markdown:\n"
        '{"intent":"code|status|research|resume","confidence":0.0}\n\n'
        f"User request:\n{state.user_request}\n"
    )
    llm_result = _invoke_agent("planner", [HumanMessage(content=router_prompt)])
    llm_intent, confidence = _parse_router_json(llm_result)

    if llm_intent:
        emit_node_end("planner", "Router", f"LLM intent: {llm_intent} (confidence={confidence:.2f})")
        return {"input_intent": llm_intent}

    fallback = "code" if is_programming_request(state.user_request or "") else "research"
    emit_status(
        "planner",
        f"Router fallback intent: {fallback} (LLM output not parseable JSON)",
        **_progress_meta(state, "planning"),
    )
    emit_node_end("planner", "Router", f"Fallback intent: {fallback}")
    return {"input_intent": fallback}


# ---------------------------------------------------------------------------
# NODE: context_loader (placeholder for Patch 02)
# ---------------------------------------------------------------------------

def context_loader_node(state: GraphState) -> dict:
    """Prepare context stage before planner (no mutation yet)."""
    emit_node_start("planner", "Context Loader", item_desc=state.user_request[:100])
    emit_status("planner", "Context pre-load step ready (placeholder)", **_progress_meta(state, "planning"))
    emit_node_end("planner", "Context Loader", "Proceeding to planner")
    return {"input_intent": "code"}


# ---------------------------------------------------------------------------
# NODE: status / research / resume (minimal non-coding branches)
# ---------------------------------------------------------------------------

def status_node(state: GraphState) -> dict:
    """Non-coding status response branch (no write/git tools)."""
    emit_node_start("planner", "Status", item_desc=state.user_request[:100])
    summary = state.get_progress_summary()
    message = f"Workflow status:\n{summary}"
    emit_status("planner", "Status request handled without coding", **_progress_meta(state, "complete"))
    emit_node_end("planner", "Status", "Status response prepared")
    return {
        "planner_response": message,
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "status_answered",
        "input_intent": "status",
    }


def research_node(state: GraphState) -> dict:
    """Research branch without repository mutation tools."""
    emit_node_start("planner", "Research", item_desc=state.user_request[:100])
    prompt = (
        "You are a research assistant for a software workflow.\n"
        "Answer the user's request in analysis mode only.\n"
        "Do not propose or perform code/file/git changes.\n\n"
        f"User request:\n{state.user_request}\n"
    )
    answer = _invoke_agent("planner", [HumanMessage(content=prompt)])
    emit_status("planner", "Research request handled without coding", **_progress_meta(state, "complete"))
    emit_node_end("planner", "Research", "Research response prepared")
    return {
        "planner_response": answer,
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "research_answered",
        "input_intent": "research",
    }


def resume_node(state: GraphState) -> dict:
    """Resume branch reusing existing todo-based resume logic."""
    emit_node_start("planner", "Resume", item_desc=state.user_request[:100])
    resumed = _resume_from_saved_todo(state)
    if resumed is not None:
        resumed["input_intent"] = "resume"
        emit_node_end("planner", "Resume", "Resumed workflow from tasks/todo.md")
        return resumed

    emit_status("planner", "No resumable TODO list found.", **_progress_meta(state, "complete"))
    emit_node_end("planner", "Resume", "Nothing to resume")
    return {
        "planner_response": "No resumable TODO items found in tasks/todo.md.",
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "resume_not_found",
        "input_intent": "resume",
    }


# ---------------------------------------------------------------------------
# NODE: planner_plan
# ---------------------------------------------------------------------------

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

    result = _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)
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

    emit_node_end("planner", "Planning", f"Plan ready - starting with item 1 -> {_coder_label(active_coder)}")
    return {
        "input_intent": intent,
        "todo_items": items if items else state.todo_items,
        "current_item_index": 0 if items else state.current_item_index,
        "branch_name": branch,
        "phase": WorkflowPhase.CODING,
        "needs_replan": False,
        "active_coder": active_coder,
        "active_reviewer": active_reviewer,
    }


def _compress_memory_file(key: str) -> None:
    """Use the planner LLM to compress an oversized memory file."""
    prompt = build_compression_prompt(key)
    if not prompt:
        return
    try:
        result = _invoke_agent("planner", [HumanMessage(content=prompt)])
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

def coder_node(state: GraphState) -> dict:
    """Dispatch to active worker (coder or documenter)."""
    item = state.current_item
    if not item:
        emit_error("system", "No current item to work on")
        return {"error_message": "No current item to work on", "phase": WorkflowPhase.STOPPED}

    active = state.active_coder
    reviewer = state.active_reviewer
    item_num = state.current_item_index + 1
    total = len(state.todo_items)

    emit_node_start(active, "Coding", item_id=item.id, item_desc=item.description)
    emit_status(
        active,
        f"[{item_num}/{total}] {_coder_label(active)} implementing: {item.description}",
        iteration=item.iteration_count + 1,
        **_progress_meta(state, "coding"),
    )

    item.status = ItemStatus.IN_PROGRESS
    item.iteration_count += 1

    settings = get_settings()
    if item.iteration_count > settings.max_iterations_per_item:
        msg = f"Item {item.id} exceeded max iterations ({settings.max_iterations_per_item})"
        emit_error(active, msg)
        return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED}

    prompt_parts = [
        f"## Task Assignment — {_coder_label(active)}",
        f"**Item ID**: {item.id}",
        f"**Task Type**: {item.task_type}",
        f"**Description**: {item.description}",
        f"**Your peer reviewer**: {_reviewer_label(reviewer)}",
        f"**Execution platform**: {state.execution_platform or platform.platform()}",
        "**OS Note**: Runtime is Windows; terminal commands use PowerShell.",
    ]
    if item.acceptance_criteria:
        prompt_parts.append("**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria))
    if item.verification_commands:
        prompt_parts.append(
            "**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands)
        )
    if item.review_notes:
        prompt_parts.append(f"**Rework Notes (from previous review)**:\n{item.review_notes}")
    if state.peer_review_notes and state.peer_review_verdict == "REWORK":
        prompt_parts.append(f"**Peer Review Feedback (REWORK)**:\n{state.peer_review_notes}")
    if item.test_report:
        prompt_parts.append(f"**Test Report (previous)**:\n{item.test_report}")

    prompt_parts.append(
        "\nImplement this task. Use tools to read the codebase, make changes, "
        "add tests where needed, and update docs. Keep diffs minimal.\n"
        "Follow the coding style and architecture decisions from shared memory."
    )

    tools = DOCUMENTER_TOOLS if active == "documenter" else CODER_TOOLS
    result = _invoke_agent(active, [HumanMessage(content="\n\n".join(prompt_parts))], tools, inject_memory=True)

    emit_node_end(active, "Coding", f"Implementation complete — handing to {_reviewer_label(reviewer)} for peer review")
    with suppress(Exception):
        _write_todo_file(state.todo_items, state.user_request)

    return {
        "last_coder_result": result,
        "phase": WorkflowPhase.PEER_REVIEWING,
        "total_iterations": state.total_iterations + 1,
    }


# ---------------------------------------------------------------------------
# NODE: peer_review  (reads shared memory)
# ---------------------------------------------------------------------------

def peer_review_node(state: GraphState) -> dict:
    """Cross-coder peer review with loop-guard escalation."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to peer-review")
        return {"error_message": "No item to peer-review", "phase": WorkflowPhase.STOPPED}

    reviewer = state.active_reviewer
    implementer = state.active_coder
    impl_label = _coder_label(implementer)
    rev_label = _reviewer_label(reviewer)

    emit_node_start(reviewer, "Peer Review", item_id=item.id, item_desc=item.description)
    emit_status(
        reviewer,
        f"{rev_label} reviewing {impl_label}: {item.description}",
        **_progress_meta(state, "peer_reviewing"),
    )

    item.status = ItemStatus.IN_REVIEW

    prompt = (
        f"## Peer Code Review\n\n"
        f"**Reviewer**: {rev_label}\n"
        f"**Implementer**: {impl_label}\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implementer's Report**:\n{state.last_coder_result}\n\n"
        f"Review the changes. Also verify consistency with the shared memory "
        f"(coding style, architecture decisions, insights):\n"
        f"1. Use `git_command` with `git diff` to see the actual changes.\n"
        f"2. Use `git_command` with `git status` to see which files changed.\n"
        f"3. Read the modified files to understand the full context.\n"
        f"4. Run the test suite and linter to check for regressions.\n"
        f"5. Verify: correct logic, minimal diff, clean architecture, tests added, docs updated.\n"
        f"6. Check: does the code follow established patterns from shared memory?\n"
        f"7. Give your verdict: APPROVE or REWORK.\n"
        f"8. If REWORK, provide specific actionable notes.\n"
        f"9. If APPROVE, suggest a Conventional Commit message.\n"
        f"10. Note any NEW patterns, conventions, or insights discovered during review\n"
        f"    (these will be added to shared memory for future tasks)."
    )

    # inject_memory=True ? reviewer sees established conventions
    result = _invoke_agent(reviewer, [HumanMessage(content=prompt)],
                           REVIEWER_TOOLS, inject_memory=True)

    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"
    if "**Verdict**: REWORK" in result or "Verdict: REWORK" in result:
        verdict = "REWORK"
    elif "**Verdict**: APPROVE" in result or "Verdict: APPROVE" in result:
        verdict = "APPROVE"

    emit_verdict(reviewer, verdict, detail=result, item_id=item.id)
    settings = get_settings()

    if verdict == "REWORK":
        item.rework_count += 1
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS
        record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=False)
        if item.rework_count >= settings.max_rework_cycles_per_item:
            emit_status(
                reviewer,
                f"Review loop threshold reached ({item.rework_count}). Escalating to tester gate.",
                **_progress_meta(state, "testing"),
            )
            phase = WorkflowPhase.TESTING
            verdict = "ESCALATE_TESTING"
        else:
            emit_status(reviewer, f"🔄 Peer review REWORK - back to {impl_label}", **_progress_meta(state, "coding"))
            phase = WorkflowPhase.CODING
    else:
        emit_status(reviewer, "✅ Peer review APPROVED - extracting learnings", **_progress_meta(state, "reviewing"))
        phase = WorkflowPhase.REVIEWING

    emit_node_end(reviewer, "Peer Review", f"Verdict: {verdict}")

    return {
        "peer_review_verdict": verdict,
        "peer_review_notes": result,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# NODE: learn_from_review  (extracts insights ? memory files)
# ---------------------------------------------------------------------------

def learn_from_review_node(state: GraphState) -> dict:
    """Extract learnings from the peer review and append to shared memory.

    Runs after peer review (regardless of APPROVE/REWORK) and before planner review.
    Uses a cheap LLM call (planner model) to extract structured insights.
    """
    item = state.current_item
    if not item:
        return {}

    review_text = state.peer_review_notes
    if not review_text:
        return {}

    emit_node_start("system", "Learning", item_id=item.id, item_desc="Extracting insights from review")

    prompt = (
        f"{LEARNING_EXTRACTION_PROMPT}\n\n"
        f"---\n\n"
        f"## Review to Analyze\n"
        f"**Item**: {item.id} — {item.description}\n"
        f"**Verdict**: {state.peer_review_verdict}\n\n"
        f"**Review Text**:\n{review_text}\n"
    )

    try:
        result = _invoke_agent("planner", [HumanMessage(content=prompt)])

        # Parse JSON from result
        learnings = _parse_learnings(result)
        total_added = 0

        for key in ["coding_style", "architecture", "insights"]:
            entries = learnings.get(key, [])
            for entry in entries:
                if entry and len(entry.strip()) > 5:
                    append_memory(key, entry.strip(), item_id=item.id)
                    total_added += 1

        if total_added > 0:
            emit_status("system",
                        f"🧠 Learned {total_added} new insight(s) from peer review of {item.id}",
                        **_progress_meta(state, "learning"))
            logger.info("Learned %d insights from review of %s", total_added, item.id)
        else:
            emit_status("system",
                        f"🧠 No new generalizable insights from review of {item.id}",
                        **_progress_meta(state, "learning"))

    except Exception as e:
        logger.warning("Learning extraction failed: %s", e)
        emit_status("system", f"⚠ Learning extraction skipped: {e}", **_progress_meta(state, "learning"))

    emit_node_end("system", "Learning")

    # Don't change phase — we continue to wherever peer_review set us
    return {}


def _parse_learnings(result: str) -> dict:
    """Extract JSON from LLM result. Handles markdown fences and partial output."""
    text = result.strip()
    # Remove markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse learnings JSON: %s", text[:200])
    return {}


# ---------------------------------------------------------------------------
# NODE: planner_review
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

    result = _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)
    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"

    emit_verdict("planner", verdict, detail=result, item_id=item.id)

    if verdict == "REWORK":
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


# ---------------------------------------------------------------------------
# NODE: tester
# ---------------------------------------------------------------------------

def tester_node(state: GraphState) -> dict:
    """Run tests and verification."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to test")
        return {"error_message": "No item to test", "phase": WorkflowPhase.STOPPED}

    emit_node_start("tester", "Testing", item_id=item.id, item_desc=item.description)
    emit_status("tester", f"🧪 Running tests for: {item.description}", **_progress_meta(state, "testing"))

    item.status = ItemStatus.TESTING

    prompt = (
        f"## Verification Task\n\n"
        f"**Item**: {item.id} — {item.description}\n"
        f"**Execution platform**: {state.execution_platform or platform.platform()}\n"
        f"**OS Note**: Runtime is Windows and terminal calls use PowerShell.\n"
    )
    if item.acceptance_criteria:
        prompt += "**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria) + "\n"
    if item.verification_commands:
        prompt += "**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands) + "\n"
    prompt += (
        "\nRun all tests, linters, and verification commands. "
        "Produce a structured test report with PASS or FAIL verdict."
    )

    result = _invoke_agent("tester", [HumanMessage(content=prompt)], TESTER_TOOLS)

    verdict = "PASS" if "PASS" in result.upper() and "FAIL" not in result.upper() else "FAIL"
    if "**Verdict**: PASS" in result or "Verdict: PASS" in result:
        verdict = "PASS"
    elif "**Verdict**: FAIL" in result or "Verdict: FAIL" in result:
        verdict = "FAIL"

    item.test_report = result

    emit_verdict("tester", verdict, detail=result, item_id=item.id)

    if verdict == "FAIL":
        item.test_fail_count += 1
        item.status = ItemStatus.IN_PROGRESS
        record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=False)
        if item.test_fail_count >= get_settings().max_rework_cycles_per_item:
            msg = f"Item {item.id} failed tests {item.test_fail_count} times; stopping to avoid loop."
            emit_error("tester", msg)
            return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED}
        emit_status(
            "tester",
            f"❌ Tests FAILED - sending back to {_coder_label(state.active_coder)}",
            **_progress_meta(state, "coding"),
        )
    else:
        emit_status("tester", "✅ All tests PASSED", **_progress_meta(state, "deciding"))

    emit_node_end("tester", "Testing", f"Verdict: {verdict}")

    return {
        "last_test_result": result,
        "phase": WorkflowPhase.DECIDING if verdict == "PASS" else WorkflowPhase.CODING,
    }


# ---------------------------------------------------------------------------
# NODE: planner_decide
# ---------------------------------------------------------------------------

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


def _extract_commit_message(peer_notes: str, planner_notes: str, fallback_desc: str) -> str:
    for source in [planner_notes, peer_notes]:
        for line in source.split("\n"):
            stripped = line.strip()
            for prefix in ["Suggested commit:", "Commit message:", "Commit:", "Suggested Conventional Commit message:"]:
                if prefix.lower() in stripped.lower():
                    stripped = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
                    break
            stripped = stripped.strip("`").strip('"').strip("'").strip()
            if any(stripped.startswith(p) for p in ["feat(", "fix(", "docs:", "test:", "refactor(", "chore("]):
                return stripped
    return f"feat: {fallback_desc[:50].lower()}"


# ---------------------------------------------------------------------------
# NODE: committer
# ---------------------------------------------------------------------------

def committer_node(state: GraphState) -> dict:
    """Commit, push, and advance to next item."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to commit")
        return {"error_message": "No item to commit", "phase": WorkflowPhase.STOPPED}

    emit_status("system", f"📦 Committing: {item.commit_message}", **_progress_meta(state, "committing"))

    result = git_commit_and_push.invoke({"message": item.commit_message, "push": True})
    emit_commit(item.commit_message, item_id=item.id)
    logger.info("commit result: %s", result[:200])

    next_index = state.current_item_index + 1
    has_more = next_index < len(state.todo_items)

    if has_more:
        next_item = state.todo_items[next_index]
        next_coder = next_item.assigned_agent or _assign_coder_pair(next_index)[0]
        next_reviewer = next_item.assigned_reviewer or _reviewer_for_worker(next_coder)
        emit_status(
            "planner",
            f"Moving to item {next_index + 1}/{len(state.todo_items)}: "
            f"{next_item.description} -> {_coder_label(next_coder)}",
            **_progress_meta(state, "coding"),
        )
        return {
            "current_item_index": next_index,
            "phase": WorkflowPhase.CODING,
            "active_coder": next_coder,
            "active_reviewer": next_reviewer,
            "peer_review_notes": "",
            "peer_review_verdict": "",
        }
    else:
        # Log final memory stats
        stats = get_memory_stats()
        total = sum(s["chars"] for s in stats.values())
        emit_status(
            "planner",
            f"🧠 Final memory: {total} chars across {len(stats)} files",
            **_progress_meta(state, "complete"),
        )
        emit_status(
            "planner",
            f"🎉 All {len(state.todo_items)} items completed! Branch: {state.branch_name}",
            **_progress_meta(state, "complete"),
        )
        return {"phase": WorkflowPhase.COMPLETE}
