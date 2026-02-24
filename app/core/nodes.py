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
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.checkpoints import checkpoint_manager
from app.core.config import get_settings
from app.core.events import (
    emit_agent_response,
    emit_agent_result,
    emit_agent_thinking,
    emit_agent_token,
    emit_approval_needed,
    emit_coder_question,
    emit_commit,
    emit_context_usage,
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_plan,
    emit_status,
    emit_token_usage,
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
from app.core.token_budget import (
    BudgetExceededException,
    TokenBudget,
    TokenUsageRecord,
    calculate_cost,
    extract_token_usage,
)
from app.core.context_window import (
    CONTEXT_WARN_FRACTION,
    compress_messages,
    context_limit_for_model,
    context_usage_fraction,
    estimate_messages_tokens,
    truncate_tool_result,
)
from app.tools.build import run_linter, run_tests
from app.tools.filesystem import list_directory, read_file, write_file
from app.tools.git import git_command, git_commit_and_push, git_create_branch, git_status
from app.tools.search import search_in_repo
from app.tools.terminal import run_terminal

logger = get_logger("core.nodes")

CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\]\s*(?:Item\s+\d+:\s*)?(?P<desc>.+)$")


def _model_name_for_role(role: str) -> str:
    """Return the configured model name for a given agent role."""
    settings = get_settings()
    mapping = {
        "planner":    settings.planner_model,
        "coder_a":    settings.coder_1_model,
        "coder_b":    settings.coder_2_model,
        "reviewer_a": settings.coder_1_model,
        "reviewer_b": settings.coder_2_model,
        "documenter": settings.documenter_model,
        "tester":     settings.tester_model,
    }
    return mapping.get(role, settings.planner_model)
ROUTER_INTENTS = {"code", "status", "research", "resume"}


# -- Helper: parse ask_human signal from coder response -------------------

def _parse_coder_question(response: str) -> dict | None:
    """Return the parsed ask_human payload if the coder asked a question, else None.

    The coder signals a question by returning a JSON object whose top-level key
    ``"action"`` equals ``"ask_human"``.  The response must be *only* that JSON
    (possibly wrapped in a single markdown code fence) — any preamble text
    disqualifies it to avoid false positives.

    Returns a dict with keys: question, context, options (may be empty list).
    """
    text = response.strip()

    # Strip optional ```json / ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence line and closing fence line
        inner = "\n".join(
            line for line in lines[1:]
            if not line.strip().startswith("```")
        )
        text = inner.strip()

    # Must look like a JSON object to avoid expensive parsing on normal output
    if not text.startswith("{"):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("action") != "ask_human":
        return None

    question = str(payload.get("question", "")).strip()
    if not question:
        return None

    return {
        "question": question,
        "context": str(payload.get("context", "")).strip(),
        "options": [str(o) for o in payload.get("options", []) if o],
    }


# -- Helper: invoke LLM with tools + event emission -----------------------

# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

# Roles that emit streaming tokens to the UI (coders, planner, reviewers).
# Tester and router are short calls where streaming adds no UX value.
_STREAMING_ROLES: frozenset[str] = frozenset({
    "coder_a", "coder_b", "reviewer_a", "reviewer_b", "planner", "documenter"
})

# Minimum token batch size before emitting a streaming event.
# Batching avoids flooding the event bus on slow models.
_TOKEN_BATCH_MIN = 8


def _stream_llm_round(
    role: str,
    llm_with_tools: object,
    all_messages: list,
) -> "AIMessage":
    """Stream one LLM round, emitting token events, and return a full AIMessage.

    Falls back to .invoke() if the model does not support streaming or if
    streaming raises an exception.
    """
    from langchain_core.messages import AIMessage, AIMessageChunk

    accumulated_text = ""
    accumulated_chunk: AIMessageChunk | None = None
    batch: list[str] = []

    def _flush_batch() -> None:
        nonlocal batch
        if batch:
            emit_agent_token(role, "".join(batch))
            batch = []

    try:
        for chunk in llm_with_tools.stream(all_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue

            # Accumulate for final reconstruction
            accumulated_chunk = chunk if accumulated_chunk is None else accumulated_chunk + chunk

            # Extract text content
            text = ""
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                    elif isinstance(part, str):
                        text += part

            if text:
                accumulated_text += text
                batch.append(text)
                if len(batch) >= _TOKEN_BATCH_MIN:
                    _flush_batch()

        _flush_batch()

        # Reconstruct a proper AIMessage from the accumulated chunk
        if accumulated_chunk is not None:
            return AIMessage(
                content=accumulated_chunk.content,
                tool_calls=list(accumulated_chunk.tool_calls) if accumulated_chunk.tool_calls else [],
                id=getattr(accumulated_chunk, "id", None),
            )
        return AIMessage(content=accumulated_text)

    except NotImplementedError:
        # Model doesn't support streaming — fall back silently
        logger.debug("streaming_fallback | %s does not support .stream()", role)
        return llm_with_tools.invoke(all_messages)
    except Exception as exc:
        logger.warning("streaming_error | %s — %s — falling back to .invoke()", role, exc)
        return llm_with_tools.invoke(all_messages)


def _invoke_agent(role: str, messages: list, tools: list | None = None,
                  inject_memory: bool = False,
                  budget: TokenBudget | None = None,
                  node: str = "") -> str:
    """Invoke an LLM agent, handle tool calls, emit events.

    Streaming is enabled automatically for roles in _STREAMING_ROLES.
    All other roles use .invoke() (blocking). Falls back to .invoke()
    gracefully if the model does not support streaming.

    If inject_memory=True, the shared long-term memory is prepended to the
    system prompt so the agent can use established conventions.

    If budget is provided, token usage is tracked after each LLM response
    and a BudgetExceededException is raised when the hard limit is hit.

    Context window management (always active):
    - Tool results are truncated to settings.tool_result_max_chars before
      being appended to all_messages (preventive).
    - After each LLM response, context usage is estimated. If it exceeds
      settings.context_warn_fraction of the model limit, old turns are
      compressed via an LLM summary call (reactive).
    """
    llm = get_llm(role)
    model_name = _model_name_for_role(role)
    system_prompt = load_system_prompt(role)
    settings = get_settings()

    # Inject shared memory into system prompt for coders/reviewers
    if inject_memory:
        memory_ctx = load_all_memory()
        if memory_ctx:
            system_prompt = system_prompt + "\n\n" + memory_ctx

    all_messages = [SystemMessage(content=system_prompt)] + messages

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    use_streaming = role in _STREAMING_ROLES

    prompt_summary = messages[-1].content[:300] if messages else ""
    emit_agent_thinking(role, prompt_summary)

    # -- (a) Warn if initial context is already large ----------------------
    initial_fraction = context_usage_fraction(all_messages, model_name)
    if initial_fraction > 0.5:
        emit_status(
            "system",
            f"ℹ️ Initial context at {initial_fraction:.0%} of {model_name} limit",
        )

    max_tool_rounds = 15
    for _round_num in range(max_tool_rounds):
        if use_streaming:
            response = _stream_llm_round(role, llm_with_tools, all_messages)
        else:
            response = llm_with_tools.invoke(all_messages)

        # -- Token tracking ------------------------------------------------
        if budget is not None:
            usage = extract_token_usage(response)
            cost = calculate_cost(model_name, usage["prompt_tokens"], usage["completion_tokens"])
            record = TokenUsageRecord(
                agent=role,
                model=model_name,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                cost_usd=cost,
                node=node,
            )
            try:
                budget.add(record)
                emit_token_usage(
                    agent=role,
                    model=model_name,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    cost_usd=cost,
                    total_cost_usd=budget.total_cost_usd,
                )
                if budget.soft_limit_hit and usage["prompt_tokens"] > 0:
                    # Emit soft-limit warning exactly once (flag stays True)
                    emit_status(
                        "system",
                        f"⚠️ Token budget soft limit reached: "
                        f"${budget.total_cost_usd:.4f} >= ${budget.soft_limit_usd:.2f}. "
                        "Workflow continues.",
                    )
            except BudgetExceededException as exc:
                emit_error(
                    "system",
                    f"❌ Hard budget limit exceeded: ${exc.total_cost:.4f} >= ${exc.limit:.2f}. "
                    "Stopping workflow.",
                )
                raise
        # ------------------------------------------------------------------

        all_messages.append(response)

        # -- (c) Context check + compression after each LLM turn -----------
        ctx_limit    = context_limit_for_model(model_name)
        ctx_tokens   = estimate_messages_tokens(all_messages)
        ctx_fraction = ctx_tokens / ctx_limit
        emit_context_usage(role, ctx_tokens, ctx_limit, ctx_fraction)

        warn_fraction = settings.context_warn_fraction
        if warn_fraction > 0 and ctx_fraction >= warn_fraction:
            emit_status(
                "system",
                f"⚠️ Context at {ctx_fraction:.0%} ({ctx_tokens:,} / {ctx_limit:,} tok)"
                " — compressing old turns",
            )
            all_messages = compress_messages(all_messages, model_name, llm)
            new_tokens   = estimate_messages_tokens(all_messages)
            new_fraction = new_tokens / ctx_limit
            emit_context_usage(role, new_tokens, ctx_limit, new_fraction, compressed=True)
            emit_status(
                "system",
                f"✅ Context compressed: {ctx_tokens:,} → {new_tokens:,} tok"
                f" ({new_fraction:.0%})",
            )
        # ------------------------------------------------------------------

        if not response.tool_calls:
            result = response.content if isinstance(response.content, str) else str(response.content)
            emit_agent_result(role, result)
            return result

        # After tool calls the next LLM turn may stream again — reset flag
        # so partial streaming blocks are visually separated.
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

            # -- (b) Truncate tool result before entering all_messages -----
            safe_result = truncate_tool_result(
                str(result), max_chars=settings.tool_result_max_chars
            )
            all_messages.append(ToolMessage(content=safe_result, tool_call_id=tc["id"]))

    emit_error(role, "Exceeded maximum tool call rounds (15)")
    return "ERROR: Exceeded maximum tool call rounds."


# -- Tool sets -------------------------------------------------------------

PLANNER_TOOLS = [read_file, write_file, list_directory, search_in_repo, git_status, run_terminal]

CODER_TOOLS = [
    read_file, write_file, list_directory,
    search_in_repo,
    run_terminal, git_status, git_command,
    run_tests, run_linter,
]

REVIEWER_TOOLS = [
    read_file,
    list_directory,
    search_in_repo,
    run_terminal,
    git_status,
    git_command,
    run_tests,
    run_linter,
]

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
        "coder_a": "Coder 1",
        "coder_b": "Coder 2",
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


def _get_budget(state: GraphState) -> TokenBudget:
    """Reconstruct the live TokenBudget from the serialised state dict.

    Also applies the configured soft/hard limits from settings so they are
    always up-to-date even after a checkpoint restore.
    """
    settings = get_settings()
    budget = TokenBudget.from_dict(state.token_budget) if state.token_budget else TokenBudget()
    budget.soft_limit_usd = settings.token_budget_soft_limit_usd
    budget.hard_limit_usd = settings.token_budget_hard_limit_usd
    return budget


def _budget_dict(budget: TokenBudget) -> dict:
    """Serialise budget back to a plain dict for GraphState storage."""
    return budget.to_dict()


def _invoke_with_budget(
    state: GraphState,
    role: str,
    messages: list,
    tools: list | None = None,
    inject_memory: bool = False,
    node: str = "",
) -> tuple[str, dict]:
    """Call _invoke_agent with budget tracking.

    Returns (result_str, {"token_budget": updated_dict}).
    The caller merges the second element into its return dict so the
    GraphState budget accumulates across nodes.

    On BudgetExceededException the node should stop the workflow.
    """
    budget = _get_budget(state)
    result = _invoke_agent(
        role, messages, tools,
        inject_memory=inject_memory,
        budget=budget,
        node=node,
    )
    return result, {"token_budget": _budget_dict(budget)}


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


def _save_checkpoint_snapshot(state: GraphState, updates: dict, checkpoint_type: str) -> None:
    """Persist a checkpoint based on current state plus node updates."""
    try:
        merged = {**state.model_dump(), **updates}
        snapshot = GraphState(**merged)
        checkpoint_manager.save_checkpoint(snapshot, checkpoint_type, repo_root=snapshot.repo_root)
    except Exception as exc:
        logger.warning("Checkpoint save failed (%s): %s", checkpoint_type, exc)


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

    _router_prompt_file = Path(__file__).parent.parent / "agents" / "prompts" / "router.txt"
    if _router_prompt_file.exists():
        _router_system = _router_prompt_file.read_text(encoding="utf-8")
    else:
        _router_system = (
            "Classify the user request into ONE intent only.\n"
            "Allowed intents: code, status, research, resume.\n"
            "Return STRICT JSON only, no markdown:\n"
            '{"intent":"code|status|research|resume","confidence":0.0}\n'
        )
    llm_result = get_llm("planner").invoke(
        [SystemMessage(content=_router_system), HumanMessage(content=f"User request:\n{state.user_request}")]
    ).content
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
    """Load repository context before planner execution."""
    emit_node_start("planner", "Context Loader", item_desc=state.user_request[:100])
    if state.context_loaded:
        emit_status("planner", "Context already loaded; skipping re-analysis", **_progress_meta(state, "planning"))
        emit_node_end("planner", "Context Loader", "Skipped (already loaded)")
        return {"input_intent": "code", "context_loaded": True}

    settings = get_settings()
    repo_root = (state.repo_root or settings.target_repo_path or "").strip()
    if not repo_root:
        emit_error("planner", "Context loader could not determine repository path.")
        emit_node_end("planner", "Context Loader", "Failed (repo path missing)")
        return {
            "repo_facts": {"error": "Missing repository path", "fallback": True},
            "context_loaded": False,
            "stop_reason": "context_repo_path_missing",
            "phase": WorkflowPhase.STOPPED,
        }

    repo_path = Path(repo_root).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        emit_error("planner", f"Context loader repository path invalid: {repo_path}")
        emit_node_end("planner", "Context Loader", "Failed (repo path invalid)")
        return {
            "repo_facts": {"error": f"Invalid repository path: {repo_path}", "fallback": True},
            "context_loaded": False,
            "stop_reason": "context_repo_path_invalid",
            "phase": WorkflowPhase.STOPPED,
        }

    emit_status("planner", "Reading repository documentation and structure", **_progress_meta(state, "planning"))

    # Daedalus' own root directory — determined once at import time.
    # AGENT.md files are ONLY read from the target repo, never from Daedalus itself.
    # This prevents Daedalus' own build-spec (AGENT.md) from leaking into tasks
    # targeting unrelated repositories.
    _daedalus_root = Path(__file__).parent.parent.parent.resolve()
    _is_self_referential = repo_path.resolve() == _daedalus_root

    doc_files = [
        "docs/AGENT.md",
        "AGENT.md",
        "AGENTS.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "CONTRIBUTING.rst",
        "README.md",
    ]

    # AGENT.md files are intentionally excluded when the target repo IS Daedalus itself.
    # When working on Daedalus, TARGET_REPO_PATH must point to a separate clone/copy —
    # in that case the copy's own AGENT.md will be read normally.
    _agent_md_files = {"docs/AGENT.md", "AGENT.md", "AGENTS.md"}

    max_chars = max(1000, int(settings.max_output_chars))
    instruction_chunks: list[str] = []
    for rel_path in doc_files:
        if _is_self_referential and rel_path in _agent_md_files:
            logger.info(
                "Context loader: skipping %s — target repo is Daedalus root. "
                "Set TARGET_REPO_PATH to a separate clone to enable self-improvement mode.",
                rel_path,
            )
            continue
        file_path = repo_path / rel_path
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            file_content = file_path.read_text(encoding="utf-8", errors="replace")
            trimmed = _truncate_context_text(file_content, limit=min(max_chars, 8000))
            instruction_chunks.append(f"=== {rel_path} ===\n{trimmed}")
        except Exception as exc:
            logger.warning("Could not read context file %s: %s", rel_path, exc)

    agent_instructions = "\n\n".join(instruction_chunks).strip()
    context_listing = _build_context_listing(repo_path, max_depth=2, max_entries=350)

    repo_facts: dict = {}
    try:
        from app.agents.analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(repo_path)
        repo_context = analyzer.analyze_repository()
        repo_facts = repo_context.to_dict()
    except Exception as exc:
        logger.warning("Structured repository analysis failed; falling back to heuristics: %s", exc)
        repo_facts = {"error": str(exc), "fallback": True}

    if repo_facts.get("fallback"):
        repo_facts = _heuristic_analysis(repo_path)

    summary = _format_context_summary(repo_facts)

    # -- Static analysis (best-effort, never blocks the workflow) ----------
    static_issues: list[dict] = []
    try:
        from app.tools.static_analysis import run_static_analysis

        language = _extract_language(repo_facts)
        if language:
            emit_status("planner", f"Running static analysis ({language})…",
                        **_progress_meta(state, "planning"))
            raw_issues = run_static_analysis(repo_path, language)
            static_issues = [i.model_dump() for i in raw_issues]
            err_count = sum(1 for i in raw_issues if i.severity == "error")
            warn_count = sum(1 for i in raw_issues if i.severity == "warning")
            emit_status(
                "planner",
                f"Static analysis complete: {err_count} error(s), {warn_count} warning(s)",
                **_progress_meta(state, "planning"),
            )
    except Exception as exc:
        logger.warning("Static analysis failed (skipping): %s", exc)

    # -- Call graph analysis (best-effort, never blocks the workflow) -------
    call_graph: dict = {}
    try:
        from app.analysis.call_graph import CallGraphAnalyzer, format_call_graph_for_prompt

        emit_status("planner", "Building call graph…", **_progress_meta(state, "planning"))
        cg_analyzer = CallGraphAnalyzer(repo_path)
        cg = cg_analyzer.analyze()
        call_graph = cg.to_dict()
        emit_status(
            "planner",
            f"Call graph ready: {len(cg.all_functions())} functions, "
            f"{sum(len(v) for v in cg.callees.values())} edges",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Call graph analysis failed (skipping): %s", exc)

    # -- Dependency graph analysis (best-effort, never blocks the workflow) --
    dependency_graph: dict = {}
    dep_cycles: list = []
    try:
        from app.analysis.dependency_graph import DependencyAnalyzer, format_dep_graph_for_prompt

        emit_status("planner", "Building dependency graph…", **_progress_meta(state, "planning"))
        dep_analyzer = DependencyAnalyzer(repo_path)
        dg = dep_analyzer.analyze()
        dependency_graph = dg.to_dict()
        dep_cycles = dg.cycles
        cycle_msg = f", {len(dg.cycles)} cycle(s) detected" if dg.cycles else ""
        emit_status(
            "planner",
            f"Dependency graph ready: {len(dg.all_modules())} modules{cycle_msg}",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Dependency graph analysis failed (skipping): %s", exc)

    # -- Code smell detection (best-effort, never blocks the workflow) --------
    code_smells: list[dict] = []
    try:
        from app.analysis.smell_detector import SmellDetector, format_smells_for_prompt

        emit_status("planner", "Detecting code smells…", **_progress_meta(state, "planning"))
        detector = SmellDetector(repo_path, call_graph=call_graph)
        raw_smells = detector.detect()
        code_smells = [s.model_dump() for s in raw_smells]
        err_count  = sum(1 for s in raw_smells if s.severity == "error")
        warn_count = sum(1 for s in raw_smells if s.severity == "warning")
        emit_status(
            "planner",
            f"Code smell detection complete: {err_count} error(s), {warn_count} warning(s), {len(raw_smells)} total",
            **_progress_meta(state, "planning"),
        )
    except Exception as exc:
        logger.warning("Code smell detection failed (skipping): %s", exc)

    emit_status(
        "planner",
        f"Repository context loaded: {summary}",
        **_progress_meta(state, "planning"),
    )
    emit_node_end("planner", "Context Loader", "Repository context ready for planner")

    return {
        "input_intent": "code",
        "agent_instructions": agent_instructions,
        "repo_facts": repo_facts,
        "context_listing": _truncate_context_text(context_listing, limit=min(max_chars, 10000)),
        "context_loaded": True,
        "static_issues": static_issues,
        "call_graph": call_graph,
        "dependency_graph": dependency_graph,
        "dep_cycles": dep_cycles,
        "code_smells": code_smells,
    }



def code_intelligence_node(state: GraphState) -> dict:
    """Run all four analysis tools and cache results by git commit hash.

    Flow:
      1. Determine repo path.
      2. Get current git commit hash → cache key.
      3. If cache hit → restore all analysis fields instantly.
      4. If cache miss → run StaticAnalysis, CallGraph, DependencyGraph,
         SmellDetector sequentially; save to cache.
      5. Return updated state fields + emit intelligence_complete event.
    """
    from app.analysis.intelligence_cache import get_commit_hash, load_cache, save_cache

    settings = get_settings()
    repo_root = (state.repo_root or settings.target_repo_path or "").strip()
    if not repo_root:
        logger.warning("code_intelligence_node: no repo path, skipping")
        return {}

    repo_path = Path(repo_root).resolve()
    if not repo_path.is_dir():
        logger.warning("code_intelligence_node: invalid repo path %s, skipping", repo_path)
        return {}

    emit_node_start("planner", "Code Intelligence", item_desc="Analysing repository…")
    emit_status("planner", "Starting code intelligence analysis…", **_progress_meta(state, "analyzing"))

    # ── Cache lookup ────────────────────────────────────────────────────────
    cache_key = get_commit_hash(repo_path) or ""
    if cache_key:
        cached = load_cache(repo_path, cache_key)
        if cached:
            emit_status(
                "planner",
                f"Code intelligence loaded from cache (commit {cache_key})",
                **_progress_meta(state, "analyzing"),
            )
            _emit_intelligence_complete(cached)
            emit_node_end("planner", "Code Intelligence", "Cache hit — analysis restored")
            return {**cached, "intelligence_cache_key": cache_key, "intelligence_cached": True}

    # ── Run all four tools ──────────────────────────────────────────────────
    language = _extract_language(state.repo_facts)

    static_issues: list[dict] = []
    try:
        from app.tools.static_analysis import run_static_analysis
        emit_status("planner", "Running static analysis…", **_progress_meta(state, "analyzing"))
        raw = run_static_analysis(repo_path, language)
        static_issues = [i.model_dump() for i in raw]
    except Exception as exc:
        logger.warning("intelligence: static analysis failed: %s", exc)

    call_graph: dict = {}
    try:
        from app.analysis.call_graph import CallGraphAnalyzer
        emit_status("planner", "Building call graph…", **_progress_meta(state, "analyzing"))
        cg = CallGraphAnalyzer(repo_path).analyze()
        call_graph = cg.to_dict()
    except Exception as exc:
        logger.warning("intelligence: call graph failed: %s", exc)

    dependency_graph: dict = {}
    dep_cycles: list = []
    try:
        from app.analysis.dependency_graph import DependencyAnalyzer
        emit_status("planner", "Building dependency graph…", **_progress_meta(state, "analyzing"))
        dg = DependencyAnalyzer(repo_path).analyze()
        dependency_graph = dg.to_dict()
        dep_cycles = dg.cycles
    except Exception as exc:
        logger.warning("intelligence: dependency graph failed: %s", exc)

    code_smells: list[dict] = []
    try:
        from app.analysis.smell_detector import SmellDetector
        emit_status("planner", "Detecting code smells…", **_progress_meta(state, "analyzing"))
        smells = SmellDetector(repo_path, call_graph=call_graph).detect()
        code_smells = [s.model_dump() for s in smells]
    except Exception as exc:
        logger.warning("intelligence: smell detection failed: %s", exc)

    result = {
        "static_issues": static_issues,
        "call_graph": call_graph,
        "dependency_graph": dependency_graph,
        "dep_cycles": dep_cycles,
        "code_smells": code_smells,
    }

    # ── Save to cache ───────────────────────────────────────────────────────
    if cache_key:
        save_cache(repo_path, cache_key, result)

    # ── Summary emit ────────────────────────────────────────────────────────
    n_errors  = sum(1 for s in static_issues if s.get("severity") == "error")
    n_smells  = len(code_smells)
    n_cycles  = len(dep_cycles)
    summary   = f"{n_errors} static error(s) · {n_smells} smell(s) · {n_cycles} cycle(s)"

    emit_status("planner", f"Code intelligence complete: {summary}", **_progress_meta(state, "analyzing"))
    _emit_intelligence_complete(result)
    emit_node_end("planner", "Code Intelligence", summary)

    return {
        **result,
        "intelligence_cache_key": cache_key,
        "intelligence_cached": False,
        "phase": WorkflowPhase.ANALYZING,
    }


def _emit_intelligence_complete(data: dict) -> None:
    """Broadcast intelligence_complete event with summary counts via WorkflowEvent."""
    try:
        from app.core.events import emit, WorkflowEvent, EventCategory
        n_static_err  = sum(1 for s in data.get("static_issues", []) if s.get("severity") == "error")
        n_static_warn = sum(1 for s in data.get("static_issues", []) if s.get("severity") == "warning")
        n_smells      = len(data.get("code_smells", []))
        n_smell_err   = sum(1 for s in data.get("code_smells", []) if s.get("severity") == "error")
        n_cycles      = len(data.get("dep_cycles", []))
        n_funcs       = len(data.get("call_graph", {}).get("callees", {}))
        n_modules     = len(data.get("dependency_graph", {}).get("imports", {}))
        emit(WorkflowEvent(
            category=EventCategory.STATUS,
            agent="planner",
            title="intelligence_complete",
            detail=(
                f"{n_static_err} static error(s), {n_static_warn} warning(s) | "
                f"{n_smells} smell(s) ({n_smell_err} error) | "
                f"{n_cycles} cycle(s) | {n_funcs} functions | {n_modules} modules"
            ),
            metadata={
                "type": "intelligence_complete",
                "static_errors": n_static_err,
                "static_warnings": n_static_warn,
                "smells_total": n_smells,
                "smells_errors": n_smell_err,
                "cycles": n_cycles,
                "functions": n_funcs,
                "modules": n_modules,
            },
        ))
    except Exception as exc:
        logger.debug("_emit_intelligence_complete: %s", exc)


def _truncate_context_text(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-tail:]


def _build_context_listing(repo_path: Path, max_depth: int = 2, max_entries: int = 300) -> str:
    """Build a compact repo listing without using mutation-capable tools."""
    lines: list[str] = []
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}

    def _walk(path: Path, depth: int, prefix: str = "") -> None:
        if len(lines) >= max_entries or depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return
        for entry in entries:
            if len(lines) >= max_entries:
                return
            if entry.name in skipped_dirs:
                continue
            if entry.name.startswith(".") and entry.name not in {".github", ".agents"}:
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                _walk(entry, depth + 1, prefix + "  ")
            else:
                lines.append(f"{prefix}{entry.name}")

    _walk(repo_path, depth=0)
    if len(lines) >= max_entries:
        lines.append("... [listing truncated]")
    return "\n".join(lines) if lines else "(empty)"


def _heuristic_analysis(repo_path: Path) -> dict:
    """Fallback file-based repository analysis."""
    facts: dict = {}

    if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
        facts["language"] = "python"
        facts["package_manager"] = "poetry" if (repo_path / "poetry.lock").exists() else "pip"
        if (repo_path / "pytest.ini").exists():
            facts["test_framework"] = "pytest"
            facts["test_command"] = "python -m pytest -q"
        else:
            facts["test_framework"] = "unittest"
            facts["test_command"] = "python -m unittest discover"
    elif (repo_path / "package.json").exists():
        facts["language"] = "javascript"
        facts["package_manager"] = "yarn" if (repo_path / "yarn.lock").exists() else "npm"
        facts["test_command"] = "npm test"
        try:
            parsed = json.loads((repo_path / "package.json").read_text(encoding="utf-8"))
            scripts = parsed.get("scripts", {})
            if isinstance(scripts, dict):
                facts["test_command"] = scripts.get("test", "npm test")
        except Exception:
            pass
    else:
        facts["language"] = "unknown"

    if (repo_path / ".github" / "workflows").exists():
        facts["ci_cd"] = "github_actions"
    elif (repo_path / ".gitlab-ci.yml").exists():
        facts["ci_cd"] = "gitlab_ci"

    facts["has_docker"] = (repo_path / "Dockerfile").exists()
    return facts


def _format_intelligence_summary_for_prompt(state: "GraphState") -> str:
    """Return a concise, consolidated intelligence block for any agent prompt.

    Combines static analysis, call graph, dependency graph and smell data
    into a single ``## Code Intelligence Summary`` section. Designed to be
    appended to any agent's prompt without repetition.

    Budget control:
    - planner / coder: full (max smells=8, issues=8, cg=8, dg=5)
    - reviewer:        reduced (smells=5, issues=5, cg=5, dg=3)
    - tester:          minimal (smells=3, issues=5)
    """
    parts: list[str] = []

    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=8))
    if state.call_graph:
        parts.append(_format_call_graph_for_prompt(state.call_graph, max_entries=8))
    if state.dependency_graph:
        parts.append(_format_dep_graph_for_prompt(state.dependency_graph, max_entries=5))
    if state.code_smells:
        parts.append(_format_code_smells_for_prompt(state.code_smells, max_smells=8))

    if not parts:
        return ""

    cached_note = ""
    if getattr(state, "intelligence_cached", False):
        key = getattr(state, "intelligence_cache_key", "")
        cached_note = f" (cached @ {key})" if key else " (cached)"

    header = f"## Code Intelligence Summary{cached_note}\n"
    return header + "\n\n".join(parts)


def _format_intelligence_summary_reviewer(state: "GraphState") -> str:
    """Reduced intelligence block for reviewer agents."""
    parts: list[str] = []
    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=5))
    if state.call_graph:
        parts.append(_format_call_graph_for_prompt(state.call_graph, max_entries=5))
    if state.dependency_graph:
        parts.append(_format_dep_graph_for_prompt(state.dependency_graph, max_entries=3))
    if state.code_smells:
        parts.append(_format_code_smells_for_prompt(state.code_smells, max_smells=5))
    if not parts:
        return ""
    cached_note = ""
    if getattr(state, "intelligence_cached", False):
        key = getattr(state, "intelligence_cache_key", "")
        cached_note = f" (cached @ {key})" if key else " (cached)"
    return f"## Code Intelligence Summary{cached_note}\n" + "\n\n".join(parts)


def _format_intelligence_summary_tester(state: "GraphState") -> str:
    """Minimal intelligence block for tester agent — smells + static errors only."""
    parts: list[str] = []
    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=5))
    if state.code_smells:
        # Tester only needs errors, not info-level smells
        errors_only = [s for s in state.code_smells if s.get("severity") == "error"]
        if errors_only:
            parts.append(_format_code_smells_for_prompt(errors_only, max_smells=3))
    if not parts:
        return ""
    return "## Code Intelligence Summary\n" + "\n\n".join(parts)


def _format_context_summary(repo_facts: dict) -> str:
    if not repo_facts:
        return "no facts detected"
    if repo_facts.get("error"):
        return f"analysis failed ({repo_facts['error']})"

    lines: list[str] = []
    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        language = tech_stack.get("language", "unknown")
        framework = tech_stack.get("framework")
        lines.append(f"language={language}")
        if framework:
            lines.append(f"framework={framework}")
    elif repo_facts.get("language"):
        lines.append(f"language={repo_facts['language']}")

    test_framework = repo_facts.get("test_framework")
    if isinstance(test_framework, dict):
        if test_framework.get("name"):
            lines.append(f"tests={test_framework['name']}")
        if test_framework.get("unit_test_command"):
            lines.append(f"test_cmd={test_framework['unit_test_command']}")
    elif isinstance(test_framework, str):
        lines.append(f"tests={test_framework}")
    elif repo_facts.get("test_command"):
        lines.append(f"test_cmd={repo_facts['test_command']}")

    conventions = repo_facts.get("conventions")
    if isinstance(conventions, dict):
        if conventions.get("linting_tool"):
            lines.append(f"lint={conventions['linting_tool']}")
        if conventions.get("formatting_tool"):
            lines.append(f"fmt={conventions['formatting_tool']}")

    ci_cd_setup = repo_facts.get("ci_cd_setup")
    if isinstance(ci_cd_setup, dict) and ci_cd_setup.get("platform"):
        lines.append(f"ci={ci_cd_setup['platform']}")
    elif repo_facts.get("ci_cd"):
        lines.append(f"ci={repo_facts['ci_cd']}")

    return ", ".join(lines) if lines else "context loaded"


def _extract_test_command(repo_facts: dict) -> str | None:
    test_framework = repo_facts.get("test_framework")
    if isinstance(test_framework, dict):
        cmd = test_framework.get("unit_test_command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
    cmd = repo_facts.get("test_command")
    if isinstance(cmd, str) and cmd.strip():
        return cmd.strip()
    return None


def _extract_language(repo_facts: dict) -> str:
    """Return a normalised language string suitable for static analysis routing."""
    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        lang = tech_stack.get("language", "")
        if isinstance(lang, str):
            return lang.lower()
    lang = repo_facts.get("language", "")
    return lang.lower() if isinstance(lang, str) else ""


def _format_call_graph_for_prompt(
    call_graph: dict,
    max_entries: int = 15,
) -> str:
    """Format a serialised CallGraph dict into a concise prompt section."""
    if not call_graph:
        return ""
    try:
        from app.analysis.call_graph import CallGraph, format_call_graph_for_prompt
        cg = CallGraph.from_dict(call_graph)
        return format_call_graph_for_prompt(cg, max_entries=max_entries)
    except Exception as exc:
        logger.debug("Could not format call graph for prompt: %s", exc)
        return ""


def _format_code_smells_for_prompt(
    code_smells: list[dict],
    max_smells: int = 10,
) -> str:
    """Format a list of serialised CodeSmell dicts into a compact prompt section."""
    if not code_smells:
        return ""
    try:
        from app.analysis.smell_detector import CodeSmell, format_smells_for_prompt
        smells = [CodeSmell(**s) for s in code_smells]
        return format_smells_for_prompt(smells, max_smells=max_smells)
    except Exception as exc:
        logger.debug("Could not format code smells for prompt: %s", exc)
        return ""


def _format_dep_graph_for_prompt(
    dependency_graph: dict,
    max_entries: int = 8,
) -> str:
    """Format a serialised DependencyGraph dict into a concise prompt section."""
    if not dependency_graph:
        return ""
    try:
        from app.analysis.dependency_graph import DependencyGraph, format_dep_graph_for_prompt
        dg = DependencyGraph.from_dict(dependency_graph)
        return format_dep_graph_for_prompt(dg, max_cycles=max_entries)
    except Exception as exc:
        logger.debug("Could not format dependency graph for prompt: %s", exc)
        return ""


def _format_static_issues_for_prompt(
    issues: list[dict],
    max_issues: int = 20,
) -> str:
    """Format serialised StaticIssue dicts into a concise prompt section."""
    if not issues:
        return ""

    errors   = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    infos    = [i for i in issues if i.get("severity") == "info"]

    selected = (errors + warnings + infos)[:max_issues]
    total = len(issues)
    shown = len(selected)

    header = (
        f"## Static Analysis — {len(errors)} error(s)"
        f", {len(warnings)} warning(s)"
        f", {len(infos)} info(s)"
    )
    if total > shown:
        header += f" (showing top {shown} of {total})"

    lines = [header, ""]
    for issue in selected:
        sev = issue.get("severity", "warning").upper()
        rule = f" [{issue['rule_id']}]" if issue.get("rule_id") else ""
        loc = f"{issue.get('file', '?')}:{issue.get('line', 0)}"
        msg = issue.get("message", "")
        tool = issue.get("tool", "")
        lines.append(f"- [{sev}]{rule} {loc} — {msg}  (tool: {tool})")

    return "\n".join(lines)


def _format_repo_context_for_prompt(repo_facts: dict) -> str:
    """Format structured context so planner/coder can follow repo conventions."""
    if not repo_facts:
        return "=== REPOSITORY CONTEXT ===\nNo repository facts detected."

    lines = ["=== REPOSITORY CONTEXT ===", "You MUST follow these detected conventions.", ""]

    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        lines.append(f"Language: {tech_stack.get('language', 'unknown')}")
        framework = tech_stack.get("framework")
        if framework:
            version = tech_stack.get("framework_version")
            lines.append(f"Framework: {framework}{f' {version}' if version else ''}")
        manager = tech_stack.get("package_manager")
        if manager:
            lines.append(f"Package Manager: {manager}")
    elif repo_facts.get("language"):
        lines.append(f"Language: {repo_facts.get('language', 'unknown')}")

    test_framework = repo_facts.get("test_framework")
    test_command = _extract_test_command(repo_facts)
    if isinstance(test_framework, dict):
        lines.append("")
        lines.append(f"Test Framework: {test_framework.get('name', 'unknown')}")
    elif isinstance(test_framework, str):
        lines.append("")
        lines.append(f"Test Framework: {test_framework}")
    if test_command:
        lines.append(f"Test Command: {test_command}")
        lines.append("CRITICAL: Prefer this test command for verification.")

    conventions = repo_facts.get("conventions")
    if isinstance(conventions, dict):
        lines.append("")
        lines.append("Code Style:")
        if conventions.get("linting_tool"):
            lines.append(f"- Linting: {conventions['linting_tool']}")
        if conventions.get("formatting_tool"):
            lines.append(f"- Formatting: {conventions['formatting_tool']}")
        if conventions.get("max_line_length"):
            lines.append(f"- Max line length: {conventions['max_line_length']}")
        if conventions.get("function_naming"):
            lines.append(f"- Function naming: {conventions['function_naming']}")
        if conventions.get("class_naming"):
            lines.append(f"- Class naming: {conventions['class_naming']}")

    architecture = repo_facts.get("architecture")
    if isinstance(architecture, dict):
        lines.append("")
        lines.append(f"Architecture: {architecture.get('type', 'unknown')}")
        layers = architecture.get("layers")
        if isinstance(layers, list) and layers:
            lines.append(f"Layers: {', '.join(str(layer) for layer in layers)}")

    ci_cd_setup = repo_facts.get("ci_cd_setup")
    if isinstance(ci_cd_setup, dict) and ci_cd_setup.get("platform"):
        lines.append("")
        lines.append(f"CI/CD: {ci_cd_setup['platform']}")
        lines.append("Be careful with CI/CD changes.")
    elif repo_facts.get("ci_cd"):
        lines.append("")
        lines.append(f"CI/CD: {repo_facts.get('ci_cd')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NODE: status / research / resume (minimal non-coding branches)
# ---------------------------------------------------------------------------

def status_node(state: GraphState) -> dict:
    """Non-coding status/conversational response branch (no write/git tools).

    Handles questions like "who are you?", "what can you do?", "what's the
    status?", etc. Uses an LLM to generate a contextual answer and emits it
    visibly to the UI via emit_agent_result so the user sees the reply.
    """
    emit_node_start("planner", "Status", item_desc=state.user_request[:100])

    # Include current workflow state as context
    summary = state.get_progress_summary()

    # Prepend context to the user message so the LLM has full picture
    context_prefix = (
        "You are Daedalus, an autonomous AI coding agent. "
        "You help developers by autonomously cloning, analysing, coding, testing, "
        "and documenting software repositories. "
        "You use a dual-coder system (Claude + GPT) with peer review, human approval "
        "gates, checkpoint/resume, and a code intelligence pipeline.\n\n"
        "Answer the user's question conversationally. "
        "If it's about workflow status, use the context below. "
        "Keep your answer concise (3-6 sentences). Do not ask follow-up questions.\n\n"
        f"Current workflow state:\n{summary}\n\n"
        f"User question: {state.user_request}"
    )

    answer = _invoke_agent(
        "planner",
        [HumanMessage(content=context_prefix)],
        tools=None,
        inject_memory=False,
    )

    # Emit the answer as a visible response — shown as an expanded reply bubble,
    # not a collapsible. The agent_result keeps it in the log too.
    emit_agent_response("planner", answer)

    emit_node_end("planner", "Status", "Response sent")
    return {
        "planner_response": answer,
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
    # Emit the answer as a visible response — shown as an expanded reply bubble
    emit_agent_response("planner", answer)
    emit_node_end("planner", "Research", "Research response prepared")
    return {
        "planner_response": answer,
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "research_answered",
        "input_intent": "research",
    }


def resume_node(state: GraphState) -> dict:
    """Resume branch using checkpoints first, then TODO fallback."""
    emit_node_start("planner", "Resume", item_desc=state.user_request[:100])

    restored = checkpoint_manager.load_checkpoint(repo_root=state.repo_root)
    if restored is not None:
        payload = restored.model_dump()
        payload["input_intent"] = "resume"
        payload["resumed_from_checkpoint"] = True

        pending = payload.get("pending_approval", {})
        approved = isinstance(pending, dict) and bool(pending.get("approved"))
        if payload.get("needs_human_approval") and not approved:
            payload["phase"] = WorkflowPhase.WAITING_FOR_APPROVAL
            payload["stop_reason"] = "human_approval_required"
            emit_node_end("planner", "Resume", "Checkpoint restored; awaiting human approval")
        elif approved:
            payload["needs_human_approval"] = False
            payload["phase"] = WorkflowPhase.COMMITTING
            payload["stop_reason"] = ""
            emit_node_end("planner", "Resume", "Checkpoint restored; approval detected, resuming commit")
        else:
            emit_node_end("planner", "Resume", "Checkpoint restored")
        return payload

    resumed = _resume_from_saved_todo(state)
    if resumed is not None:
        resumed["input_intent"] = "resume"
        resumed["resumed_from_checkpoint"] = False
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

    updates = {
        "input_intent": intent,
        "todo_items": items if items else state.todo_items,
        "current_item_index": 0 if items else state.current_item_index,
        "branch_name": branch,
        "phase": WorkflowPhase.CODING,
        "needs_replan": False,
        "active_coder": active_coder,
        "active_reviewer": active_reviewer,
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
    if state.repo_facts:
        prompt_parts.append(_format_repo_context_for_prompt(state.repo_facts))
    intelligence_summary = _format_intelligence_summary_for_prompt(state)
    if intelligence_summary:
        prompt_parts.append(intelligence_summary)
    if state.agent_instructions:
        prompt_parts.append(
            "**Repository Documentation (excerpt)**:\n"
            + _truncate_context_text(state.agent_instructions, limit=3000)
        )

    preferred_test_cmd = _extract_test_command(state.repo_facts)

    prompt_parts.append(
        "\nImplement this task. Use tools to read the codebase, make changes, "
        "add tests where needed, and update docs. Keep diffs minimal.\n"
        "Follow the coding style and architecture decisions from shared memory.\n\n"
        "CRITICAL WORKFLOW:\n"
        "1. Search for similar code first via `search_in_repo`.\n"
        "2. Read existing implementations with `read_file` before editing.\n"
        "3. Prefer minimal edits and keep architecture patterns consistent.\n"
        "4. Add or update tests with every behavior change.\n"
        f"5. Verify using preferred command: `{preferred_test_cmd or 'python -m pytest -q'}`."
    )

    # If the human answered a previous question from this coder, inject the
    # answer as the first message so the coder continues with that context.
    messages: list = []
    if state.coder_question_answer and state.coder_question_asked_by == active:
        messages.append(HumanMessage(
            content=(
                "[Human answered your earlier question]\n"
                f"Q: {state.coder_question}\n"
                f"A: {state.coder_question_answer}\n\n"
                "Continue implementing the task using this answer as your guide."
            )
        ))

    messages.append(HumanMessage(content="\n\n".join(prompt_parts)))

    tools = DOCUMENTER_TOOLS if active == "documenter" else CODER_TOOLS
    try:
        result, budget_update = _invoke_with_budget(state, active, messages, tools, inject_memory=True, node="coder")
    except BudgetExceededException:
        return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}

    # Detect ask_human signal — coder wants to pause for human input
    question_payload = _parse_coder_question(result)
    if question_payload:
        emit_coder_question(
            asked_by=active,
            question=question_payload["question"],
            context=question_payload["context"],
            options=question_payload["options"],
            item_id=item.id,
        )
        emit_node_end(active, "Coding", "Paused — coder is asking the human a question")
        return {
            "needs_coder_answer": True,
            "coder_question": question_payload["question"],
            "coder_question_context": question_payload["context"],
            "coder_question_options": question_payload["options"],
            "coder_question_asked_by": active,
            "coder_question_answer": "",   # clear any previous answer
            "phase": WorkflowPhase.WAITING_FOR_ANSWER,
            "stop_reason": "waiting_for_coder_answer",
        }

    emit_node_end(active, "Coding", f"Implementation complete — handing to {_reviewer_label(reviewer)} for peer review")
    with suppress(Exception):
        _write_todo_file(state.todo_items, state.user_request)

    updates = {
        "last_coder_result": result,
        "phase": WorkflowPhase.PEER_REVIEWING,
        "total_iterations": state.total_iterations + 1,
        # Clear any lingering question state from previous items
        "needs_coder_answer": False,
        "coder_question": "",
        "coder_question_context": "",
        "coder_question_options": [],
        "coder_question_asked_by": "",
        "coder_question_answer": "",
        **budget_update,
    }
    _save_checkpoint_snapshot(state, updates, "code_complete")
    return updates


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

    intelligence_ctx = _format_intelligence_summary_reviewer(state)

    prompt = (
        f"## Peer Code Review\n\n"
        f"**Reviewer**: {rev_label}\n"
        f"**Implementer**: {impl_label}\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implementer's Report**:\n{state.last_coder_result}\n\n"
        + (f"{intelligence_ctx}\n\n" if intelligence_ctx else "")
        + f"Review the changes. Also verify consistency with the shared memory "
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
    try:
        result, budget_update = _invoke_with_budget(
            state, reviewer, [HumanMessage(content=prompt)],
            REVIEWER_TOOLS, inject_memory=True, node="peer_review",
        )
    except BudgetExceededException:
        return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}

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
        **budget_update,
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

# Patterns in test/runner output that indicate an environment problem,
# not a test failure. The tester detects these and hands off to planner_env_fix.
_ENV_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"'pytest'\s+is not recognized", re.IGNORECASE),
    re.compile(r'The term ["\']pytest["\'] is not recognized', re.IGNORECASE),
    re.compile(r"No module named\s+\S+", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"cannot import name", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),          # Node.js
    re.compile(r"error: externally-managed-environment", re.IGNORECASE),
    re.compile(r"Could not find.*executable", re.IGNORECASE),
    re.compile(r"No such file or directory.*python", re.IGNORECASE),
    re.compile(r"python.*not found", re.IGNORECASE),
    re.compile(r"node.*not found", re.IGNORECASE),
]

_TEST_PASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d+ passed", re.IGNORECASE),
    re.compile(r"All tests passed", re.IGNORECASE),
    re.compile(r"Tests passed", re.IGNORECASE),
    re.compile(r"OK\s*$", re.MULTILINE),
]

# Maximum env-fix rounds before giving up and routing to human gate
_MAX_ENV_FIX_ATTEMPTS = 2


def _is_env_failure(output: str) -> bool:
    """Return True if the output indicates a missing/broken test environment."""
    return any(pat.search(output) for pat in _ENV_FAILURE_PATTERNS)


def _is_test_pass(output: str) -> bool:
    """Return True if the output clearly indicates all tests passed (no failures)."""
    has_failure = re.search(r"\d+\s+failed", output, re.IGNORECASE)
    if has_failure:
        return False
    return any(pat.search(output) for pat in _TEST_PASS_PATTERNS)


def _classify_test_output(output: str) -> str:
    """Classify raw test runner output as 'pass', 'env_failure', or 'test_failure'."""
    if _is_env_failure(output):
        return "env_failure"
    if _is_test_pass(output):
        return "pass"
    return "test_failure"


def tester_node(state: GraphState) -> dict:
    """Run tests and verification.

    Classifies the test runner output into three categories:
    - pass        → advance to decide/human-gate
    - test_failure → route back to coder for a fix
    - env_failure  → route to planner_env_fix (never handled here)

    The tester's job is to write, run, and analyse tests.
    It never modifies the environment — that is the planner's responsibility.
    """
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
    intelligence_ctx = _format_intelligence_summary_tester(state)
    if intelligence_ctx:
        prompt += f"\n\n{intelligence_ctx}"
    prompt += (
        "\nRun all tests, linters, and verification commands. "
        "Produce a structured test report with PASS or FAIL verdict. "
        "If the test runner itself cannot start (missing interpreter, missing module, "
        "command not found), report that clearly — do NOT attempt to install packages."
    )

    try:
        result, budget_update = _invoke_with_budget(
            state, "tester", [HumanMessage(content=prompt)], TESTER_TOOLS, node="tester",
        )
    except BudgetExceededException:
        return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}

    # -- Classify the raw LLM output ----------------------------------------
    if "**Verdict**: PASS" in result or "Verdict: PASS" in result:
        llm_verdict = "PASS"
    elif "**Verdict**: FAIL" in result or "Verdict: FAIL" in result:
        llm_verdict = "FAIL"
    elif "PASS" in result.upper() and "FAIL" not in result.upper():
        llm_verdict = "PASS"
    else:
        llm_verdict = "FAIL"

    # Override: if output signals env failure, trust that over LLM verdict
    classification = _classify_test_output(result)

    item.test_report = result

    # -- ENV FAILURE: hand off to planner_env_fix ---------------------------
    if classification == "env_failure":
        if state.env_fix_attempts >= _MAX_ENV_FIX_ATTEMPTS:
            msg = (
                f"Environment setup failed after {state.env_fix_attempts} fix attempt(s). "
                "Human intervention required."
            )
            emit_status("tester", f"❌ {msg}", **_progress_meta(state, "stopped"))
            emit_node_end("tester", "Testing", msg)
            return {
                "last_test_result": result,
                "stop_reason": "env_setup_failed",
                "phase": WorkflowPhase.STOPPED,
                **budget_update,
            }

        emit_status(
            "tester",
            "⚠️ Tester: missing dependency or broken environment detected "
            "— handing to planner for auto-fix",
            **_progress_meta(state, "env_fixing"),
        )
        emit_node_end("tester", "Testing", "ENV_FAILURE — routing to planner_env_fix")
        return {
            "last_test_result": result,
            "phase": WorkflowPhase.ENV_FIXING,
            **budget_update,
        }

    # -- Normal PASS / FAIL -------------------------------------------------
    emit_verdict("tester", llm_verdict, detail=result, item_id=item.id)

    if llm_verdict == "FAIL":
        item.test_fail_count += 1
        item.status = ItemStatus.IN_PROGRESS
        record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=False)
        if item.test_fail_count >= get_settings().max_rework_cycles_per_item:
            msg = f"Item {item.id} failed tests {item.test_fail_count} times; stopping to avoid loop."
            emit_error("tester", msg)
            return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED, **budget_update}
        emit_status(
            "tester",
            f"❌ Tests FAILED - sending back to {_coder_label(state.active_coder)}",
            **_progress_meta(state, "coding"),
        )
    else:
        emit_status("tester", "✅ All tests PASSED", **_progress_meta(state, "deciding"))

    emit_node_end("tester", "Testing", f"Verdict: {llm_verdict}")

    updates = {
        "last_test_result": result,
        "env_fix_attempts": 0,  # reset on successful test run (pass or genuine fail)
        "phase": WorkflowPhase.DECIDING if llm_verdict == "PASS" else WorkflowPhase.CODING,
        **budget_update,
    }
    if llm_verdict == "PASS":
        _save_checkpoint_snapshot(state, updates, "test_pass")
    return updates


# ---------------------------------------------------------------------------
# NODE: planner_env_fix
# ---------------------------------------------------------------------------

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

    raw = _invoke_agent("planner", [HumanMessage(content=prompt)], tools=None, inject_memory=False)

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


def _count_lines_in_numstat(diff_numstat: str) -> int:
    total = 0
    for line in diff_numstat.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("["):
            continue
        parts = clean.split("\t")
        if len(parts) < 3:
            continue
        try:
            added = 0 if parts[0] == "-" else int(parts[0])
            deleted = 0 if parts[1] == "-" else int(parts[1])
        except ValueError:
            continue
        total += added + deleted
    return total


def _parse_changed_files_from_status(porcelain_status: str) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    deleted: list[str] = []
    for line in porcelain_status.splitlines():
        clean = line.rstrip()
        if len(clean) < 4 or clean.startswith("["):
            continue
        status = clean[:2]
        file_path = clean[3:].strip()
        if not file_path:
            continue
        changed.append(file_path)
        if "D" in status:
            deleted.append(file_path)
    return changed, deleted


def answer_gate_node(state: GraphState) -> dict:
    """Pause the workflow until the human answers the coder's question.

    The node is visited after coder_node emits a ``coder_question`` event and
    sets ``needs_coder_answer=True``.  It has two exit paths:

    * Answer received (``coder_question_answer`` is non-empty):
      Clear the question fields and advance to ``CODING`` so the coder
      resumes with the answer injected into its next invocation.

    * Still waiting:
      Return ``WAITING_FOR_ANSWER`` so the orchestrator halts the graph.
      The web server or Telegram bot will call ``/api/answer`` to populate
      ``coder_question_answer``, then queue a resume task.
    """
    emit_node_start("system", "Answer Gate", item_desc=state.user_request[:100])

    # If the coder question was already answered (e.g. on resume after answer),
    # clear the fields and let the coder continue.
    if state.coder_question_answer:
        emit_status(
            "system",
            f"✅ Answer received — resuming {state.coder_question_asked_by}",
            **_progress_meta(state, "coding"),
        )
        emit_node_end("system", "Answer Gate", "Answer delivered, coder will continue")
        return {
            "needs_coder_answer": False,
            "phase": WorkflowPhase.CODING,
            "stop_reason": "",
        }

    # Still waiting — halt the workflow
    emit_status(
        "system",
        f"⏳ Waiting for human answer to: {state.coder_question[:120]}",
        **_progress_meta(state, "waiting_for_answer"),
    )
    emit_node_end("system", "Answer Gate", "Halted — waiting for human answer")
    return {
        "phase": WorkflowPhase.WAITING_FOR_ANSWER,
        "stop_reason": "waiting_for_coder_answer",
    }


def human_gate_node(state: GraphState) -> dict:
    """Pause for human approval before commit and risky operations."""
    emit_node_start("planner", "Human Gate", item_desc=state.user_request[:100])
    emit_status("planner", "Evaluating whether human approval is required", **_progress_meta(state, "reviewing"))

    pending = state.pending_approval or {}
    if pending.get("approved"):
        emit_node_end("planner", "Human Gate", "Approval already granted")
        return {"needs_human_approval": False, "phase": WorkflowPhase.COMMITTING, "stop_reason": ""}

    if state.needs_human_approval and not pending.get("approved"):
        emit_node_end("planner", "Human Gate", "Still waiting for human approval")
        return {"phase": WorkflowPhase.WAITING_FOR_APPROVAL, "stop_reason": "human_approval_required"}

    approval_triggers = [{"type": "commit", "reason": "Commit requires human approval"}]

    status_raw = git_command.invoke({"command": "git status --porcelain"})
    changed_files, deleted_files = _parse_changed_files_from_status(status_raw)
    for file_path in deleted_files:
        approval_triggers.append({"type": "file_deletion", "reason": f"File deletion: {file_path}"})

    ci_markers = (".github/workflows/", ".gitlab-ci.yml", "Jenkinsfile", ".circleci/")
    for file_path in changed_files:
        if any(marker in file_path for marker in ci_markers):
            approval_triggers.append({"type": "ci_config_change", "reason": f"CI/CD change: {file_path}"})

    diff_numstat = git_command.invoke({"command": "git diff --numstat"})
    lines_changed = _count_lines_in_numstat(diff_numstat)
    if lines_changed > 400:
        approval_triggers.append({"type": "large_diff", "reason": f"Large diff: {lines_changed} changed lines"})

    diff_preview = git_command.invoke({"command": "git diff"})
    if len(diff_preview) > 5000:
        diff_preview = diff_preview[:5000] + "\n\n... (truncated)"

    payload = {
        "type": "commit",
        "triggers": approval_triggers,
        "summary": f"{len(changed_files)} files changed, {lines_changed} changed lines",
        "files": changed_files,
        "git_status": status_raw,
        "diff_preview": diff_preview,
        "approved": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    history = list(state.approval_history)
    history.append({"timestamp": payload["timestamp"], "approved": None, "triggers": approval_triggers})

    updates = {
        "needs_human_approval": True,
        "pending_approval": payload,
        "approval_history": history,
        "phase": WorkflowPhase.WAITING_FOR_APPROVAL,
        "stop_reason": "human_approval_required",
    }
    _save_checkpoint_snapshot(state, updates, "await_approval")

    emit_status(
        "planner",
        "Human approval required before commit",
        **_progress_meta(state, "waiting_for_approval"),
    )
    emit_approval_needed(payload)
    emit_node_end("planner", "Human Gate", "Paused for human approval")
    return updates


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
    _save_checkpoint_snapshot(state, {"phase": WorkflowPhase.COMMITTING}, "commit_success")

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
            "needs_human_approval": False,
            "pending_approval": {},
            "stop_reason": "",
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
        return {
            "phase": WorkflowPhase.COMPLETE,
            "needs_human_approval": False,
            "pending_approval": {},
            "stop_reason": "",
        }

# ---------------------------------------------------------------------------
# Documenter Node
# ---------------------------------------------------------------------------

# Patterns in a git diff that indicate documentation should be updated.
# Matched against the raw diff text (additions + context lines).
_DOCS_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\+\s*def [a-z]", re.MULTILINE),         # new public function
    re.compile(r"^\+\s*async def [a-z]", re.MULTILINE),   # new public async function
    re.compile(r"^\+\s*class [A-Z]", re.MULTILINE),        # new public class
    re.compile(r"^\+.*@(app|router)\.(get|post|put|delete|patch|head)\b", re.MULTILINE),  # new API endpoint
    re.compile(r"^\+.*settings\.\w+\s*=", re.MULTILINE),  # new settings assignment
    re.compile(r"^\+\s*[A-Z_]{3,}\s*=\s*", re.MULTILINE), # new constant / env var
    re.compile(r"^\+.*argparse\|add_argument", re.MULTILINE),  # new CLI arg
]


def _diff_needs_docs(diff: str) -> bool:
    """Return True if the git diff contains documentation-worthy changes.

    Uses lightweight regex heuristics so no LLM call is made for trivial
    commits (test-only changes, typo fixes, pure refactors).
    """
    return any(pat.search(diff) for pat in _DOCS_TRIGGER_PATTERNS)


def documenter_node(state: GraphState) -> dict:
    """Update project documentation after a successful commit.

    Runs after every commit. Uses a heuristic diff scan to decide whether
    an LLM call is warranted. If the diff contains no documentation-worthy
    changes the node exits immediately without an LLM call.

    The documenter writes documentation changes to disk but does NOT commit
    them — they will be picked up by the next commit cycle or can be
    committed manually.
    """
    emit_node_start("documenter", "Documenting", item_desc="Checking diff for documentation updates")
    emit_status(
        "documenter",
        "📝 Documenter: scanning commit diff…",
        **_progress_meta(state, "documenting"),
    )

    # -- 1. Get diff of the last commit -----------------------------------
    try:
        diff = git_command.invoke({"command": "git diff HEAD~1 HEAD"})
    except Exception as exc:
        logger.warning("documenter | git diff failed: %s", exc)
        diff = ""

    if not diff:
        emit_node_end("documenter", "Documenting", "No diff available — skipping documentation update")
        return {}

    # -- 2. Heuristic gate — skip LLM if diff is not doc-worthy ----------
    if not _diff_needs_docs(diff):
        emit_status(
            "documenter",
            "📝 Documenter: no documentation-worthy changes detected — skipping",
            **_progress_meta(state, "documenting"),
        )
        emit_node_end("documenter", "Documenting", "Skipped — diff contains no public API or config changes")
        return {}

    # -- 3. LLM call with full diff context ------------------------------
    emit_status(
        "documenter",
        "📝 Documenter: documentation-worthy changes detected — updating docs…",
        **_progress_meta(state, "documenting"),
    )

    prompt = (
        "## Documentation Task\n\n"
        "A commit was just made. Review the diff below and update the project documentation "
        "following your system instructions.\n\n"
        f"### Git Diff (last commit)\n\n```diff\n{diff[:8000]}\n```\n\n"
        "Start by reading any existing CHANGELOG.md and README.md with the available tools, "
        "then make the necessary updates. Output your structured summary when done."
    )

    try:
        result, budget_update = _invoke_with_budget(
            state, "documenter", [HumanMessage(content=prompt)],
            DOCUMENTER_TOOLS, node="documenter",
        )
    except BudgetExceededException:
        emit_node_end("documenter", "Documenting", "Budget limit exceeded — skipping documentation")
        return {}

    emit_node_end("documenter", "Documenting", result[:400] if result else "Documentation updated")
    return {**budget_update}
