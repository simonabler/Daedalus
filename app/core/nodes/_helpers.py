"""Shared helpers used across multiple node modules."""
from __future__ import annotations

import json
import platform
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.checkpoints import checkpoint_manager
from app.core.config import get_settings
from app.core.events import (
    emit_agent_result,
    emit_agent_thinking,
    emit_context_usage,
    emit_error,
    emit_status,
    emit_token_usage,
    emit_tool_call,
    emit_tool_result,
)
from app.core.logging import get_logger
from app.core.memory import load_all_memory
from app.core.state import GraphState, ItemStatus, TodoItem
from app.core.token_budget import (
    BudgetExceededException,
    TokenBudget,
    TokenUsageRecord,
    calculate_cost,
    extract_token_usage,
)
from app.core.context_window import (
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

from ._streaming import _STREAMING_ROLES, _stream_llm_round

logger = get_logger("core.nodes._helpers")

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
        "context":  str(payload.get("context", "")).strip(),
        "options":  [str(o) for o in payload.get("options", []) if o],
        "urgency":  payload.get("urgency", "blocking") if payload.get("urgency") in ("blocking", "advisory") else "blocking",
        "default_if_skipped": str(payload.get("default_if_skipped", "")).strip(),
    }



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


def _os_note(execution_platform: str) -> str:
    """Return a platform-appropriate OS note for agent prompts.

    Uses the detected execution platform string (from ``platform.platform()``)
    to produce a concise, accurate shell-usage hint so agents do not guess
    wrong commands.
    """
    p = (execution_platform or "").lower()
    if "windows" in p:
        return "Runtime is **Windows** — use PowerShell syntax for all terminal commands."
    if "darwin" in p or "macos" in p:
        return "Runtime is **macOS** — use standard bash/zsh syntax for all terminal commands."
    # Default: Linux / unknown
    return "Runtime is **Linux** — use standard bash syntax for all terminal commands."


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

