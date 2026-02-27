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
    EventCategory,
    WorkflowEvent,
    emit,
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
    emit_plan_approval_needed,
    emit_pr_created,
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
    parse_issue_ref,
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


PLANNER_TOOLS = [read_file, write_file, list_directory, search_in_repo, git_status, run_terminal]

CODER_TOOLS = [
    read_file, write_file, list_directory, search_in_repo,
    git_status, run_terminal,
]
REVIEWER_TOOLS = [
    read_file, list_directory, search_in_repo,
    git_status, run_terminal,
]
TESTER_TOOLS = [read_file, list_directory, run_terminal, run_tests, run_linter, git_status]
DOCUMENTER_TOOLS = [read_file, write_file, list_directory, run_terminal, git_status, git_command]

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
    invoke_agent = globals().get("_invoke_agent")
    if invoke_agent is None:
        from ._streaming import _invoke_agent as invoke_agent
    result = invoke_agent(
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

__all__ = [name for name in globals() if not name.startswith("__")]
