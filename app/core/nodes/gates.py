"""Gate nodes — human approval, plan approval, and answer gates."""
from __future__ import annotations

import re
from datetime import UTC, datetime

from app.core.events import (
    emit,
    emit_plan,
    emit_approval_needed,
    emit_node_end,
    emit_node_start,
    emit_plan_approval_needed,
    emit_status,
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase
from app.tools.git import git_command, git_status

from ._helpers import (
    _coder_label,
    _progress_meta,
    _save_checkpoint_snapshot,
)

logger = get_logger("core.nodes.gates")

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


def _format_plan_for_human(items: list) -> str:
    """Format TODO items as a compact numbered list for human review."""
    if not items:
        return "(empty plan)"
    lines = []
    for idx, item in enumerate(items, start=1):
        task_type = getattr(item, "task_type", "coding") or "coding"
        desc = getattr(item, "description", "") or ""
        agent = getattr(item, "assigned_agent", "") or ""
        agent_label = _coder_label(agent) if agent else ""
        suffix = f"  → {agent_label}" if agent_label else ""
        status = getattr(item, "status", None)
        if status is not None and str(status) == "done":
            mark = "✅"
        else:
            mark = "⬜"
        lines.append(f"  {mark} {idx}. [{task_type}] {desc}{suffix}")
    return "\n".join(lines)


def plan_approval_gate_node(state: GraphState) -> dict:
    """Pause after planning — wait for human GO before the coder starts.

    Mirrors answer_gate_node. Three exit paths:

    * plan_approved=True, feedback empty   → CODING (proceed as-is)
    * plan_approved=True, feedback present → PLANNING (planner revises once,
                                             plan_revision_count incremented)
    * still waiting                        → WAITING_FOR_PLAN_APPROVAL (graph halts)

    The web server or Telegram bot calls POST /api/plan-approve to set
    plan_approved=True (and optionally plan_approval_feedback), then queues
    a resume task to restart the graph.

    Env-fix ops plans bypass this gate (planner_env_fix_node never sets
    needs_plan_approval=True) so internal fix cycles are unaffected.
    """
    emit_node_start("planner", "Plan Approval Gate", item_desc=state.user_request[:100])

    if state.plan_approved:
        if state.plan_approval_feedback:
            # Human wants a revision — send back to planner for one more round
            feedback_preview = state.plan_approval_feedback[:80]
            emit_status(
                "planner",
                f"🔄 Plan revision requested: {feedback_preview}",
                **_progress_meta(state, "planning"),
            )
            emit_node_end("planner", "Plan Approval Gate", "Revision requested — replanning")
            return {
                "needs_plan_approval": False,
                "plan_approved": False,
                "plan_revision_count": state.plan_revision_count + 1,
                "phase": WorkflowPhase.PLANNING,
                "stop_reason": "",
            }

        # Pure GO — proceed to coding
        emit_status(
            "planner",
            "✅ Plan approved — starting coder",
            **_progress_meta(state, "coding"),
        )
        emit_node_end("planner", "Plan Approval Gate", "Approved — coding starts")
        return {
            "needs_plan_approval": False,
            "plan_approved": False,
            "plan_approval_feedback": "",
            "phase": WorkflowPhase.CODING,
            "stop_reason": "",
        }

    # Still waiting — emit plan for display and halt
    plan_summary = _format_plan_for_human(state.todo_items)
    emit_plan_approval_needed("planner", plan_summary, state.todo_items)
    pending_plan_items = [
        {
            "id": getattr(i, "id", ""),
            "description": getattr(i, "description", ""),
            "task_type": getattr(i, "task_type", "coding"),
            "assigned_agent": getattr(i, "assigned_agent", "") or "",
        }
        for i in state.todo_items
    ]
    emit_status(
        "planner",
        f"⏳ Waiting for plan approval ({len(state.todo_items)} items) — "
        "approve, revise, or cancel in the UI",
        **{
            **_progress_meta(state, "waiting_for_plan_approval"),
            "needs_plan_approval": True,
            "pending_plan_items": pending_plan_items,
            "plan_summary": plan_summary,
        },
    )
    emit_node_end("planner", "Plan Approval Gate", "Halted — waiting for human plan approval")
    return {
        "needs_plan_approval": True,
        "phase": WorkflowPhase.WAITING_FOR_PLAN_APPROVAL,
        "stop_reason": "waiting_for_plan_approval",
    }


def answer_gate_node(state: GraphState) -> dict:
    """Pause the workflow until the human answers the coder's question.

    The node is visited after coder_node emits a ``coder_question`` event and
    sets ``needs_coder_answer=True``.  It has two exit paths:

    * Answer received (``coder_question_answer`` is non-empty):
      Clear the question fields and advance to ``CODING`` so the coder
      resumes with the answer injected into its next invocation.

      Special sentinel ``"__skip__"``: the human chose the "Skip — use default"
      button for an advisory question.  The gate replaces the sentinel with
      ``coder_question_default`` (or a generic fallback) so the coder receives
      a concrete assumption rather than the raw sentinel value.

    * Still waiting:
      Return ``WAITING_FOR_ANSWER`` so the orchestrator halts the graph.
      The web server or Telegram bot will call ``/api/answer`` to populate
      ``coder_question_answer``, then queue a resume task.
    """
    emit_node_start("system", "Answer Gate", item_desc=state.user_request[:100])

    # If the coder question was already answered (e.g. on resume after answer),
    # clear the fields and let the coder continue.
    if state.coder_question_answer:
        answer = state.coder_question_answer

        # Handle __skip__ sentinel: replace with the coder's stated default
        if answer == "__skip__":
            default_ = state.coder_question_default or "Proceeding with the most reasonable default."
            answer   = default_
            emit_status(
                "system",
                f"⚡ Human chose to skip — using default: {default_[:120]}",
                **_progress_meta(state, "coding"),
            )
        else:
            emit_status(
                "system",
                f"✅ Answer received — resuming {state.coder_question_asked_by}",
                **_progress_meta(state, "coding"),
            )

        emit_node_end("system", "Answer Gate", "Answer delivered, coder will continue")
        return {
            "needs_coder_answer": False,
            "coder_question_answer": answer,
            "coder_question_urgency": "blocking",  # reset to safe default
            "coder_question_default": "",
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
        "will_create_pr": bool(state.repo_ref and get_settings().auto_create_pr),
        "branch": state.branch_name,
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


