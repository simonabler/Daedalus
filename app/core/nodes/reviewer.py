"""Reviewer nodes — peer review and learning extraction."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import load_system_prompt
from app.core.events import (
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_status,
    emit_verdict,
)
from app.core.logging import get_logger
from app.core.memory import LEARNING_EXTRACTION_PROMPT, append_memory
from app.core.state import GraphState, ItemStatus, WorkflowPhase
from app.core.config import get_settings
from app.core.task_routing import record_agent_outcome
from app.core.token_budget import BudgetExceededException

from app.tools.git import git_command

from ._helpers import (
    REVIEWER_TOOLS,
    _coder_label,
    _invoke_with_budget,
    _progress_meta,
    _reviewer_label,
)
from ._context_format import _format_intelligence_summary_reviewer

logger = get_logger("core.nodes.reviewer")

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
        result, budget_update = _invoke_with_budget(
            state, "planner", [HumanMessage(content=prompt)],
            tools=None, inject_memory=False, node="planner_review",
        )

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

