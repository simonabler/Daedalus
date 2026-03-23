"""Validation node — guards agent handoff points against silent error propagation.

Addresses findings:
  STRAT-COMP-001  Unsupervised chain with silent error propagation
  STRAT-COMP-004  Silent errors across data boundaries
  STRAT-SI-001a   Unvalidated error boundary planner → coder
  STRAT-SI-002    Silent error swallowing in catch-all handler

Design
------
``validate_node`` is a lightweight, stateless guard inserted between the major
handoff points in the graph:

    planner     → [validate] → coder
    coder       → [validate] → peer_review
    peer_review → [validate] → learn
    tester      → [validate] → decide

Each transition has a specific *contract*: a set of invariants that MUST hold
for the downstream node to produce sensible output.  When a contract is
violated, the node sets ``error_message`` and ``validation_error`` on state
and returns ``WorkflowPhase.STOPPED`` so the conditional edge can route to the
error handler instead of propagating the bad state silently.

``error_handler_node`` provides a consistent, human-readable error report and
terminates the workflow cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.core.events import emit_error, emit_node_end, emit_node_start, emit_status
from app.core.logging import get_logger
from app.core.state import GraphState, TodoItem, WorkflowPhase

logger = get_logger("core.nodes.validation")

# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------


@dataclass
class ValidationFailure:
    """A single broken invariant."""
    rule: str        # short machine-readable tag, e.g. "todo_items_empty"
    message: str     # human-readable explanation


@dataclass
class Contract:
    """A named set of invariants for a particular handoff point."""
    name: str
    checks: list[Callable[[GraphState], ValidationFailure | None]] = field(default_factory=list)

    def validate(self, state: GraphState) -> list[ValidationFailure]:
        failures = []
        for check in self.checks:
            failure = check(state)
            if failure is not None:
                failures.append(failure)
        return failures


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_todo_items_not_empty(state: GraphState) -> ValidationFailure | None:
    if not state.todo_items:
        return ValidationFailure(
            rule="todo_items_empty",
            message="Planner produced an empty plan — todo_items is empty. "
                    "Cannot proceed to coder without at least one item.",
        )
    return None


def _check_current_item_in_bounds(state: GraphState) -> ValidationFailure | None:
    if state.current_item_index < 0 or state.current_item_index >= len(state.todo_items):
        return ValidationFailure(
            rule="current_item_index_out_of_bounds",
            message=f"current_item_index={state.current_item_index} is out of range "
                    f"for todo_items (len={len(state.todo_items)}).",
        )
    return None


def _check_current_item_has_description(state: GraphState) -> ValidationFailure | None:
    item: TodoItem | None = state.current_item
    if item is None:
        return None  # already caught by _check_current_item_in_bounds
    if not item.description or not item.description.strip():
        return ValidationFailure(
            rule="current_item_empty_description",
            message=f"Item {item.id!r} has an empty description. "
                    "Coder cannot implement a task with no description.",
        )
    return None


def _check_todo_items_have_valid_task_types(state: GraphState) -> ValidationFailure | None:
    valid = {"coding", "documentation", "testing", "ops"}
    bad = [
        item.id for item in state.todo_items
        if item.task_type not in valid
    ]
    if bad:
        return ValidationFailure(
            rule="invalid_task_type",
            message=f"Items {bad} have unrecognised task_type values. "
                    f"Expected one of {sorted(valid)}.",
        )
    return None


def _check_coder_result_not_empty(state: GraphState) -> ValidationFailure | None:
    if not state.last_coder_result or not state.last_coder_result.strip():
        return ValidationFailure(
            rule="coder_result_empty",
            message="Coder produced an empty result. "
                    "Peer reviewer cannot review nothing.",
        )
    return None


def _check_peer_review_notes_not_empty(state: GraphState) -> ValidationFailure | None:
    # Notes are only required when phase indicates review just happened
    if state.phase in (WorkflowPhase.REVIEWING, WorkflowPhase.CODING, WorkflowPhase.TESTING):
        if not state.peer_review_notes or not state.peer_review_notes.strip():
            return ValidationFailure(
                rule="peer_review_notes_empty",
                message="Peer review completed but produced no notes. "
                        "Learner/planner-review cannot extract insights from an empty review.",
            )
    return None


def _check_test_result_not_empty(state: GraphState) -> ValidationFailure | None:
    if not state.last_test_result or not state.last_test_result.strip():
        return ValidationFailure(
            rule="test_result_empty",
            message="Tester produced an empty result. "
                    "Decide node cannot make a verdict without test output.",
        )
    return None


def _check_phase_not_stopped(state: GraphState) -> ValidationFailure | None:
    if state.phase == WorkflowPhase.STOPPED:
        return ValidationFailure(
            rule="phase_already_stopped",
            message=f"State phase is STOPPED before handoff. "
                    f"stop_reason={state.stop_reason!r}",
        )
    return None


def _check_error_message_clear(state: GraphState) -> ValidationFailure | None:
    """Detect errors that were set on state but not acted on (silent propagation)."""
    if state.error_message and state.error_message.strip():
        return ValidationFailure(
            rule="unhandled_error_message",
            message=f"An error was recorded on state but not handled: "
                    f"{state.error_message!r}. Blocking propagation.",
        )
    return None


# ---------------------------------------------------------------------------
# Named contracts per handoff point
# ---------------------------------------------------------------------------

#: planner → coder
CONTRACT_PLANNER_TO_CODER = Contract(
    name="planner_to_coder",
    checks=[
        _check_phase_not_stopped,
        _check_error_message_clear,
        _check_todo_items_not_empty,
        _check_todo_items_have_valid_task_types,
        _check_current_item_in_bounds,
        _check_current_item_has_description,
    ],
)

#: coder → peer_review
CONTRACT_CODER_TO_PEER_REVIEW = Contract(
    name="coder_to_peer_review",
    checks=[
        _check_phase_not_stopped,
        _check_error_message_clear,
        _check_current_item_in_bounds,
        _check_coder_result_not_empty,
    ],
)

#: peer_review → learn
CONTRACT_PEER_REVIEW_TO_LEARN = Contract(
    name="peer_review_to_learn",
    checks=[
        _check_phase_not_stopped,
        _check_error_message_clear,
        _check_peer_review_notes_not_empty,
    ],
)

#: tester → decide
CONTRACT_TESTER_TO_DECIDE = Contract(
    name="tester_to_decide",
    checks=[
        _check_phase_not_stopped,
        _check_error_message_clear,
        _check_test_result_not_empty,
    ],
)

# Registry: maps contract name → Contract object (used by validate_node)
_CONTRACTS: dict[str, Contract] = {
    c.name: c for c in [
        CONTRACT_PLANNER_TO_CODER,
        CONTRACT_CODER_TO_PEER_REVIEW,
        CONTRACT_PEER_REVIEW_TO_LEARN,
        CONTRACT_TESTER_TO_DECIDE,
    ]
}


# ---------------------------------------------------------------------------
# validate_node
# ---------------------------------------------------------------------------

def validate_node(state: GraphState) -> dict:
    """Validate GraphState at a handoff point before passing to the next node.

    The contract to apply is determined by ``state.validation_contract``.
    On success, returns an empty dict (no state changes) and sets
    ``validation_passed=True``.
    On failure, sets ``error_message``, ``validation_error``, and
    ``phase=STOPPED`` so the conditional edge routes to error_handler_node.
    """
    contract_name = state.validation_contract
    if not contract_name:
        # No contract requested — pass through (should not happen in normal flow)
        logger.debug("validate_node called with no contract set — passing through")
        return {"validation_passed": True, "validation_contract": ""}

    contract = _CONTRACTS.get(contract_name)
    if contract is None:
        logger.warning("validate_node: unknown contract %r — passing through", contract_name)
        return {"validation_passed": True, "validation_contract": ""}

    emit_node_start("system", f"Validate [{contract.name}]")

    failures = contract.validate(state)

    if not failures:
        emit_node_end("system", f"Validate [{contract.name}]", "✅ All checks passed")
        logger.debug("validate_node [%s]: all checks passed", contract.name)
        return {"validation_passed": True, "validation_contract": ""}

    # Build a human-readable error summary
    summary_lines = [
        f"❌ Validation failed at handoff [{contract.name}] "
        f"({len(failures)} violation(s)):"
    ]
    for f in failures:
        summary_lines.append(f"  • [{f.rule}] {f.message}")
        logger.error(
            "validate_node [%s] FAILED rule=%r: %s",
            contract.name, f.rule, f.message,
        )

    summary = "\n".join(summary_lines)
    emit_error("system", summary)
    emit_node_end("system", f"Validate [{contract.name}]", "FAILED")

    return {
        "validation_passed": False,
        "validation_contract": "",
        "error_message": summary,
        "validation_failures": [{"rule": f.rule, "message": f.message} for f in failures],
        "phase": WorkflowPhase.STOPPED,
        "stop_reason": f"validation_failed:{contract.name}",
    }


# ---------------------------------------------------------------------------
# error_handler_node
# ---------------------------------------------------------------------------

def error_handler_node(state: GraphState) -> dict:
    """Terminal error handler — surfaces error details and stops the workflow cleanly.

    Receives control when validate_node (or any node) sets phase=STOPPED with
    an error_message.  Emits a structured error event for the UI and logs a
    full diagnostic summary.
    """
    emit_node_start("system", "Error Handler")

    error = state.error_message or state.stop_reason or "Unknown error"
    failures = state.validation_failures

    lines = ["🛑 Workflow stopped due to an unrecoverable error.", "", f"Error: {error}"]

    if failures:
        lines.append("")
        lines.append("Validation failures:")
        for f in failures:
            lines.append(f"  • [{f['rule']}] {f['message']}")

    lines += [
        "",
        f"Phase at failure: {state.phase}",
        f"Current item: {state.current_item_index} / {len(state.todo_items)}",
        f"Completed items: {state.completed_items}",
        "",
        "To resume: fix the underlying issue, then submit a new task or resume from checkpoint.",
    ]

    report = "\n".join(lines)
    emit_status("system", report, phase="stopped")

    logger.error(
        "error_handler_node: workflow terminated | stop_reason=%r | error=%r",
        state.stop_reason,
        state.error_message,
    )

    emit_node_end("system", "Error Handler", "Workflow terminated")

    return {
        "phase": WorkflowPhase.STOPPED,
        "stop_reason": state.stop_reason or f"error_handler:{state.error_message[:80]}",
    }
