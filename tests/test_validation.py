"""Tests for the validation node system.

Covers:
  STRAT-COMP-001  Unsupervised chain with silent error propagation
  STRAT-COMP-004  Silent errors across data boundaries
  STRAT-SI-001a   Unvalidated error boundary planner → coder
  STRAT-SI-002    Silent error swallowing in catch-all handler
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.nodes.validation import (
    CONTRACT_CODER_TO_PEER_REVIEW,
    CONTRACT_PEER_REVIEW_TO_LEARN,
    CONTRACT_PLANNER_TO_CODER,
    CONTRACT_TESTER_TO_DECIDE,
    ValidationFailure,
    _check_coder_result_not_empty,
    _check_current_item_has_description,
    _check_current_item_in_bounds,
    _check_error_message_clear,
    _check_peer_review_notes_not_empty,
    _check_phase_not_stopped,
    _check_test_result_not_empty,
    _check_todo_items_have_valid_task_types,
    _check_todo_items_not_empty,
    error_handler_node,
    validate_node,
)
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(id="item-001", desc="Do something", task_type="coding") -> TodoItem:
    return TodoItem(id=id, description=desc, task_type=task_type)


def _state(**kwargs) -> GraphState:
    defaults = dict(user_request="test task")
    defaults.update(kwargs)
    return GraphState(**defaults)


def _state_with_item(item: TodoItem, **kwargs) -> GraphState:
    return _state(todo_items=[item], current_item_index=0, **kwargs)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

class TestCheckTodoItemsNotEmpty:
    def test_passes_with_items(self):
        state = _state(todo_items=[_item()])
        assert _check_todo_items_not_empty(state) is None

    def test_fails_with_no_items(self):
        state = _state(todo_items=[])
        result = _check_todo_items_not_empty(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "todo_items_empty"


class TestCheckCurrentItemInBounds:
    def test_passes_with_valid_index(self):
        state = _state(todo_items=[_item()], current_item_index=0)
        assert _check_current_item_in_bounds(state) is None

    def test_fails_with_negative_index(self):
        state = _state(todo_items=[_item()], current_item_index=-1)
        result = _check_current_item_in_bounds(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "current_item_index_out_of_bounds"

    def test_fails_with_index_beyond_list(self):
        state = _state(todo_items=[_item()], current_item_index=5)
        result = _check_current_item_in_bounds(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "current_item_index_out_of_bounds"


class TestCheckCurrentItemHasDescription:
    def test_passes_with_description(self):
        state = _state_with_item(_item(desc="Add endpoint"))
        assert _check_current_item_has_description(state) is None

    def test_fails_with_empty_description(self):
        state = _state_with_item(_item(desc=""))
        result = _check_current_item_has_description(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "current_item_empty_description"

    def test_fails_with_whitespace_only_description(self):
        state = _state_with_item(_item(desc="   "))
        result = _check_current_item_has_description(state)
        assert isinstance(result, ValidationFailure)

    def test_passes_with_no_current_item(self):
        # Out-of-bounds index: _check_current_item_in_bounds catches it, this one skips
        state = _state(todo_items=[], current_item_index=-1)
        assert _check_current_item_has_description(state) is None


class TestCheckTaskTypes:
    def test_passes_with_valid_types(self):
        items = [
            _item(id="a", task_type="coding"),
            _item(id="b", task_type="testing"),
            _item(id="c", task_type="documentation"),
            _item(id="d", task_type="ops"),
        ]
        state = _state(todo_items=items)
        assert _check_todo_items_have_valid_task_types(state) is None

    def test_fails_with_invalid_type(self):
        state = _state(todo_items=[_item(task_type="magic")])
        result = _check_todo_items_have_valid_task_types(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "invalid_task_type"


class TestCheckCoderResult:
    def test_passes_with_result(self):
        state = _state(last_coder_result="I implemented the endpoint.")
        assert _check_coder_result_not_empty(state) is None

    def test_fails_with_empty_result(self):
        state = _state(last_coder_result="")
        result = _check_coder_result_not_empty(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "coder_result_empty"

    def test_fails_with_whitespace_result(self):
        state = _state(last_coder_result="   ")
        result = _check_coder_result_not_empty(state)
        assert isinstance(result, ValidationFailure)


class TestCheckPeerReviewNotes:
    def test_passes_when_reviewing_phase_has_notes(self):
        state = _state(phase=WorkflowPhase.REVIEWING, peer_review_notes="Looks good.")
        assert _check_peer_review_notes_not_empty(state) is None

    def test_fails_when_reviewing_phase_has_no_notes(self):
        state = _state(phase=WorkflowPhase.REVIEWING, peer_review_notes="")
        result = _check_peer_review_notes_not_empty(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "peer_review_notes_empty"

    def test_passes_in_non_review_phase(self):
        # In PLANNING phase, empty notes are fine
        state = _state(phase=WorkflowPhase.PLANNING, peer_review_notes="")
        assert _check_peer_review_notes_not_empty(state) is None


class TestCheckTestResult:
    def test_passes_with_result(self):
        state = _state(last_test_result="5 passed in 0.4s")
        assert _check_test_result_not_empty(state) is None

    def test_fails_with_empty_result(self):
        state = _state(last_test_result="")
        result = _check_test_result_not_empty(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "test_result_empty"


class TestCheckPhaseStopped:
    def test_passes_when_not_stopped(self):
        state = _state(phase=WorkflowPhase.CODING)
        assert _check_phase_not_stopped(state) is None

    def test_fails_when_stopped(self):
        state = _state(phase=WorkflowPhase.STOPPED)
        result = _check_phase_not_stopped(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "phase_already_stopped"


class TestCheckErrorMessageClear:
    def test_passes_when_no_error(self):
        state = _state(error_message="")
        assert _check_error_message_clear(state) is None

    def test_fails_when_error_set(self):
        state = _state(error_message="Something went wrong upstream")
        result = _check_error_message_clear(state)
        assert isinstance(result, ValidationFailure)
        assert result.rule == "unhandled_error_message"

    def test_fails_with_any_non_empty_error(self):
        state = _state(error_message="No current item to work on")
        result = _check_error_message_clear(state)
        assert result is not None


# ---------------------------------------------------------------------------
# validate_node integration
# ---------------------------------------------------------------------------

def _run_validate(contract: str, state: GraphState) -> dict:
    with patch("app.core.nodes.validation.emit_node_start"), \
         patch("app.core.nodes.validation.emit_node_end"), \
         patch("app.core.nodes.validation.emit_status"), \
         patch("app.core.nodes.validation.emit_error"):
        patched = state.model_copy(update={"validation_contract": contract})
        return validate_node(patched)


class TestValidateNode:
    def test_passes_with_no_contract(self):
        state = _state()
        result = _run_validate("", state)
        assert result["validation_passed"] is True

    def test_passes_with_unknown_contract(self):
        state = _state()
        result = _run_validate("nonexistent_contract", state)
        assert result["validation_passed"] is True

    def test_planner_to_coder_passes_valid_state(self):
        item = _item()
        state = _state_with_item(item, phase=WorkflowPhase.CODING)
        result = _run_validate("planner_to_coder", state)
        assert result["validation_passed"] is True
        assert result["validation_contract"] == ""

    def test_planner_to_coder_fails_empty_items(self):
        state = _state(todo_items=[], current_item_index=-1, phase=WorkflowPhase.CODING)
        result = _run_validate("planner_to_coder", state)
        assert result["validation_passed"] is False
        assert result["phase"] == WorkflowPhase.STOPPED
        assert "todo_items_empty" in result["stop_reason"] or \
               any(f["rule"] == "todo_items_empty" for f in result["validation_failures"])

    def test_planner_to_coder_fails_unhandled_error(self):
        item = _item()
        state = _state_with_item(item, error_message="Planner crashed", phase=WorkflowPhase.CODING)
        result = _run_validate("planner_to_coder", state)
        assert result["validation_passed"] is False
        assert result["phase"] == WorkflowPhase.STOPPED
        assert any(f["rule"] == "unhandled_error_message" for f in result["validation_failures"])

    def test_planner_to_coder_fails_stopped_phase(self):
        item = _item()
        state = _state_with_item(item, phase=WorkflowPhase.STOPPED)
        result = _run_validate("planner_to_coder", state)
        assert result["validation_passed"] is False

    def test_coder_to_peer_review_passes(self):
        item = _item()
        state = _state_with_item(item, last_coder_result="Implemented.", phase=WorkflowPhase.PEER_REVIEWING)
        result = _run_validate("coder_to_peer_review", state)
        assert result["validation_passed"] is True

    def test_coder_to_peer_review_fails_empty_result(self):
        item = _item()
        state = _state_with_item(item, last_coder_result="", phase=WorkflowPhase.PEER_REVIEWING)
        result = _run_validate("coder_to_peer_review", state)
        assert result["validation_passed"] is False
        assert any(f["rule"] == "coder_result_empty" for f in result["validation_failures"])

    def test_peer_review_to_learn_passes(self):
        state = _state(phase=WorkflowPhase.REVIEWING, peer_review_notes="Good work.")
        result = _run_validate("peer_review_to_learn", state)
        assert result["validation_passed"] is True

    def test_peer_review_to_learn_fails_empty_notes(self):
        state = _state(phase=WorkflowPhase.REVIEWING, peer_review_notes="")
        result = _run_validate("peer_review_to_learn", state)
        assert result["validation_passed"] is False

    def test_tester_to_decide_passes(self):
        state = _state(last_test_result="3 passed", phase=WorkflowPhase.DECIDING)
        result = _run_validate("tester_to_decide", state)
        assert result["validation_passed"] is True

    def test_tester_to_decide_fails_empty_result(self):
        state = _state(last_test_result="", phase=WorkflowPhase.DECIDING)
        result = _run_validate("tester_to_decide", state)
        assert result["validation_passed"] is False
        assert any(f["rule"] == "test_result_empty" for f in result["validation_failures"])

    def test_multiple_failures_all_reported(self):
        """All failures are reported, not just the first."""
        # Empty items AND stopped phase AND error message → 3 failures
        state = _state(
            todo_items=[],
            current_item_index=-1,
            phase=WorkflowPhase.STOPPED,
            error_message="Some prior error",
        )
        result = _run_validate("planner_to_coder", state)
        assert result["validation_passed"] is False
        assert len(result["validation_failures"]) >= 2

    def test_validation_failures_field_structure(self):
        """Each failure entry has 'rule' and 'message' keys."""
        state = _state(todo_items=[], current_item_index=-1, phase=WorkflowPhase.CODING)
        result = _run_validate("planner_to_coder", state)
        for failure in result["validation_failures"]:
            assert "rule" in failure
            assert "message" in failure
            assert isinstance(failure["rule"], str)
            assert isinstance(failure["message"], str)

    def test_stop_reason_includes_contract_name(self):
        state = _state(todo_items=[], current_item_index=-1, phase=WorkflowPhase.CODING)
        result = _run_validate("planner_to_coder", state)
        assert "planner_to_coder" in result["stop_reason"]


# ---------------------------------------------------------------------------
# error_handler_node
# ---------------------------------------------------------------------------

class TestErrorHandlerNode:
    def _run(self, **kwargs) -> dict:
        state = _state(**kwargs)
        with patch("app.core.nodes.validation.emit_node_start"), \
             patch("app.core.nodes.validation.emit_node_end"), \
             patch("app.core.nodes.validation.emit_status"):
            return error_handler_node(state)

    def test_returns_stopped_phase(self):
        result = self._run(error_message="Something broke")
        assert result["phase"] == WorkflowPhase.STOPPED

    def test_sets_stop_reason(self):
        result = self._run(error_message="Planner crashed", stop_reason="planner_llm_failure")
        assert result["stop_reason"] == "planner_llm_failure"

    def test_uses_error_message_in_stop_reason_when_no_stop_reason(self):
        result = self._run(error_message="No items in plan")
        assert "error_handler" in result["stop_reason"]

    def test_handles_empty_error_gracefully(self):
        result = self._run()
        assert result["phase"] == WorkflowPhase.STOPPED
        assert result["stop_reason"]


# ---------------------------------------------------------------------------
# State field tests
# ---------------------------------------------------------------------------

class TestValidationStateFields:
    def test_validation_contract_default(self):
        state = GraphState(user_request="test")
        assert state.validation_contract == ""

    def test_validation_passed_default(self):
        state = GraphState(user_request="test")
        assert state.validation_passed is False

    def test_validation_failures_default(self):
        state = GraphState(user_request="test")
        assert state.validation_failures == []

    def test_fields_survive_checkpoint_round_trip(self):
        state = GraphState(
            user_request="test",
            validation_contract="planner_to_coder",
            validation_passed=True,
            validation_failures=[{"rule": "todo_items_empty", "message": "no items"}],
        )
        restored = GraphState(**state.model_dump())
        assert restored.validation_contract == "planner_to_coder"
        assert restored.validation_passed is True
        assert restored.validation_failures[0]["rule"] == "todo_items_empty"


# ---------------------------------------------------------------------------
# Orchestrator routing tests for validation nodes
# ---------------------------------------------------------------------------

class TestOrchestratorValidationRouting:
    def test_route_after_validate_passes(self):
        from app.core.orchestrator import _route_after_validate
        state = _state(validation_passed=True, phase=WorkflowPhase.CODING)
        route = _route_after_validate("coder")
        assert route(state) == "coder"

    def test_route_after_validate_fails(self):
        from app.core.orchestrator import _route_after_validate
        state = _state(validation_passed=False, phase=WorkflowPhase.STOPPED)
        route = _route_after_validate("coder")
        assert route(state) == "error_handler"

    def test_route_after_validate_stopped_goes_to_error(self):
        from app.core.orchestrator import _route_after_validate
        state = _state(validation_passed=True, phase=WorkflowPhase.STOPPED)
        route = _route_after_validate("coder")
        assert route(state) == "error_handler"

    def test_make_validate_node_sets_contract(self):
        from app.core.orchestrator import _make_validate_node
        item = _item()
        state = _state_with_item(item, phase=WorkflowPhase.CODING)
        node_fn = _make_validate_node("planner_to_coder")
        with patch("app.core.nodes.validation.emit_node_start"), \
             patch("app.core.nodes.validation.emit_node_end"), \
             patch("app.core.nodes.validation.emit_status"), \
             patch("app.core.nodes.validation.emit_error"):
            result = node_fn(state)
        assert result["validation_passed"] is True

    def test_make_validate_node_has_descriptive_name(self):
        from app.core.orchestrator import _make_validate_node
        node_fn = _make_validate_node("tester_to_decide")
        assert "tester_to_decide" in node_fn.__name__
