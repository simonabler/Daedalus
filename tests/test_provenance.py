"""Tests for agent output provenance and coder decision audit logging.

Covers:
  STRAT-DC-003  Unobserved decision point in delegation chain
  STRAT-SI-007  Output aggregation loses source attribution
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from app.core.state import (
    CoderDecision,
    CoderOutput,
    GraphState,
    ItemStatus,
    TodoItem,
    WorkflowPhase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(id="item-001", desc="Add endpoint", rework_count=0, iteration_count=0) -> TodoItem:
    return TodoItem(
        id=id,
        description=desc,
        task_type="coding",
        rework_count=rework_count,
        iteration_count=iteration_count,
    )


def _state(**kwargs) -> GraphState:
    defaults = dict(user_request="test", active_coder="coder_a", active_reviewer="reviewer_b")
    defaults.update(kwargs)
    return GraphState(**defaults)


def _state_with_item(item: TodoItem, **kwargs) -> GraphState:
    return _state(todo_items=[item], current_item_index=0, **kwargs)


def _run_coder_node(state: GraphState, llm_result: str = "I implemented the feature.") -> dict:
    """Run coder_node with all external calls mocked."""
    from app.core.nodes.coder import coder_node

    with patch("app.core.nodes.coder._invoke_with_budget", return_value=(llm_result, {})), \
         patch("app.core.nodes.coder._format_intelligence_summary_for_prompt", return_value=""), \
         patch("app.core.nodes.coder._format_repo_context_for_prompt", return_value=""), \
         patch("app.core.nodes.coder.emit_node_start"), \
         patch("app.core.nodes.coder.emit_node_end"), \
         patch("app.core.nodes.coder.emit_status"), \
         patch("app.core.nodes.coder._write_todo_file"), \
         patch("app.core.nodes.coder.load_system_prompt", return_value="system"):
        return coder_node(state)


def _run_peer_review(state: GraphState, verdict_text: str = "**Verdict**: APPROVE\nLooks good.") -> dict:
    """Run peer_review_node with all external calls mocked."""
    from app.core.nodes.reviewer import peer_review_node

    with patch("app.core.nodes.reviewer._invoke_with_budget", return_value=(verdict_text, {})), \
         patch("app.core.nodes.reviewer._format_intelligence_summary_reviewer", return_value=""), \
         patch("app.core.nodes.reviewer.emit_node_start"), \
         patch("app.core.nodes.reviewer.emit_node_end"), \
         patch("app.core.nodes.reviewer.emit_status"), \
         patch("app.core.nodes.reviewer.emit_verdict"), \
         patch("app.core.nodes.reviewer.record_agent_outcome"):
        return peer_review_node(state)


# ---------------------------------------------------------------------------
# CoderOutput model
# ---------------------------------------------------------------------------

class TestCoderOutputModel:
    def test_fields_exist(self):
        out = CoderOutput(
            agent="coder_a",
            item_id="item-001",
            iteration=1,
            rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00",
            result_summary="Implemented endpoint.",
        )
        assert out.agent == "coder_a"
        assert out.item_id == "item-001"
        assert out.iteration == 1
        assert out.rework_cycle == 0
        assert out.verdict == ""  # default

    def test_verdict_can_be_set(self):
        out = CoderOutput(
            agent="coder_b", item_id="item-002", iteration=2, rework_cycle=1,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Fixed tests.",
            verdict="APPROVE",
        )
        assert out.verdict == "APPROVE"

    def test_model_copy_update(self):
        out = CoderOutput(
            agent="coder_a", item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Done.",
        )
        updated = out.model_copy(update={"verdict": "REWORK"})
        assert updated.verdict == "REWORK"
        assert out.verdict == ""  # original unchanged

    def test_serializes_for_checkpoint(self):
        out = CoderOutput(
            agent="coder_a", item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Done.",
            verdict="APPROVE",
        )
        d = out.model_dump()
        restored = CoderOutput(**d)
        assert restored.agent == out.agent
        assert restored.verdict == out.verdict


# ---------------------------------------------------------------------------
# CoderDecision model
# ---------------------------------------------------------------------------

class TestCoderDecisionModel:
    def test_fields_exist(self):
        dec = CoderDecision(
            trigger="item_start",
            from_coder="",
            to_coder="coder_a",
            item_id="item-001",
            iteration=1,
            rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00",
            reason="Starting item",
        )
        assert dec.trigger == "item_start"
        assert dec.from_coder == ""
        assert dec.to_coder == "coder_a"

    def test_valid_triggers(self):
        for trigger in ("item_start", "rework", "item_advance", "rework_via_tester", "escalate"):
            dec = CoderDecision(
                trigger=trigger, from_coder="coder_a", to_coder="coder_b",
                item_id="item-001", iteration=1, rework_cycle=1,
                timestamp="2026-01-01T00:00:00+00:00",
            )
            assert dec.trigger == trigger

    def test_serializes_for_checkpoint(self):
        dec = CoderDecision(
            trigger="rework", from_coder="coder_a", to_coder="coder_a",
            item_id="item-001", iteration=2, rework_cycle=1,
            timestamp="2026-01-01T00:00:00+00:00", reason="Reviewer requested changes",
        )
        d = dec.model_dump()
        restored = CoderDecision(**d)
        assert restored.trigger == dec.trigger
        assert restored.reason == dec.reason


# ---------------------------------------------------------------------------
# GraphState log fields
# ---------------------------------------------------------------------------

class TestGraphStateLogFields:
    def test_agent_output_log_default_empty(self):
        state = GraphState(user_request="test")
        assert state.agent_output_log == []

    def test_coder_decision_log_default_empty(self):
        state = GraphState(user_request="test")
        assert state.coder_decision_log == []

    def test_logs_survive_round_trip(self):
        out = CoderOutput(
            agent="coder_a", item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Done.",
        )
        dec = CoderDecision(
            trigger="item_start", from_coder="", to_coder="coder_a",
            item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        state = GraphState(user_request="test", agent_output_log=[out], coder_decision_log=[dec])
        restored = GraphState(**state.model_dump())
        assert len(restored.agent_output_log) == 1
        assert restored.agent_output_log[0].agent == "coder_a"
        assert len(restored.coder_decision_log) == 1
        assert restored.coder_decision_log[0].trigger == "item_start"


# ---------------------------------------------------------------------------
# coder_node provenance emission
# ---------------------------------------------------------------------------

class TestCoderNodeProvenance:
    def test_emits_coder_output_on_success(self):
        item = _item()
        state = _state_with_item(item, phase=WorkflowPhase.CODING)
        result = _run_coder_node(state, llm_result="Here is my implementation.")
        assert "agent_output_log" in result
        assert len(result["agent_output_log"]) == 1
        output = result["agent_output_log"][0]
        assert output.agent == "coder_a"
        assert output.item_id == "item-001"
        assert output.iteration == 1
        assert output.rework_cycle == 0
        assert output.result_summary.startswith("Here is my implementation")
        assert output.verdict == ""  # not yet reviewed

    def test_result_summary_truncated_to_300_chars(self):
        item = _item()
        state = _state_with_item(item)
        long_result = "X" * 500
        result = _run_coder_node(state, llm_result=long_result)
        output = result["agent_output_log"][0]
        assert len(output.result_summary) == 300

    def test_emits_coder_decision_on_first_pass(self):
        item = _item(rework_count=0)
        state = _state_with_item(item)
        result = _run_coder_node(state)
        assert "coder_decision_log" in result
        assert len(result["coder_decision_log"]) == 1
        decision = result["coder_decision_log"][0]
        assert decision.trigger == "item_start"
        assert decision.to_coder == "coder_a"
        assert decision.item_id == "item-001"
        assert decision.rework_cycle == 0

    def test_emits_rework_trigger_on_subsequent_pass(self):
        item = _item(rework_count=1)
        state = _state_with_item(item)
        result = _run_coder_node(state)
        decision = result["coder_decision_log"][0]
        assert decision.trigger == "rework"
        assert decision.rework_cycle == 1

    def test_appends_to_existing_logs(self):
        existing_output = CoderOutput(
            agent="coder_b", item_id="item-000", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Previous item.",
        )
        existing_decision = CoderDecision(
            trigger="item_advance", from_coder="coder_b", to_coder="coder_a",
            item_id="item-001", iteration=0, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        item = _item()
        state = _state_with_item(
            item,
            agent_output_log=[existing_output],
            coder_decision_log=[existing_decision],
        )
        result = _run_coder_node(state)
        # Both old and new entries present
        assert len(result["agent_output_log"]) == 2
        assert result["agent_output_log"][0].agent == "coder_b"
        assert result["agent_output_log"][1].agent == "coder_a"
        assert len(result["coder_decision_log"]) == 2

    def test_output_timestamp_is_iso8601(self):
        item = _item()
        state = _state_with_item(item)
        result = _run_coder_node(state)
        ts = result["agent_output_log"][0].timestamp
        # Should parse without error
        datetime.fromisoformat(ts)

    def test_decision_timestamp_is_iso8601(self):
        item = _item()
        state = _state_with_item(item)
        result = _run_coder_node(state)
        ts = result["coder_decision_log"][0].timestamp
        datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# peer_review_node verdict stamping (STRAT-SI-007)
# ---------------------------------------------------------------------------

class TestPeerReviewVerdictStamping:
    def _state_after_coder(self, verdict: str = "APPROVE") -> GraphState:
        """Build a state that has one CoderOutput in the log."""
        existing_output = CoderOutput(
            agent="coder_a", item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Done.",
            verdict="",
        )
        item = _item()
        return _state_with_item(
            item,
            agent_output_log=[existing_output],
            last_coder_result="Implementation here.",
            phase=WorkflowPhase.PEER_REVIEWING,
            peer_review_notes="",
        )

    def test_stamps_approve_verdict_on_last_output(self):
        state = self._state_after_coder()
        result = _run_peer_review(state, "**Verdict**: APPROVE\nLooks good.")
        assert "agent_output_log" in result
        assert result["agent_output_log"][-1].verdict == "APPROVE"

    def test_stamps_rework_verdict_on_last_output(self):
        state = self._state_after_coder()
        result = _run_peer_review(state, "**Verdict**: REWORK\nAdd tests.")
        assert result["agent_output_log"][-1].verdict == "REWORK"

    def test_does_not_modify_previous_outputs(self):
        """Only the last output gets the verdict — earlier entries are unchanged."""
        earlier = CoderOutput(
            agent="coder_b", item_id="item-000", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00", result_summary="Previous.",
            verdict="APPROVE",
        )
        latest = CoderOutput(
            agent="coder_a", item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-02T00:00:00+00:00", result_summary="Current.",
            verdict="",
        )
        item = _item()
        state = _state_with_item(
            item,
            agent_output_log=[earlier, latest],
            last_coder_result="Implementation.",
            phase=WorkflowPhase.PEER_REVIEWING,
        )
        result = _run_peer_review(state, "**Verdict**: REWORK\nMore work needed.")
        log = result["agent_output_log"]
        assert log[0].verdict == "APPROVE"  # unchanged
        assert log[1].verdict == "REWORK"   # updated

    def test_handles_empty_output_log_gracefully(self):
        """If no CoderOutput exists yet, peer_review should not crash."""
        item = _item()
        state = _state_with_item(
            item,
            agent_output_log=[],
            last_coder_result="Implementation.",
            phase=WorkflowPhase.PEER_REVIEWING,
        )
        result = _run_peer_review(state, "**Verdict**: APPROVE\nLooks good.")
        # Should complete without error; log stays empty
        assert result["agent_output_log"] == []


# ---------------------------------------------------------------------------
# committer_node item_advance decision logging (STRAT-DC-003)
# ---------------------------------------------------------------------------

class TestCommitterDecisionLog:
    def _run_committer(self, state: GraphState) -> dict:
        from app.core.nodes.committer import committer_node
        with patch("app.core.nodes.committer.git_commit_and_push") as mock_git, \
             patch("app.core.nodes.committer.emit_status"), \
             patch("app.core.nodes.committer.emit_commit"), \
             patch("app.core.nodes.committer.emit_node_start"), \
             patch("app.core.nodes.committer.emit_node_end"), \
             patch("app.core.nodes.committer._save_checkpoint_snapshot"), \
             patch("app.core.nodes.committer.get_memory_stats", return_value={}), \
             patch("app.core.nodes.committer._try_post_pr_link_on_issue"), \
             patch("app.core.nodes.committer._create_pr_for_branch", return_value=None):
            mock_git.invoke.return_value = "commit abc123"
            return committer_node(state)

    def test_logs_item_advance_decision(self):
        item_done = _item(id="item-001", desc="First item")
        item_done.status = ItemStatus.DONE
        item_done.commit_message = "feat: first item"
        item_next = _item(id="item-002", desc="Second item")
        state = _state(
            todo_items=[item_done, item_next],
            current_item_index=0,
            active_coder="coder_a",
            active_reviewer="reviewer_b",
            phase=WorkflowPhase.COMMITTING,
            needs_human_approval=False,
        )
        result = self._run_committer(state)
        assert "coder_decision_log" in result
        assert len(result["coder_decision_log"]) >= 1
        decision = result["coder_decision_log"][-1]
        assert decision.trigger == "item_advance"
        assert decision.item_id == "item-002"

    def test_decision_from_coder_matches_current(self):
        item_done = _item(id="item-001")
        item_done.status = ItemStatus.DONE
        item_done.commit_message = "feat: done"
        item_next = _item(id="item-002")
        state = _state(
            todo_items=[item_done, item_next],
            current_item_index=0,
            active_coder="coder_a",
        )
        result = self._run_committer(state)
        decision = result["coder_decision_log"][-1]
        assert decision.from_coder == "coder_a"

    def test_appends_to_existing_decision_log(self):
        existing = CoderDecision(
            trigger="item_start", from_coder="", to_coder="coder_a",
            item_id="item-001", iteration=1, rework_cycle=0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        item_done = _item(id="item-001")
        item_done.status = ItemStatus.DONE
        item_done.commit_message = "feat: done"
        item_next = _item(id="item-002")
        state = _state(
            todo_items=[item_done, item_next],
            current_item_index=0,
            active_coder="coder_a",
            coder_decision_log=[existing],
        )
        result = self._run_committer(state)
        # Original entry preserved + new item_advance entry appended
        assert len(result["coder_decision_log"]) == 2
        assert result["coder_decision_log"][0].trigger == "item_start"
        assert result["coder_decision_log"][1].trigger == "item_advance"
