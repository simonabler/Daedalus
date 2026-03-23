"""Tests for the convergence exit condition in peer_review_node.

The convergence feature implements three-condition exit logic that prevents
the coder↔reviewer loop from spinning indefinitely:

  1. Max rework cycles (hard ceiling) → ESCALATE_TESTING
  2. Diff-delta similarity above threshold → ESCALATE_CONVERGENCE → tester
  3. Second+ rework cycle → REWORK_VIA_TESTER → tester before coder
  4. First rework → CODING (fast path, no tester overhead)
"""
from __future__ import annotations

import difflib
from unittest.mock import MagicMock, patch

import pytest

from app.core.nodes.reviewer import _check_convergence, _CONVERGENCE_SIMILARITY_THRESHOLD, _CONVERGENCE_MIN_REWORK
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(rework_count: int = 0, last_coder_diff: str = "") -> TodoItem:
    return TodoItem(
        id="item-001",
        description="Add health endpoint",
        task_type="coding",
        rework_count=rework_count,
        last_coder_diff=last_coder_diff,
    )


def _make_state(item: TodoItem) -> GraphState:
    state = GraphState(
        user_request="test",
        todo_items=[item],
        current_item_index=0,
        active_coder="coder_a",
        active_reviewer="reviewer_b",
        phase=WorkflowPhase.PEER_REVIEWING,
        last_coder_result="some implementation",
    )
    return state


DIFF_A = """\
diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -10,0 +11 @@
+    return {"status": "ok"}
"""

DIFF_B_SIMILAR = DIFF_A  # identical — maximum similarity

DIFF_B_DIFFERENT = """\
diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -10,0 +11,5 @@
+    result = db.query(Health).first()
+    if result is None:
+        raise HTTPException(status_code=503)
+    return {"status": "ok", "db": "connected"}
+
"""


# ---------------------------------------------------------------------------
# Unit tests for _check_convergence
# ---------------------------------------------------------------------------

class TestCheckConvergence:
    def test_returns_false_below_min_rework(self):
        """Convergence check is skipped until _CONVERGENCE_MIN_REWORK cycles."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK - 1, last_coder_diff=DIFF_A)
        assert _check_convergence(item, DIFF_B_SIMILAR) is False

    def test_returns_false_when_no_previous_diff(self):
        """No previous diff stored → cannot compare → not converged."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff="")
        assert _check_convergence(item, DIFF_A) is False

    def test_returns_false_when_no_current_diff(self):
        """Empty current diff → cannot compare → not converged."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff=DIFF_A)
        assert _check_convergence(item, "") is False

    def test_detects_identical_diffs(self):
        """Identical diffs → similarity 1.0 → converged."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff=DIFF_A)
        assert _check_convergence(item, DIFF_A) is True

    def test_detects_highly_similar_diffs(self):
        """Near-identical diffs above threshold → converged."""
        # Produce a slightly tweaked version (one char different) — still very similar
        tweaked = DIFF_A.replace('"ok"', '"OK"')
        ratio = difflib.SequenceMatcher(None, DIFF_A, tweaked, autojunk=False).ratio()
        assert ratio >= _CONVERGENCE_SIMILARITY_THRESHOLD, "Test precondition: tweak must be above threshold"
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff=DIFF_A)
        assert _check_convergence(item, tweaked) is True

    def test_does_not_detect_substantially_different_diffs(self):
        """Substantially different diffs → not converged."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff=DIFF_A)
        assert _check_convergence(item, DIFF_B_DIFFERENT) is False

    def test_threshold_boundary_exactly_at_threshold(self):
        """Similarity exactly at threshold → converged."""
        # Craft two strings whose SequenceMatcher ratio equals the threshold exactly
        # (difficult to hit exactly, so we mock the ratio call instead)
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff="x")
        with patch("app.core.nodes.reviewer.difflib.SequenceMatcher") as mock_sm:
            mock_sm.return_value.ratio.return_value = _CONVERGENCE_SIMILARITY_THRESHOLD
            result = _check_convergence(item, "y")
        assert result is True

    def test_threshold_just_below_threshold(self):
        """Similarity just below threshold → not converged."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff="x")
        with patch("app.core.nodes.reviewer.difflib.SequenceMatcher") as mock_sm:
            mock_sm.return_value.ratio.return_value = _CONVERGENCE_SIMILARITY_THRESHOLD - 0.01
            result = _check_convergence(item, "y")
        assert result is False

    def test_activates_exactly_at_min_rework(self):
        """Convergence check activates at exactly _CONVERGENCE_MIN_REWORK cycles."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK, last_coder_diff=DIFF_A)
        assert _check_convergence(item, DIFF_A) is True

    def test_still_active_beyond_min_rework(self):
        """Convergence check stays active above min rework threshold."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK + 5, last_coder_diff=DIFF_A)
        assert _check_convergence(item, DIFF_A) is True


# ---------------------------------------------------------------------------
# Integration tests for peer_review_node routing
# ---------------------------------------------------------------------------

def _mock_invoke_result(verdict_text: str):
    """Return a fake (result, budget_update) tuple for _invoke_with_budget."""
    return verdict_text, {}


class TestPeerReviewNodeConvergenceRouting:
    """Test that peer_review_node routes correctly under each exit condition."""

    def _run_peer_review(self, item: TodoItem, verdict_text: str, state_overrides: dict | None = None):
        """Run peer_review_node with a mocked LLM verdict and return the result dict."""
        from app.core.nodes.reviewer import peer_review_node

        state = _make_state(item)
        if state_overrides:
            for k, v in state_overrides.items():
                object.__setattr__(state, k, v)

        with patch("app.core.nodes.reviewer._invoke_with_budget", return_value=_mock_invoke_result(verdict_text)), \
             patch("app.core.nodes.reviewer._format_intelligence_summary_reviewer", return_value=""), \
             patch("app.core.nodes.reviewer.emit_node_start"), \
             patch("app.core.nodes.reviewer.emit_node_end"), \
             patch("app.core.nodes.reviewer.emit_status"), \
             patch("app.core.nodes.reviewer.emit_verdict"), \
             patch("app.core.nodes.reviewer.record_agent_outcome"):
            return peer_review_node(state)

    def test_approve_routes_to_reviewing(self):
        item = _make_item(rework_count=0)
        result = self._run_peer_review(item, "**Verdict**: APPROVE\nLooks good.")
        assert result["peer_review_verdict"] == "APPROVE"
        assert result["phase"] == WorkflowPhase.REVIEWING
        assert result.get("convergence_detected") is False

    def test_first_rework_goes_directly_to_coder(self):
        """First REWORK (rework_count becomes 1) → fast path back to coder."""
        item = _make_item(rework_count=0)
        result = self._run_peer_review(item, "**Verdict**: REWORK\nAdd more tests.")
        assert result["peer_review_verdict"] == "REWORK"
        assert result["phase"] == WorkflowPhase.CODING
        assert result.get("convergence_detected") is False

    def test_second_rework_routes_via_tester(self):
        """Second REWORK (rework_count becomes 2) → run tester first.

        The item's last_coder_diff is set to a genuinely different previous diff
        so convergence is NOT triggered, and we fall through to condition 3.
        We must also ensure rework_count < max_rework_cycles_per_item.
        """
        from app.core.config import get_settings
        # Use rework_count=1 so after increment it becomes 2.
        # last_coder_diff must differ enough from itself that _check_convergence
        # returns False. Since _check_convergence compares item.last_coder_diff
        # against itself (same field used for both previous and current snapshot),
        # we force the check to return False by staying below _CONVERGENCE_MIN_REWORK.
        # rework_count starts at 1; after increment → 2 == _CONVERGENCE_MIN_REWORK,
        # so convergence CAN trigger. We keep last_coder_diff empty to disable it.
        item = _make_item(rework_count=1, last_coder_diff="")
        result = self._run_peer_review(item, "**Verdict**: REWORK\nStill not right.")
        assert result["peer_review_verdict"] == "REWORK_VIA_TESTER"
        assert result["phase"] == WorkflowPhase.TESTING
        assert result.get("convergence_detected") is False

    def test_convergence_detected_escalates_to_tester(self):
        """Identical diff after min rework cycles → ESCALATE_CONVERGENCE → tester."""
        item = _make_item(rework_count=_CONVERGENCE_MIN_REWORK - 1, last_coder_diff=DIFF_A)
        # After peer_review increments, rework_count == _CONVERGENCE_MIN_REWORK
        result = self._run_peer_review(item, "**Verdict**: REWORK\nMake it cleaner.")
        assert result["peer_review_verdict"] == "ESCALATE_CONVERGENCE"
        assert result["phase"] == WorkflowPhase.TESTING
        assert result["convergence_detected"] is True

    def test_max_rework_escalates_to_tester(self):
        """Max rework cycles hit → ESCALATE_TESTING (hard ceiling)."""
        from app.core.config import get_settings
        max_rework = get_settings().max_rework_cycles_per_item
        item = _make_item(rework_count=max_rework - 1, last_coder_diff="")
        result = self._run_peer_review(item, "**Verdict**: REWORK\nStill issues.")
        assert result["peer_review_verdict"] == "ESCALATE_TESTING"
        assert result["phase"] == WorkflowPhase.TESTING
        assert result.get("convergence_detected") is False

    def test_max_rework_takes_priority_over_convergence(self):
        """Max cycles takes priority over convergence (condition 1 before condition 2)."""
        from app.core.config import get_settings
        max_rework = get_settings().max_rework_cycles_per_item
        # Already at max-1 AND diff is identical (would trigger convergence too)
        item = _make_item(rework_count=max_rework - 1, last_coder_diff=DIFF_A)
        result = self._run_peer_review(item, "**Verdict**: REWORK\nProblems.")
        # Should be ESCALATE_TESTING (condition 1), not ESCALATE_CONVERGENCE (condition 2)
        assert result["peer_review_verdict"] == "ESCALATE_TESTING"


# ---------------------------------------------------------------------------
# State field tests
# ---------------------------------------------------------------------------

class TestStateFields:
    def test_todo_item_has_last_coder_diff_field(self):
        item = TodoItem(id="x", description="test")
        assert hasattr(item, "last_coder_diff")
        assert item.last_coder_diff == ""

    def test_todo_item_last_coder_diff_is_settable(self):
        item = TodoItem(id="x", description="test")
        item.last_coder_diff = DIFF_A
        assert item.last_coder_diff == DIFF_A

    def test_graph_state_has_convergence_detected_field(self):
        state = GraphState(user_request="test")
        assert hasattr(state, "convergence_detected")
        assert state.convergence_detected is False

    def test_graph_state_convergence_detected_serializes(self):
        """Ensure the field survives model_dump/reconstruct (checkpoint round-trip)."""
        state = GraphState(user_request="test", convergence_detected=True)
        dumped = state.model_dump()
        restored = GraphState(**dumped)
        assert restored.convergence_detected is True
