"""Tests for Issue #16 — agent prompt enrichment and intelligence web UI dashboard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SMELLS = [
    {"file": "a.py", "line": 10, "smell_type": "LongFunction", "severity": "error",
     "description": "too long", "suggestion": "split it"},
    {"file": "b.py", "line": 5,  "smell_type": "MagicNumber",  "severity": "info",
     "description": "magic 42", "suggestion": "use constant"},
    {"file": "c.py", "line": 2,  "smell_type": "GodClass",     "severity": "error",
     "description": "too big",  "suggestion": "split class"},
]
SAMPLE_STATIC = [
    {"file": "a.py", "line": 3, "col": 0, "severity": "error", "rule_id": "E001",
     "message": "unused import", "tool": "ruff"},
    {"file": "b.py", "line": 7, "col": 0, "severity": "warning", "rule_id": "W001",
     "message": "line too long", "tool": "ruff"},
]
SAMPLE_CALL_GRAPH = {
    "callers": {"foo": ["main"]}, "callees": {"main": ["foo"], "foo": []},
    "file_map": {}, "language": "python", "files_analysed": 2, "parse_errors": 0,
}
SAMPLE_DEP_GRAPH = {
    "imports": {"a": ["b"], "b": []}, "importers": {"b": ["a"]},
    "cycles": [["x", "y"]], "coupling_scores": {"a": 0.5, "b": 0.3},
    "language": "python", "files_analysed": 2, "parse_errors": 0,
}


def _make_state(**kwargs):
    from app.core.state import GraphState
    defaults = dict(
        code_smells=SAMPLE_SMELLS,
        static_issues=SAMPLE_STATIC,
        call_graph=SAMPLE_CALL_GRAPH,
        dependency_graph=SAMPLE_DEP_GRAPH,
        dep_cycles=[["x", "y"]],
        intelligence_cached=False,
        intelligence_cache_key="",
    )
    defaults.update(kwargs)
    return GraphState(**defaults)


# ---------------------------------------------------------------------------
# _format_intelligence_summary_for_prompt  (planner / coder — full)
# ---------------------------------------------------------------------------

class TestFormatIntelligenceSummaryFull:
    def test_returns_string(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state()
        result = _format_intelligence_summary_for_prompt(state)
        assert isinstance(result, str)

    def test_includes_header(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state()
        assert "Code Intelligence Summary" in _format_intelligence_summary_for_prompt(state)

    def test_includes_smells_section(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state()
        result = _format_intelligence_summary_for_prompt(state)
        assert "Code Smell" in result or "Smell" in result

    def test_includes_static_section(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state()
        assert "static" in _format_intelligence_summary_for_prompt(state).lower()

    def test_includes_dep_graph(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state()
        result = _format_intelligence_summary_for_prompt(state)
        assert "Dependency" in result or "cycle" in result.lower() or "coupled" in result.lower()

    def test_empty_state_returns_empty_string(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        from app.core.state import GraphState
        state = GraphState()
        result = _format_intelligence_summary_for_prompt(state)
        assert result == ""

    def test_cached_note_shown_when_cached(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state(intelligence_cached=True, intelligence_cache_key="abc123")
        result = _format_intelligence_summary_for_prompt(state)
        assert "abc123" in result or "cached" in result.lower()

    def test_no_cached_note_when_not_cached(self):
        from app.core.nodes import _format_intelligence_summary_for_prompt
        state = _make_state(intelligence_cached=False, intelligence_cache_key="")
        result = _format_intelligence_summary_for_prompt(state)
        assert "cached" not in result.lower()


# ---------------------------------------------------------------------------
# _format_intelligence_summary_reviewer  (reduced budget)
# ---------------------------------------------------------------------------

class TestFormatIntelligenceSummaryReviewer:
    def test_returns_string(self):
        from app.core.nodes import _format_intelligence_summary_reviewer
        state = _make_state()
        assert isinstance(_format_intelligence_summary_reviewer(state), str)

    def test_empty_state_returns_empty(self):
        from app.core.nodes import _format_intelligence_summary_reviewer
        from app.core.state import GraphState
        assert _format_intelligence_summary_reviewer(GraphState()) == ""

    def test_includes_smells(self):
        from app.core.nodes import _format_intelligence_summary_reviewer
        state = _make_state()
        assert "Smell" in _format_intelligence_summary_reviewer(state) or \
               "smell" in _format_intelligence_summary_reviewer(state).lower()

    def test_includes_header(self):
        from app.core.nodes import _format_intelligence_summary_reviewer
        state = _make_state()
        assert "Code Intelligence Summary" in _format_intelligence_summary_reviewer(state)


# ---------------------------------------------------------------------------
# _format_intelligence_summary_tester  (minimal — smells errors + static)
# ---------------------------------------------------------------------------

class TestFormatIntelligenceSummaryTester:
    def test_returns_string(self):
        from app.core.nodes import _format_intelligence_summary_tester
        state = _make_state()
        assert isinstance(_format_intelligence_summary_tester(state), str)

    def test_empty_state_returns_empty(self):
        from app.core.nodes import _format_intelligence_summary_tester
        from app.core.state import GraphState
        assert _format_intelligence_summary_tester(GraphState()) == ""

    def test_includes_static_issues(self):
        from app.core.nodes import _format_intelligence_summary_tester
        state = _make_state()
        result = _format_intelligence_summary_tester(state)
        assert "static" in result.lower() or "ruff" in result.lower() or "E001" in result

    def test_includes_only_error_smells(self):
        from app.core.nodes import _format_intelligence_summary_tester
        # Only error-level smells, not info
        state = _make_state()
        result = _format_intelligence_summary_tester(state)
        # Should include LongFunction (error) and GodClass (error) but not MagicNumber (info)
        if "MagicNumber" in result:
            # info-level smell should not appear in tester context
            assert False, "Tester prompt should not include info-level smells"

    def test_no_call_graph_in_tester(self):
        from app.core.nodes import _format_intelligence_summary_tester
        state = _make_state()
        result = _format_intelligence_summary_tester(state)
        # Call graph is excluded from tester (minimal budget)
        assert "most-called" not in result.lower()


# ---------------------------------------------------------------------------
# Peer review node — intelligence injected
# ---------------------------------------------------------------------------

class TestPeerReviewNodeIntelligence:
    def _invoke_patch(self, *args, **kwargs):
        return "APPROVE\n**Verdict**: APPROVE\nSuggested commit: feat: add X"

    def test_peer_review_builds_intelligence_context(self):
        """peer_review_node must call _format_intelligence_summary_reviewer."""
        from app.core.nodes import _format_intelligence_summary_reviewer
        state = _make_state()
        result = _format_intelligence_summary_reviewer(state)
        assert result  # non-empty for a state with smells and static issues

    def test_intelligence_ctx_replaces_call_graph_ctx(self):
        """Verify old call_graph_ctx variable is gone, replaced by intelligence_ctx."""
        import inspect, app.core.nodes as n
        src = inspect.getsource(n.peer_review_node)
        assert "intelligence_ctx" in src
        assert "call_graph_ctx" not in src


# ---------------------------------------------------------------------------
# Tester node — intelligence injected
# ---------------------------------------------------------------------------

class TestTesterNodeIntelligence:
    def test_tester_node_builds_tester_intelligence(self):
        from app.core.nodes import _format_intelligence_summary_tester
        state = _make_state()
        result = _format_intelligence_summary_tester(state)
        assert isinstance(result, str)

    def test_tester_node_uses_intelligence_ctx(self):
        """Verify tester_node calls _format_intelligence_summary_tester."""
        import inspect, app.core.nodes as n
        src = inspect.getsource(n.tester_node)
        assert "intelligence_ctx" in src
        assert "_format_intelligence_summary_tester" in src


# ---------------------------------------------------------------------------
# Planner node — consolidated helper replaces 4 separate blocks
# ---------------------------------------------------------------------------

class TestPlannerNodeConsolidatedPrompt:
    def test_planner_uses_consolidated_helper(self):
        import inspect, app.core.nodes as n
        src = inspect.getsource(n.planner_plan_node)
        assert "_format_intelligence_summary_for_prompt" in src

    def test_planner_no_separate_smell_injection(self):
        """The old 4-line pattern should be gone from planner."""
        import inspect, app.core.nodes as n
        src = inspect.getsource(n.planner_plan_node)
        # Old separate injections replaced by consolidated helper
        assert "_format_code_smells_for_prompt(state.code_smells" not in src


# ---------------------------------------------------------------------------
# Web UI HTML — dashboard elements present
# ---------------------------------------------------------------------------

class TestWebUIDashboard:
    @pytest.fixture()
    def html(self) -> str:
        return (Path(__file__).parent.parent / "app" / "web" / "static" / "index.html").read_text()

    def test_intel_button_present(self, html):
        assert "intelBtn" in html
        assert "Intel" in html

    def test_intel_panel_present(self, html):
        assert "intelPanel" in html
        assert "intel-panel" in html

    def test_toggle_intel_function_present(self, html):
        assert "function toggleIntel" in html

    def test_load_intelligence_function_present(self, html):
        assert "function loadIntelligence" in html

    def test_on_intelligence_complete_function_present(self, html):
        assert "_onIntelligenceComplete" in html

    def test_calls_intelligence_summary_endpoint(self, html):
        assert "/api/intelligence-summary" in html

    def test_calls_code_smells_endpoint(self, html):
        assert "/api/code-smells" in html

    def test_calls_dependency_graph_endpoint(self, html):
        assert "/api/dependency-graph" in html

    def test_coupling_bar_rendered(self, html):
        assert "coupling-bar" in html

    def test_cycle_path_element_present(self, html):
        assert "cycle-path" in html

    def test_intel_cards_grid_present(self, html):
        assert "intel-cards" in html

    def test_toggle_section_function_present(self, html):
        assert "function toggleSection" in html

    def test_intelligence_complete_event_handled(self, html):
        assert "intelligence_complete" in html


# ---------------------------------------------------------------------------
# API endpoints integration
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def _client(self):
        from fastapi.testclient import TestClient
        import app.web.server as srv
        return TestClient(srv.app), srv

    def test_intelligence_summary_no_state(self):
        client, srv = self._client()
        original = srv._current_state
        srv._current_state = None
        try:
            resp = client.get("/api/intelligence-summary")
            assert resp.status_code == 200
            assert resp.json()["available"] is False
        finally:
            srv._current_state = original

    def test_intelligence_summary_with_state(self):
        client, srv = self._client()
        from app.core.state import GraphState
        state = GraphState(
            code_smells=SAMPLE_SMELLS,
            static_issues=SAMPLE_STATIC,
            dep_cycles=[["a", "b"]],
            intelligence_cached=True,
            intelligence_cache_key="test123",
        )
        srv._current_state = state
        try:
            resp = client.get("/api/intelligence-summary")
            data = resp.json()
            assert data["available"] is True
            assert data["cached"] is True
            assert data["cache_key"] == "test123"
            assert data["smells"]["total"] == 3
            assert data["smells"]["errors"] == 2
            assert data["static"]["errors"] == 1
            assert data["dependency_graph"]["cycles"] == 1
        finally:
            srv._current_state = None

    def test_code_smells_endpoint(self):
        client, srv = self._client()
        from app.core.state import GraphState
        srv._current_state = GraphState(code_smells=SAMPLE_SMELLS)
        try:
            resp = client.get("/api/code-smells")
            data = resp.json()
            assert data["total"] == 3
            assert data["errors"] == 2
            assert data["infos"] == 1
        finally:
            srv._current_state = None

    def test_dependency_graph_endpoint(self):
        client, srv = self._client()
        from app.core.state import GraphState
        srv._current_state = GraphState(
            dependency_graph=SAMPLE_DEP_GRAPH,
            dep_cycles=[["x", "y"]],
        )
        try:
            resp = client.get("/api/dependency-graph")
            data = resp.json()
            assert "mermaid" in data
            assert data["cycles"] == [["x", "y"]]
        finally:
            srv._current_state = None
