"""Tests for code intelligence node and analysis caching — Issue #15."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.analysis.intelligence_cache import (
    cache_path,
    get_commit_hash,
    load_cache,
    save_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def py_repo(tmp_path: Path) -> Path:
    """Minimal valid Python repo (not a git repo by default)."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (tmp_path / "main.py").write_text("def hello():\n    return 42\n")
    (tmp_path / "utils.py").write_text("from main import hello\ndef run():\n    return hello()\n")
    return tmp_path


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Minimal git repo with one commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (tmp_path / "main.py").write_text("def hello():\n    return 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    return tmp_path


# ---------------------------------------------------------------------------
# intelligence_cache module
# ---------------------------------------------------------------------------

class TestIntelligenceCache:
    def test_get_commit_hash_git_repo(self, git_repo):
        key = get_commit_hash(git_repo)
        assert key is not None
        assert len(key) >= 4  # short hash

    def test_get_commit_hash_non_git_returns_none(self, py_repo):
        key = get_commit_hash(py_repo)
        assert key is None

    def test_save_and_load_roundtrip(self, py_repo):
        data = {"static_issues": [], "call_graph": {"callees": {}}, "code_smells": []}
        save_cache(py_repo, "abc123", data)
        loaded = load_cache(py_repo, "abc123")
        assert loaded is not None
        assert loaded == data

    def test_load_missing_key_returns_none(self, py_repo):
        result = load_cache(py_repo, "nonexistent_key_xyz")
        assert result is None

    def test_load_empty_key_returns_none(self, py_repo):
        result = load_cache(py_repo, "")
        assert result is None

    def test_save_empty_key_is_noop(self, py_repo):
        save_cache(py_repo, "", {"data": 1})
        assert not (py_repo / ".daedalus" / "intelligence_cache" / ".json").exists()

    def test_cache_file_created_in_correct_location(self, py_repo):
        save_cache(py_repo, "testkey", {"x": 1})
        expected = py_repo / ".daedalus" / "intelligence_cache" / "testkey.json"
        assert expected.exists()

    def test_cache_contains_valid_json(self, py_repo):
        save_cache(py_repo, "key1", {"foo": "bar", "n": 42})
        path = cache_path(py_repo, "key1")
        data = json.loads(path.read_text())
        assert data["foo"] == "bar"

    def test_cache_path_helper(self, py_repo):
        p = cache_path(py_repo, "abc")
        assert str(p).endswith("abc.json")
        assert ".daedalus" in str(p)

    def test_corrupt_cache_returns_none(self, py_repo):
        path = cache_path(py_repo, "corrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json {{{{")
        result = load_cache(py_repo, "corrupt")
        assert result is None

    def test_cache_overwrite(self, py_repo):
        save_cache(py_repo, "key2", {"v": 1})
        save_cache(py_repo, "key2", {"v": 2})
        loaded = load_cache(py_repo, "key2")
        assert loaded["v"] == 2


# ---------------------------------------------------------------------------
# code_intelligence_node — unit tests (mocked analysis tools)
# ---------------------------------------------------------------------------

class TestCodeIntelligenceNode:
    def _make_state(self, repo_path: str) -> "GraphState":
        from app.core.state import GraphState
        return GraphState(repo_root=str(repo_path))

    def test_node_returns_all_analysis_fields(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert "static_issues" in result
        assert "call_graph" in result
        assert "dependency_graph" in result
        assert "dep_cycles" in result
        assert "code_smells" in result

    def test_node_returns_cache_fields(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert "intelligence_cache_key" in result
        assert "intelligence_cached" in result

    def test_node_empty_repo_path_returns_empty(self):
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState
        state = GraphState()
        result = code_intelligence_node(state)
        assert result == {}

    def test_node_invalid_path_returns_empty(self):
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState
        state = GraphState(repo_root="/nonexistent/path/xyz")
        result = code_intelligence_node(state)
        assert result == {}

    def test_cache_hit_sets_intelligence_cached_true(self, py_repo):
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState

        # Pre-populate cache
        cached_data = {
            "static_issues": [{"file": "f.py", "line": 1, "severity": "error",
                                "rule_id": "E001", "message": "err", "tool": "ruff", "col": 0}],
            "call_graph": {"callers": {}, "callees": {}, "file_map": {},
                           "language": "python", "files_analysed": 1, "parse_errors": 0},
            "dependency_graph": {"imports": {}, "importers": {}, "cycles": [],
                                  "coupling_scores": {}, "language": "python",
                                  "files_analysed": 1, "parse_errors": 0},
            "dep_cycles": [],
            "code_smells": [],
        }

        with patch("app.analysis.intelligence_cache.get_commit_hash", return_value="abc123"), \
             patch("app.analysis.intelligence_cache.load_cache", return_value=cached_data):
            state = GraphState(repo_root=str(py_repo))
            result = code_intelligence_node(state)

        assert result["intelligence_cached"] is True
        assert result["intelligence_cache_key"] == "abc123"

    def test_cache_miss_sets_intelligence_cached_false(self, py_repo):
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState

        with patch("app.analysis.intelligence_cache.get_commit_hash", return_value="xyz789"), \
             patch("app.analysis.intelligence_cache.load_cache", return_value=None), \
             patch("app.analysis.intelligence_cache.save_cache"):
            state = GraphState(repo_root=str(py_repo))
            result = code_intelligence_node(state)

        assert result["intelligence_cached"] is False
        assert result["intelligence_cache_key"] == "xyz789"

    def test_analysis_saved_to_cache_on_miss(self, py_repo):
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState

        with patch("app.analysis.intelligence_cache.get_commit_hash", return_value="save_test"), \
             patch("app.analysis.intelligence_cache.load_cache", return_value=None) as mock_load, \
             patch("app.analysis.intelligence_cache.save_cache") as mock_save:
            state = GraphState(repo_root=str(py_repo))
            code_intelligence_node(state)

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][2]  # data argument is not empty

    def test_static_issues_are_list_of_dicts(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert isinstance(result["static_issues"], list)
        for item in result["static_issues"]:
            assert isinstance(item, dict)

    def test_call_graph_is_dict(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert isinstance(result["call_graph"], dict)

    def test_code_smells_are_list_of_dicts(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert isinstance(result["code_smells"], list)
        for item in result["code_smells"]:
            assert isinstance(item, dict)
            assert "smell_type" in item
            assert "severity" in item

    def test_dep_cycles_is_list(self, py_repo):
        from app.core.nodes import code_intelligence_node
        state = self._make_state(py_repo)
        result = code_intelligence_node(state)
        assert isinstance(result["dep_cycles"], list)

    def test_node_runs_on_daedalus_itself(self):
        """Self-analysis: node must not crash on the Daedalus repo."""
        from app.core.nodes import code_intelligence_node
        from app.core.state import GraphState
        root = str(Path(__file__).parent.parent)
        state = GraphState(repo_root=root)
        result = code_intelligence_node(state)
        assert "call_graph" in result
        assert result["call_graph"]  # should have found functions


# ---------------------------------------------------------------------------
# WorkflowPhase.ANALYZING
# ---------------------------------------------------------------------------

class TestWorkflowPhaseAnalyzing:
    def test_analyzing_phase_exists(self):
        from app.core.state import WorkflowPhase
        assert hasattr(WorkflowPhase, "ANALYZING")
        assert WorkflowPhase.ANALYZING == "analyzing"


# ---------------------------------------------------------------------------
# GraphState cache fields
# ---------------------------------------------------------------------------

class TestGraphStateCacheFields:
    def test_intelligence_cache_key_default(self):
        from app.core.state import GraphState
        state = GraphState()
        assert state.intelligence_cache_key == ""

    def test_intelligence_cached_default(self):
        from app.core.state import GraphState
        state = GraphState()
        assert state.intelligence_cached is False

    def test_fields_accept_values(self):
        from app.core.state import GraphState
        state = GraphState(intelligence_cache_key="abc123", intelligence_cached=True)
        assert state.intelligence_cache_key == "abc123"
        assert state.intelligence_cached is True

    def test_fields_serialize(self):
        from app.core.state import GraphState
        state = GraphState(intelligence_cache_key="xyz", intelligence_cached=True)
        d = state.model_dump()
        assert d["intelligence_cache_key"] == "xyz"
        assert d["intelligence_cached"] is True


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------

class TestOrchestratorWiring:
    def test_intelligence_node_in_graph(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        # LangGraph exposes nodes via .nodes dict
        assert "intelligence" in graph.nodes

    def test_context_leads_to_intelligence(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        # Check that intelligence is wired (not context → planner directly)
        assert "intelligence" in graph.nodes
        assert "context" in graph.nodes

    def test_code_intelligence_node_importable(self):
        from app.core.nodes import code_intelligence_node
        assert callable(code_intelligence_node)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

class TestIntelligenceSummaryEndpoint:
    def test_endpoint_no_state_returns_unavailable(self):
        from fastapi.testclient import TestClient
        import app.web.server as srv
        original = srv._current_state
        srv._current_state = None
        try:
            client = TestClient(srv.app)
            resp = client.get("/api/intelligence-summary")
            assert resp.status_code == 200
            assert resp.json()["available"] is False
        finally:
            srv._current_state = original

    def test_endpoint_with_state_returns_counts(self):
        from fastapi.testclient import TestClient
        from app.core.state import GraphState
        import app.web.server as srv

        state = GraphState(
            code_smells=[{"severity": "error", "smell_type": "X", "file": "f.py",
                           "line": 1, "description": "d", "suggestion": ""}],
            static_issues=[{"severity": "warning", "file": "f.py", "line": 1,
                             "col": 0, "rule_id": "W1", "message": "w", "tool": "ruff"}],
            dep_cycles=[["a", "b"]],
            intelligence_cached=True,
            intelligence_cache_key="abc",
        )
        srv._current_state = state
        try:
            client = TestClient(srv.app)
            resp = client.get("/api/intelligence-summary")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is True
            assert data["cached"] is True
            assert data["smells"]["errors"] == 1
            assert data["static"]["warnings"] == 1
            assert data["dependency_graph"]["cycles"] == 1
        finally:
            srv._current_state = None
