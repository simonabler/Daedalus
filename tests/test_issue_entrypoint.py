"""Tests for Issue #47 — Issue-to-task entrypoint.

Covers:
- parse_issue_ref: URL patterns, bare issue+repo, #N shorthand
- IssueRef model
- router_node: issue detection → code intent
- _hydrate_issue: forge client called, enriched request, comment posted
- context_loader_node: hydration invoked when issue_ref set
- run_workflow: issue_ref forwarded to initial state
- StatusResponse: issue_ref serialised
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from app.core.state import IssueRef, GraphState
from app.core.task_routing import parse_issue_ref


# ═══════════════════════════════════════════════════════════════════════════
# 1. parse_issue_ref — URL detection
# ═══════════════════════════════════════════════════════════════════════════

class TestParseIssueRefUrl:

    def test_github_issue_url(self):
        ref = parse_issue_ref("Please fix https://github.com/owner/repo/issues/42")
        assert ref is not None
        assert ref.issue_id == 42
        assert "github.com/owner/repo" in ref.repo_ref
        assert ref.platform == "github"

    def test_gitlab_com_issue_url(self):
        ref = parse_issue_ref("See https://gitlab.com/group/project/-/issues/7")
        assert ref is not None
        assert ref.issue_id == 7
        assert "gitlab.com" in ref.repo_ref
        assert ref.platform == "gitlab"

    def test_gitlab_self_hosted_issue_url(self):
        ref = parse_issue_ref("fix https://gitlab.internal/team/proj/-/issues/99")
        assert ref is not None
        assert ref.issue_id == 99
        assert "gitlab.internal" in ref.repo_ref
        assert ref.platform == "gitlab"

    def test_github_url_without_dash(self):
        # github uses /issues/ directly (no /-/)
        ref = parse_issue_ref("https://github.com/octocat/Hello-World/issues/1")
        assert ref is not None
        assert ref.issue_id == 1

    def test_url_in_middle_of_sentence(self):
        ref = parse_issue_ref("Can you look at https://github.com/org/repo/issues/5 and fix it?")
        assert ref is not None
        assert ref.issue_id == 5

    def test_url_detection_takes_precedence_over_hash(self):
        ref = parse_issue_ref("fix #3 and https://github.com/org/repo/issues/42")
        assert ref is not None
        # URL match wins
        assert ref.issue_id == 42

    def test_subgroup_gitlab_url(self):
        ref = parse_issue_ref("https://gitlab.com/group/sub/proj/-/issues/12")
        assert ref is not None
        assert ref.issue_id == 12
        assert "group/sub/proj" in ref.repo_ref


# ═══════════════════════════════════════════════════════════════════════════
# 2. parse_issue_ref — "issue N in repo" pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestParseIssueRefBare:

    def test_issue_N_in_repo(self):
        ref = parse_issue_ref("fix issue 42 in owner/repo")
        assert ref is not None
        assert ref.issue_id == 42
        assert ref.repo_ref == "owner/repo"

    def test_issue_N_for_repo(self):
        ref = parse_issue_ref("implement issue 7 for my-org/my-api")
        assert ref is not None
        assert ref.issue_id == 7
        assert "my-org/my-api" in ref.repo_ref

    def test_issue_N_on_repo(self):
        ref = parse_issue_ref("close issue 3 on team/project")
        assert ref is not None
        assert ref.issue_id == 3

    def test_issue_keyword_case_insensitive(self):
        ref = parse_issue_ref("ISSUE 99 in org/repo")
        assert ref is not None
        assert ref.issue_id == 99


# ═══════════════════════════════════════════════════════════════════════════
# 3. parse_issue_ref — #N shorthand
# ═══════════════════════════════════════════════════════════════════════════

class TestParseIssueRefHash:

    def test_hash_n_with_fallback_repo(self):
        ref = parse_issue_ref("fix #42", fallback_repo_ref="owner/repo")
        assert ref is not None
        assert ref.issue_id == 42
        assert ref.repo_ref == "owner/repo"

    def test_hash_n_without_fallback_returns_none(self):
        ref = parse_issue_ref("fix #42")
        assert ref is None

    def test_hash_n_in_sentence_with_fallback(self):
        ref = parse_issue_ref("please close #7 thanks", fallback_repo_ref="org/proj")
        assert ref is not None
        assert ref.issue_id == 7

    def test_issue_hash_n_with_fallback(self):
        ref = parse_issue_ref("resolve issue #15", fallback_repo_ref="a/b")
        assert ref is not None
        assert ref.issue_id == 15


# ═══════════════════════════════════════════════════════════════════════════
# 4. parse_issue_ref — no match
# ═══════════════════════════════════════════════════════════════════════════

class TestParseIssueRefNoMatch:

    def test_plain_task_returns_none(self):
        assert parse_issue_ref("add a health check endpoint") is None

    def test_unrelated_url_returns_none(self):
        assert parse_issue_ref("see https://docs.example.com for details") is None

    def test_empty_string_returns_none(self):
        assert parse_issue_ref("") is None

    def test_hash_without_digits_returns_none(self):
        assert parse_issue_ref("add #feature to API") is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. IssueRef model
# ═══════════════════════════════════════════════════════════════════════════

class TestIssueRefModel:

    def test_basic_fields(self):
        ref = IssueRef(repo_ref="owner/repo", issue_id=42)
        assert ref.repo_ref == "owner/repo"
        assert ref.issue_id == 42
        assert ref.platform == ""

    def test_platform_stored(self):
        ref = IssueRef(repo_ref="github.com/o/r", issue_id=1, platform="github")
        assert ref.platform == "github"

    def test_round_trip_model_dump(self):
        ref = IssueRef(repo_ref="o/r", issue_id=5, platform="gitlab")
        d = ref.model_dump()
        ref2 = IssueRef(**d)
        assert ref2 == ref

    def test_graphstate_has_issue_ref_field(self):
        s = GraphState(user_request="test")
        assert s.issue_ref is None

    def test_graphstate_issue_ref_stored(self):
        ref = IssueRef(repo_ref="owner/repo", issue_id=99)
        s = GraphState(user_request="test", issue_ref=ref)
        assert s.issue_ref == ref

    def test_graphstate_issue_ref_survives_model_dump(self):
        ref = IssueRef(repo_ref="owner/repo", issue_id=7)
        s = GraphState(user_request="test", issue_ref=ref)
        d = s.model_dump()
        s2 = GraphState(**d)
        assert s2.issue_ref == ref


# ═══════════════════════════════════════════════════════════════════════════
# 6. router_node — issue detection
# ═══════════════════════════════════════════════════════════════════════════

class TestRouterNodeIssueDetection:

    def test_router_detects_github_issue_url(self):
        from app.core import nodes

        state = GraphState(user_request="fix https://github.com/owner/repo/issues/42")
        result = nodes.router_node(state)

        assert result["input_intent"] == "code"
        assert result["issue_ref"] is not None
        assert result["issue_ref"].issue_id == 42

    def test_router_detects_hash_n_with_repo_ref(self):
        from app.core import nodes

        state = GraphState(
            user_request="fix #5",
            repo_ref="owner/repo",
        )
        result = nodes.router_node(state)

        assert result["input_intent"] == "code"
        assert result["issue_ref"] is not None
        assert result["issue_ref"].issue_id == 5

    def test_router_no_issue_ref_for_plain_task(self):
        from app.core import nodes

        state = GraphState(user_request="add health check endpoint")
        result = nodes.router_node(state)

        assert result.get("issue_ref") is None

    def test_router_issue_ref_sets_repo_ref(self):
        from app.core import nodes

        state = GraphState(user_request="fix https://github.com/myorg/myrepo/issues/3")
        result = nodes.router_node(state)

        assert "myorg/myrepo" in result.get("repo_ref", "")

    def test_router_preserves_existing_issue_ref(self):
        """If state.issue_ref is already set, router must keep it."""
        from app.core import nodes

        existing = IssueRef(repo_ref="owner/repo", issue_id=77)
        state = GraphState(user_request="fix the bug", issue_ref=existing)
        result = nodes.router_node(state)

        assert result["input_intent"] == "code"
        assert result["issue_ref"] == existing


# ═══════════════════════════════════════════════════════════════════════════
# 7. _hydrate_issue: forge client called, enriched request
# ═══════════════════════════════════════════════════════════════════════════

class TestHydrateIssue:

    def _make_state(self, repo_ref="owner/repo", issue_id=42, platform="github"):
        return GraphState(
            user_request="fix the bug",
            repo_ref=repo_ref,
            issue_ref=IssueRef(repo_ref=repo_ref, issue_id=issue_id, platform=platform),
        )

    def _mock_issue(self, title="Fix null pointer", description="It crashes", labels=None, author="alice", url="https://github.com/o/r/issues/42"):
        from infra.forge import Issue
        from datetime import datetime, timezone
        return Issue(
            id=42,
            title=title,
            description=description,
            url=url,
            labels=labels or ["bug"],
            author=author,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_hydrate_enriches_user_request(self):
        from app.core.nodes import _hydrate_issue

        state = self._make_state()
        mock_issue = self._mock_issue()
        mock_client = MagicMock()
        mock_client.get_issue.return_value = mock_issue
        mock_client.post_comment.return_value = None

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            enriched, _ = _hydrate_issue(state)

        assert "Issue #42" in enriched
        assert "Fix null pointer" in enriched
        assert "It crashes" in enriched
        assert "alice" in enriched
        assert "bug" in enriched

    def test_hydrate_calls_get_issue(self):
        from app.core.nodes import _hydrate_issue

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue()

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            _hydrate_issue(state)

        mock_client.get_issue.assert_called_once_with("owner/repo", 42)

    def test_hydrate_posts_working_comment(self):
        from app.core.nodes import _hydrate_issue

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue()

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            _hydrate_issue(state)

        # comment must have been posted
        mock_client.post_comment.assert_called_once()
        args = mock_client.post_comment.call_args[0]
        assert args[0] == "owner/repo"
        assert args[1] == 42
        assert "Daedalus" in args[2]

    def test_hydrate_comment_failure_does_not_raise(self):
        from app.core.nodes import _hydrate_issue

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue()
        mock_client.post_comment.side_effect = Exception("API rate limit")

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            enriched, _ = _hydrate_issue(state)  # must not raise

        assert "Issue #42" in enriched

    def test_hydrate_forge_failure_falls_back_to_original(self):
        from app.core.nodes import _hydrate_issue
        from infra.forge import ForgeError

        state = self._make_state()

        with patch("infra.factory.get_forge_client", side_effect=ForgeError("auth failed")):
            enriched, _ = _hydrate_issue(state)

        assert enriched == state.user_request

    def test_hydrate_no_issue_ref_returns_original(self):
        from app.core.nodes import _hydrate_issue

        state = GraphState(user_request="plain task")
        enriched, _ = _hydrate_issue(state)
        assert enriched == "plain task"

    def test_hydrate_includes_issue_url_in_enriched(self):
        from app.core.nodes import _hydrate_issue

        state = self._make_state()
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue(url="https://github.com/owner/repo/issues/42")

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            enriched, _ = _hydrate_issue(state)

        assert "https://github.com/owner/repo/issues/42" in enriched

    def test_hydrate_gitlab_uses_correct_repo_path(self):
        """For gitlab.com/group/project, the repo_path passed to forge client is group/project."""
        from app.core.nodes import _hydrate_issue

        state = self._make_state(
            repo_ref="gitlab.com/group/project",
            platform="gitlab",
        )
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue()

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            _hydrate_issue(state)

        call_args = mock_client.get_issue.call_args[0]
        # repo path must be "group/project" (host stripped)
        assert call_args[0] == "group/project"

    def test_hydrate_short_form_github(self):
        """Short owner/repo form → github.com/owner/repo → get_issue('owner/repo', N)."""
        from app.core.nodes import _hydrate_issue

        state = self._make_state(repo_ref="owner/repo", platform="github")
        mock_client = MagicMock()
        mock_client.get_issue.return_value = self._mock_issue()

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            _hydrate_issue(state)

        call_args = mock_client.get_issue.call_args[0]
        assert call_args[0] == "owner/repo"
        assert call_args[1] == 42


# ═══════════════════════════════════════════════════════════════════════════
# 8. context_loader_node — issue hydration integration
# ═══════════════════════════════════════════════════════════════════════════

class TestContextLoaderIssueHydration:

    def _make_settings(self, tmp_path):
        return SimpleNamespace(
            target_repo_path=str(tmp_path),
            daedalus_workspace_dir=str(tmp_path / "ws"),
            max_output_chars=10000,
            context_warn_fraction=0.7,
            tool_result_max_chars=8000,
        )

    def test_issue_ref_triggers_hydration(self, tmp_path, monkeypatch):
        from app.core import nodes
        from app.core.state import IssueRef

        (tmp_path / "README.md").write_text("# Hello")
        monkeypatch.setattr(nodes, "get_settings", lambda: self._make_settings(tmp_path))

        issue_ref = IssueRef(repo_ref="owner/repo", issue_id=42, platform="github")
        state = GraphState(
            user_request="fix #42",
            repo_root=str(tmp_path),
            issue_ref=issue_ref,
        )

        mock_client = MagicMock()
        from infra.forge import Issue
        from datetime import datetime, timezone
        mock_client.get_issue.return_value = Issue(
            id=42, title="Crash on startup", description="NullPtr in auth",
            url="https://github.com/owner/repo/issues/42",
            labels=["bug"], author="alice",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mock_client.post_comment.return_value = None

        with patch("infra.factory.get_forge_client", return_value=mock_client):
            result = nodes.context_loader_node(state)

        assert result["context_loaded"] is True
        # user_request must be enriched
        assert "Crash on startup" in result["user_request"]
        assert "Issue #42" in result["user_request"]

    def test_no_issue_ref_leaves_request_unchanged(self, tmp_path, monkeypatch):
        from app.core import nodes

        (tmp_path / "README.md").write_text("# Hello")
        monkeypatch.setattr(nodes, "get_settings", lambda: self._make_settings(tmp_path))

        state = GraphState(
            user_request="add health endpoint",
            repo_root=str(tmp_path),
            issue_ref=None,
        )
        result = nodes.context_loader_node(state)

        assert result["user_request"] == "add health endpoint"

    def test_hydration_failure_does_not_stop_workflow(self, tmp_path, monkeypatch):
        from app.core import nodes
        from app.core.state import IssueRef
        from infra.forge import ForgeError

        (tmp_path / "README.md").write_text("# Hello")
        monkeypatch.setattr(nodes, "get_settings", lambda: self._make_settings(tmp_path))

        issue_ref = IssueRef(repo_ref="owner/repo", issue_id=42)
        state = GraphState(
            user_request="fix #42",
            repo_root=str(tmp_path),
            issue_ref=issue_ref,
        )

        with patch("infra.factory.get_forge_client", side_effect=ForgeError("network error")):
            result = nodes.context_loader_node(state)

        # Workflow continues even when forge is down
        assert result["context_loaded"] is True
        # Original request preserved
        assert result["user_request"] == "fix #42"


# ═══════════════════════════════════════════════════════════════════════════
# 9. run_workflow — issue_ref forwarded
# ═══════════════════════════════════════════════════════════════════════════

class TestRunWorkflowIssueRef:

    @pytest.mark.asyncio
    async def test_issue_ref_in_initial_state(self):
        from app.core.orchestrator import run_workflow
        from app.core.state import WorkflowPhase

        ref = IssueRef(repo_ref="owner/repo", issue_id=42, platform="github")
        captured = []

        async def fake_to_thread(fn, state_dict):
            captured.append(state_dict)
            s = GraphState(user_request="t", phase=WorkflowPhase.COMPLETE)
            return s.model_dump()

        with patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch("app.core.orchestrator.compile_graph"):
            await run_workflow("fix #42", "/tmp/repo", repo_ref="owner/repo", issue_ref=ref)

        assert len(captured) == 1
        stored = captured[0].get("issue_ref")
        assert stored is not None
        assert stored["issue_id"] == 42

    @pytest.mark.asyncio
    async def test_none_issue_ref_accepted(self):
        from app.core.orchestrator import run_workflow
        from app.core.state import WorkflowPhase

        captured = []

        async def fake_to_thread(fn, state_dict):
            captured.append(state_dict)
            s = GraphState(user_request="t", phase=WorkflowPhase.COMPLETE)
            return s.model_dump()

        with patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch("app.core.orchestrator.compile_graph"):
            await run_workflow("add endpoint", "/tmp/repo")

        assert captured[0].get("issue_ref") is None


# ═══════════════════════════════════════════════════════════════════════════
# 10. StatusResponse — issue_ref exposed
# ═══════════════════════════════════════════════════════════════════════════

class TestStatusResponseIssueRef:

    def test_status_response_has_issue_ref_field(self):
        from app.web.server import StatusResponse
        sr = StatusResponse(
            phase="planning", progress="0/0", branch="", error="",
            items_total=0, items_done=0,
            issue_ref={"repo_ref": "owner/repo", "issue_id": 42, "platform": "github"},
        )
        assert sr.issue_ref is not None
        assert sr.issue_ref["issue_id"] == 42

    def test_status_response_issue_ref_defaults_none(self):
        from app.web.server import StatusResponse
        sr = StatusResponse(
            phase="idle", progress="", branch="", error="",
            items_total=0, items_done=0,
        )
        assert sr.issue_ref is None
