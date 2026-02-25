"""Tests for Issue #48 — Auto PR/MR creation after commit and push.

Covers:
- PRResult model (state.py)
- GraphState.pr_result field
- Config: auto_create_pr key
- _build_pr_repo_path helper
- _create_pr_for_branch: happy path (GitHub + GitLab), opt-out, missing repo_ref
- _create_pr_for_branch: PR body contains task description, files, issue link
- _create_pr_for_branch: forge failure is non-fatal
- _try_post_pr_link_on_issue: posts comment; failure non-fatal
- committer_node: pr_result in return dict when all items done
- committer_node: no pr_result when has_more items
- committer_node: pr_result absent when auto_create_pr=False
- emit_pr_created: event emitted with correct metadata
- StatusResponse: pr_result field
- human_gate_node: will_create_pr in payload
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from app.core.state import (
    GraphState,
    IssueRef,
    ItemStatus,
    PRResult,
    TodoItem,
    WorkflowPhase,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_done_item(idx: int = 0) -> TodoItem:
    return TodoItem(
        id=f"item_{idx}",
        description="Fix something",
        status=ItemStatus.DONE,
        commit_message=f"fix(auth): resolve null pointer #{idx}",
    )


def _completed_state(**kwargs) -> GraphState:
    """Minimal GraphState with all items done, ready for PR creation."""
    defaults = dict(
        user_request="fix the null pointer bug",
        repo_root="/tmp/fake-repo",
        repo_ref="owner/repo",
        branch_name="daedalus/fix-auth-null-pointer",
        todo_items=[_make_done_item(0)],
        current_item_index=0,
        phase=WorkflowPhase.COMMITTING,
    )
    defaults.update(kwargs)
    return GraphState(**defaults)


def _mock_forge_client(pr_url="https://github.com/owner/repo/pull/17", pr_number=17):
    from infra.forge import PRResult as ForgePRResult
    client = MagicMock()
    client.create_pr.return_value = ForgePRResult(id=17, url=pr_url, number=pr_number)
    client.post_comment.return_value = None
    return client


def _mock_settings(auto_create_pr=True):
    return SimpleNamespace(
        auto_create_pr=auto_create_pr,
        target_repo_path="",
        github_token="ghp_test",
        gitlab_token="",
        gitlab_url="https://gitlab.com",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. PRResult model
# ═══════════════════════════════════════════════════════════════════════════

class TestPRResultModel:

    def test_basic_fields(self):
        r = PRResult(url="https://github.com/o/r/pull/1", number=1, platform="github")
        assert r.url == "https://github.com/o/r/pull/1"
        assert r.number == 1
        assert r.platform == "github"

    def test_platform_defaults_empty(self):
        r = PRResult(url="https://x.com", number=5)
        assert r.platform == ""

    def test_gitlab_platform(self):
        r = PRResult(url="https://gitlab.com/o/r/-/merge_requests/3", number=3, platform="gitlab")
        assert r.platform == "gitlab"

    def test_round_trip_model_dump(self):
        r = PRResult(url="https://x.com", number=7, platform="github")
        d = r.model_dump()
        r2 = PRResult(**d)
        assert r2 == r

    def test_graphstate_pr_result_field_exists(self):
        s = GraphState(user_request="test")
        assert s.pr_result is None

    def test_graphstate_pr_result_stored(self):
        r = PRResult(url="https://x.com/pull/1", number=1, platform="github")
        s = GraphState(user_request="test", pr_result=r)
        assert s.pr_result == r

    def test_graphstate_pr_result_survives_model_dump(self):
        r = PRResult(url="https://x.com/pull/2", number=2, platform="gitlab")
        s = GraphState(user_request="test", pr_result=r)
        d = s.model_dump()
        s2 = GraphState(**d)
        assert s2.pr_result == r


# ═══════════════════════════════════════════════════════════════════════════
# 2. Config: auto_create_pr
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoCreatePrConfig:

    def test_key_exists(self):
        from app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "auto_create_pr")

    def test_default_is_true(self):
        from app.core.config import get_settings
        s = get_settings()
        assert s.auto_create_pr is True

    def test_env_override_false(self, monkeypatch):
        monkeypatch.setenv("AUTO_CREATE_PR", "false")
        from app.core import config as cfg_mod
        settings = cfg_mod.Settings()
        assert settings.auto_create_pr is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. _build_pr_repo_path helper
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPrRepoPath:

    def test_full_https_url(self):
        from app.core.nodes import _build_pr_repo_path
        url, path = _build_pr_repo_path("https://github.com/owner/repo")
        assert url == "https://github.com/owner/repo"
        assert path == "owner/repo"

    def test_host_qualified_path(self):
        from app.core.nodes import _build_pr_repo_path
        url, path = _build_pr_repo_path("github.com/owner/repo")
        assert url == "https://github.com/owner/repo"
        assert path == "owner/repo"

    def test_short_owner_repo(self):
        from app.core.nodes import _build_pr_repo_path
        url, path = _build_pr_repo_path("owner/repo")
        assert "github.com" in url
        assert path == "owner/repo"

    def test_gitlab_self_hosted(self):
        from app.core.nodes import _build_pr_repo_path
        url, path = _build_pr_repo_path("gitlab.internal/team/project")
        assert url == "https://gitlab.internal/team/project"
        assert path == "team/project"

    def test_empty_returns_empty(self):
        from app.core.nodes import _build_pr_repo_path
        url, path = _build_pr_repo_path("")
        assert url == ""
        assert path == ""


# ═══════════════════════════════════════════════════════════════════════════
# 4. _create_pr_for_branch — happy path
# ═══════════════════════════════════════════════════════════════════════════

class TestCreatePrForBranch:

    def test_returns_pr_result_on_success(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state()
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is not None
        assert result.number == 17
        assert result.url == "https://github.com/owner/repo/pull/17"
        assert result.platform == "github"

    def test_create_pr_called_with_correct_args(self):
        from app.core.nodes import _create_pr_for_branch
        from infra.forge import PRRequest

        state = _completed_state(
            branch_name="daedalus/fix-auth",
            repo_ref="owner/repo",
        )
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        client.create_pr.assert_called_once()
        api_path, pr_req = client.create_pr.call_args[0]
        assert api_path == "owner/repo"
        assert pr_req.head_branch == "daedalus/fix-auth"
        assert isinstance(pr_req, PRRequest)

    def test_gitlab_platform_detection(self):
        from app.core.nodes import _create_pr_for_branch
        from infra.forge import PRResult as ForgePRResult

        state = _completed_state(repo_ref="gitlab.com/group/project")
        client = MagicMock()
        client.create_pr.return_value = ForgePRResult(
            id=3, url="https://gitlab.com/group/project/-/merge_requests/3", number=3
        )

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is not None
        assert result.platform == "gitlab"

    def test_pr_body_contains_task_description(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state(user_request="Fix the null pointer crash in auth middleware")
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        _, pr_req = client.create_pr.call_args[0]
        assert "Fix the null pointer crash" in pr_req.body

    def test_pr_body_contains_closes_issue_when_issue_ref_set(self):
        from app.core.nodes import _create_pr_for_branch

        issue_ref = IssueRef(repo_ref="owner/repo", issue_id=42, platform="github")
        state = _completed_state(issue_ref=issue_ref)
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        _, pr_req = client.create_pr.call_args[0]
        assert "Closes #42" in pr_req.body

    def test_pr_body_no_issue_link_when_no_issue_ref(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state(issue_ref=None)
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        _, pr_req = client.create_pr.call_args[0]
        assert "Closes #" not in pr_req.body

    def test_pr_title_from_commit_message(self):
        from app.core.nodes import _create_pr_for_branch

        item = _make_done_item(0)
        item.commit_message = "fix(auth): resolve null pointer"
        state = _completed_state(todo_items=[item])
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        _, pr_req = client.create_pr.call_args[0]
        assert "resolve null pointer" in pr_req.title


# ═══════════════════════════════════════════════════════════════════════════
# 5. _create_pr_for_branch — opt-out and edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestCreatePrEdgeCases:

    def test_returns_none_when_auto_create_pr_false(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state()
        with patch("app.core.nodes.get_settings", return_value=_mock_settings(auto_create_pr=False)):
            result = _create_pr_for_branch(state)

        assert result is None

    def test_returns_none_when_repo_ref_empty(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state(repo_ref="")
        with patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is None

    def test_returns_none_when_branch_name_empty(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state(branch_name="")
        with patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is None

    def test_forge_error_returns_none_not_raises(self):
        from app.core.nodes import _create_pr_for_branch
        from infra.forge import ForgeError

        state = _completed_state()
        with patch("infra.factory.get_forge_client", side_effect=ForgeError("auth failed")), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is None

    def test_forge_create_pr_failure_returns_none(self):
        from app.core.nodes import _create_pr_for_branch

        state = _completed_state()
        client = MagicMock()
        client.create_pr.side_effect = RuntimeError("network timeout")

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = _create_pr_for_branch(state)

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. _try_post_pr_link_on_issue
# ═══════════════════════════════════════════════════════════════════════════

class TestTryPostPrLinkOnIssue:

    def test_posts_pr_link_comment(self):
        from app.core.nodes import _try_post_pr_link_on_issue

        client = MagicMock()
        _try_post_pr_link_on_issue(
            client, "owner/repo", 42, "https://github.com/owner/repo/pull/17", 17, "github"
        )

        client.post_comment.assert_called_once()
        args = client.post_comment.call_args[0]
        assert args[0] == "owner/repo"
        assert args[1] == 42
        assert "PR #17" in args[2]
        assert "https://github.com/owner/repo/pull/17" in args[2]

    def test_uses_mr_label_for_gitlab(self):
        from app.core.nodes import _try_post_pr_link_on_issue

        client = MagicMock()
        _try_post_pr_link_on_issue(
            client, "group/proj", 5, "https://gitlab.com/group/proj/-/merge_requests/3", 3, "gitlab"
        )
        args = client.post_comment.call_args[0]
        assert "MR #3" in args[2]

    def test_post_failure_does_not_raise(self):
        from app.core.nodes import _try_post_pr_link_on_issue

        client = MagicMock()
        client.post_comment.side_effect = Exception("rate limit")
        # Must not raise
        _try_post_pr_link_on_issue(
            client, "owner/repo", 1, "https://x.com", 1, "github"
        )

    def test_issue_triggered_pr_posts_comment_to_issue(self):
        """When issue_ref is set, _create_pr_for_branch posts the PR link on the issue."""
        from app.core.nodes import _create_pr_for_branch

        issue_ref = IssueRef(repo_ref="owner/repo", issue_id=42, platform="github")
        state = _completed_state(issue_ref=issue_ref)
        client = _mock_forge_client()

        with patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            _create_pr_for_branch(state)

        # post_comment called at least once (once for PR link on issue)
        assert client.post_comment.call_count >= 1
        # Check one of the calls mentions the PR
        pr_link_calls = [
            c for c in client.post_comment.call_args_list
            if "PR" in str(c) or "pull/17" in str(c)
        ]
        assert len(pr_link_calls) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. emit_pr_created event
# ═══════════════════════════════════════════════════════════════════════════

class TestEmitPrCreated:

    def test_emit_pr_created_fires(self):
        from app.core.events import emit_pr_created, subscribe_sync, clear_listeners
        from app.core.events import WorkflowEvent

        received = []
        subscribe_sync(lambda e: received.append(e))
        try:
            emit_pr_created(
                url="https://github.com/o/r/pull/1",
                number=1,
                platform="github",
                branch="daedalus/feat",
            )
        finally:
            clear_listeners()

        assert len(received) == 1
        evt = received[0]
        assert evt.metadata["pr_url"] == "https://github.com/o/r/pull/1"
        assert evt.metadata["pr_number"] == 1
        assert evt.metadata["platform"] == "github"
        assert evt.metadata["branch"] == "daedalus/feat"

    def test_emit_mr_label_for_gitlab(self):
        from app.core.events import emit_pr_created, subscribe_sync, clear_listeners

        received = []
        subscribe_sync(lambda e: received.append(e))
        try:
            emit_pr_created(
                url="https://gitlab.com/g/p/-/merge_requests/3",
                number=3,
                platform="gitlab",
                branch="feat-branch",
            )
        finally:
            clear_listeners()

        assert len(received) == 1
        evt = received[0]
        assert evt.metadata["label"] == "MR"
        assert evt.metadata["platform"] == "gitlab"


# ═══════════════════════════════════════════════════════════════════════════
# 8. committer_node integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCommitterNodePR:

    def _make_committer_state(self, n_items=1, repo_ref="owner/repo", issue_ref=None):
        items = [_make_done_item(i) for i in range(n_items)]
        return GraphState(
            user_request="fix bugs",
            repo_root="/tmp/fake",
            repo_ref=repo_ref,
            branch_name="daedalus/fix-bugs",
            todo_items=items,
            current_item_index=n_items - 1,
            phase=WorkflowPhase.COMMITTING,
            needs_human_approval=False,
            pending_approval={"approved": True},
            issue_ref=issue_ref,
        )

    def test_pr_result_in_return_when_all_done(self):
        from app.core import nodes

        state = self._make_committer_state(n_items=1)
        client = _mock_forge_client()

        with patch.object(nodes, "git_commit_and_push") as mock_push, \
             patch.object(nodes, "git_command", return_value=""), \
             patch("infra.factory.get_forge_client", return_value=client), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            mock_push.invoke = MagicMock(return_value="[main abc1234] fix commit")
            result = nodes.committer_node(state)

        assert "pr_result" in result
        assert result["pr_result"].number == 17
        assert result["phase"] == WorkflowPhase.COMPLETE

    def test_pr_result_absent_when_has_more_items(self):
        from app.core import nodes

        state = self._make_committer_state(n_items=2)
        # current_item_index = 1 (last), but n_items = 2 → has_more = False
        # Let's test with index=0 (first item, more remain)
        state = GraphState(
            user_request="fix bugs",
            repo_root="/tmp/fake",
            repo_ref="owner/repo",
            branch_name="daedalus/fix-bugs",
            todo_items=[_make_done_item(0), _make_done_item(1)],
            current_item_index=0,   # item 0 done, item 1 still pending
            phase=WorkflowPhase.COMMITTING,
            pending_approval={"approved": True},
        )

        with patch.object(nodes, "git_commit_and_push") as mock_push, \
             patch.object(nodes, "git_command", return_value=""):
            mock_push.invoke = MagicMock(return_value="[main abc] commit")
            result = nodes.committer_node(state)

        # Should NOT have pr_result (more items to process)
        assert "pr_result" not in result
        assert result.get("phase") == WorkflowPhase.CODING

    def test_no_pr_when_auto_create_pr_false(self):
        from app.core import nodes

        state = self._make_committer_state(n_items=1)

        with patch.object(nodes, "git_commit_and_push") as mock_push, \
             patch.object(nodes, "git_command", return_value=""), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings(auto_create_pr=False)):
            mock_push.invoke = MagicMock(return_value="pushed")
            result = nodes.committer_node(state)

        # pr_result key may be absent or None
        assert result.get("pr_result") is None

    def test_no_pr_when_repo_ref_missing(self):
        from app.core import nodes

        state = self._make_committer_state(n_items=1, repo_ref="")

        with patch.object(nodes, "git_commit_and_push") as mock_push, \
             patch.object(nodes, "git_command", return_value=""), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            mock_push.invoke = MagicMock(return_value="pushed")
            result = nodes.committer_node(state)

        assert result.get("pr_result") is None

    def test_forge_failure_still_completes_workflow(self):
        from app.core import nodes
        from infra.forge import ForgeError

        state = self._make_committer_state(n_items=1)

        with patch.object(nodes, "git_commit_and_push") as mock_push, \
             patch.object(nodes, "git_command", return_value=""), \
             patch("infra.factory.get_forge_client", side_effect=ForgeError("auth")), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            mock_push.invoke = MagicMock(return_value="pushed")
            result = nodes.committer_node(state)

        # Workflow completes even when PR creation fails
        assert result["phase"] == WorkflowPhase.COMPLETE
        assert result.get("pr_result") is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. human_gate_node: will_create_pr in payload
# ═══════════════════════════════════════════════════════════════════════════

class TestHumanGatePRHint:

    def test_will_create_pr_true_when_repo_ref_set(self):
        from app.core import nodes

        state = GraphState(
            user_request="fix stuff",
            repo_root="/tmp/fake",
            repo_ref="owner/repo",
            branch_name="daedalus/fix",
            phase=WorkflowPhase.COMMITTING,
            needs_human_approval=False,
        )

        with patch.object(nodes, "git_command", return_value="M app/main.py"), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings(auto_create_pr=True)):
            result = nodes.human_gate_node(state)

        payload = result.get("pending_approval", {})
        assert payload.get("will_create_pr") is True

    def test_will_create_pr_false_when_opt_out(self):
        from app.core import nodes

        state = GraphState(
            user_request="fix stuff",
            repo_root="/tmp/fake",
            repo_ref="owner/repo",
            branch_name="daedalus/fix",
            phase=WorkflowPhase.COMMITTING,
            needs_human_approval=False,
        )

        with patch.object(nodes, "git_command", return_value="M app/main.py"), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings(auto_create_pr=False)):
            result = nodes.human_gate_node(state)

        payload = result.get("pending_approval", {})
        assert payload.get("will_create_pr") is False

    def test_will_create_pr_false_when_no_repo_ref(self):
        from app.core import nodes

        state = GraphState(
            user_request="fix stuff",
            repo_root="/tmp/fake",
            repo_ref="",
            branch_name="daedalus/fix",
            phase=WorkflowPhase.COMMITTING,
            needs_human_approval=False,
        )

        with patch.object(nodes, "git_command", return_value="M app/main.py"), \
             patch("app.core.nodes.get_settings", return_value=_mock_settings()):
            result = nodes.human_gate_node(state)

        payload = result.get("pending_approval", {})
        assert payload.get("will_create_pr") is False


# ═══════════════════════════════════════════════════════════════════════════
# 10. StatusResponse: pr_result field
# ═══════════════════════════════════════════════════════════════════════════

class TestStatusResponsePrResult:

    def test_pr_result_field_exists(self):
        from app.web.server import StatusResponse
        sr = StatusResponse(
            phase="complete", progress="1/1", branch="daedalus/feat", error="",
            items_total=1, items_done=1,
            pr_result={"url": "https://github.com/o/r/pull/1", "number": 1, "platform": "github"},
        )
        assert sr.pr_result is not None
        assert sr.pr_result["number"] == 1

    def test_pr_result_defaults_none(self):
        from app.web.server import StatusResponse
        sr = StatusResponse(
            phase="idle", progress="", branch="", error="",
            items_total=0, items_done=0,
        )
        assert sr.pr_result is None

    def test_get_status_includes_pr_result(self):
        """GET /api/status serialises pr_result when state has it."""
        from app.web import server as srv
        from app.core.state import WorkflowPhase

        pr = PRResult(url="https://github.com/o/r/pull/5", number=5, platform="github")
        fake_state = GraphState(
            user_request="test",
            phase=WorkflowPhase.COMPLETE,
            pr_result=pr,
        )
        # Patch _current_state
        original = srv._current_state
        srv._current_state = fake_state
        try:
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(srv.get_status())
            assert result.pr_result is not None
            assert result.pr_result["number"] == 5
        finally:
            srv._current_state = original
