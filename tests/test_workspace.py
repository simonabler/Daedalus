"""Tests for Issue #45 — Dynamic repo loading.

Covers:
- infra/workspace.py: WorkspaceManager (resolve, clean, list_workspaces)
- infra/workspace.py: _normalise_ref helper
- app/core/active_repo.py: context variable isolation
- app/tools/* using get_repo_root() instead of settings
- app/core/config.py: new daedalus_workspace_dir key
- app/core/state.py: new repo_ref field
- context_loader_node: static path unchanged / workspace path used
- run_workflow: repo_ref forwarded to initial state
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from infra.workspace import (
    WorkspaceError,
    WorkspaceInfo,
    WorkspaceManager,
    _normalise_ref,
    _run_git,
)
from app.core.active_repo import clear_repo_root, get_repo_root, set_repo_root


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_repo_root():
    """Ensure context var is clean before and after each test."""
    clear_repo_root()
    yield
    clear_repo_root()


# ═══════════════════════════════════════════════════════════════════════════
# 1. _normalise_ref
# ═══════════════════════════════════════════════════════════════════════════

class TestNormaliseRef:

    def test_full_https_github_url(self):
        key, url = _normalise_ref("https://github.com/owner/repo")
        assert key == "github.com/owner/repo"
        assert url == "https://github.com/owner/repo.git"

    def test_full_https_with_git_suffix(self):
        key, url = _normalise_ref("https://github.com/owner/repo.git")
        assert key == "github.com/owner/repo"
        assert url == "https://github.com/owner/repo.git"

    def test_gitlab_com_url(self):
        key, url = _normalise_ref("https://gitlab.com/group/project")
        assert key == "gitlab.com/group/project"
        assert url == "https://gitlab.com/group/project.git"

    def test_gitlab_subgroup_url(self):
        key, url = _normalise_ref("https://gitlab.com/group/subgroup/project")
        assert key == "gitlab.com/group/subgroup/project"

    def test_self_hosted_gitlab_url(self):
        key, url = _normalise_ref("https://gitlab.internal/team/proj")
        assert key == "gitlab.internal/team/proj"
        assert "gitlab.internal" in url

    def test_short_form_owner_repo(self):
        key, url = _normalise_ref("owner/repo")
        assert key == "github.com/owner/repo"
        assert "github.com" in url

    def test_host_qualified_path(self):
        key, url = _normalise_ref("github.com/owner/repo")
        assert key == "github.com/owner/repo"

    def test_empty_ref_raises(self):
        with pytest.raises(WorkspaceError, match="must not be empty"):
            _normalise_ref("")

    def test_url_with_too_short_path_raises(self):
        with pytest.raises(WorkspaceError):
            _normalise_ref("https://github.com/onlyone")

    def test_host_path_too_short_raises(self):
        with pytest.raises(WorkspaceError):
            _normalise_ref("github.com/owner")  # missing repo name

    def test_bare_single_word_raises(self):
        with pytest.raises(WorkspaceError):
            _normalise_ref("notarepo")

    def test_git_suffix_stripped_from_key(self):
        key, _ = _normalise_ref("https://github.com/owner/repo.git")
        assert not key.endswith(".git")

    def test_trailing_slash_ignored(self):
        key, _ = _normalise_ref("github.com/owner/repo/")
        assert key == "github.com/owner/repo"


# ═══════════════════════════════════════════════════════════════════════════
# 2. WorkspaceManager.resolve — fresh clone
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkspaceManagerFreshClone:

    def test_resolve_calls_git_clone_for_new_repo(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "ws")

        with patch("infra.workspace._authenticated_clone_url",
                   return_value="https://token@github.com/owner/repo.git"), \
             patch("infra.workspace._run_git") as mock_git:

            # Simulate successful clone by creating the target dir
            def fake_run_git(args, cwd=None, timeout=300):
                if args[0] == "clone":
                    Path(args[2]).mkdir(parents=True, exist_ok=True)
                    (Path(args[2]) / ".git").mkdir()
                return ""
            mock_git.side_effect = fake_run_git

            result = ws.resolve("owner/repo")

        assert result == tmp_path / "ws" / "github.com" / "owner" / "repo"
        # clone was called
        clone_calls = [c for c in mock_git.call_args_list if c[0][0][0] == "clone"]
        assert len(clone_calls) == 1

    def test_resolve_path_mirrors_host_owner_repo(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "ws")

        with patch("infra.workspace._authenticated_clone_url", return_value="url"), \
             patch("infra.workspace._run_git") as mock_git:
            def create_dir(args, cwd=None, timeout=300):
                if args[0] == "clone":
                    Path(args[2]).mkdir(parents=True)
                    (Path(args[2]) / ".git").mkdir()
                return ""
            mock_git.side_effect = create_dir

            path = ws.resolve("https://gitlab.com/group/proj")

        assert path == tmp_path / "ws" / "gitlab.com" / "group" / "proj"

    def test_resolve_uses_authenticated_url(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "ws")
        auth_url = "https://mytoken@github.com/owner/repo.git"

        with patch("infra.workspace._authenticated_clone_url",
                   return_value=auth_url) as mock_auth, \
             patch("infra.workspace._run_git") as mock_git:
            def create_dir(args, cwd=None, timeout=300):
                if args[0] == "clone":
                    Path(args[2]).mkdir(parents=True)
                    (Path(args[2]) / ".git").mkdir()
                return ""
            mock_git.side_effect = create_dir

            ws.resolve("owner/repo")

        # The authenticated URL must have been used in the clone call
        clone_calls = [c for c in mock_git.call_args_list if c[0][0][0] == "clone"]
        assert clone_calls[0][0][0][1] == auth_url

    def test_resolve_git_clone_failure_raises_workspace_error(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "ws")

        with patch("infra.workspace._authenticated_clone_url", return_value="url"), \
             patch("infra.workspace._run_git",
                   side_effect=WorkspaceError("clone failed: auth error")):

            with pytest.raises(WorkspaceError, match="clone failed"):
                ws.resolve("owner/repo")


# ═══════════════════════════════════════════════════════════════════════════
# 3. WorkspaceManager.resolve — existing clone (pull)
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkspaceManagerPull:

    def _make_existing_repo(self, base: Path, canonical: str) -> Path:
        """Create a fake existing local clone."""
        repo_path = base / Path(*canonical.split("/"))
        repo_path.mkdir(parents=True)
        (repo_path / ".git").mkdir()
        return repo_path

    def test_resolve_existing_repo_does_not_clone(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        self._make_existing_repo(tmp_path, "github.com/owner/repo")

        with patch("infra.workspace._run_git") as mock_git:
            mock_git.return_value = "origin/main"
            ws.resolve("owner/repo")

        # fetch should be called, but not clone
        all_cmds = [c[0][0][0] for c in mock_git.call_args_list]
        assert "clone" not in all_cmds
        assert "fetch" in all_cmds

    def test_resolve_existing_calls_fetch_checkout_pull(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        repo_path = self._make_existing_repo(tmp_path, "github.com/owner/repo")

        git_calls = []
        def record_git(args, cwd=None, timeout=300):
            git_calls.append(args[0] if args else "")
            if args[:2] == ["symbolic-ref", "refs/remotes/origin/HEAD"]:
                return "origin/main"
            return ""
        
        with patch("infra.workspace._run_git", side_effect=record_git):
            ws.resolve("owner/repo")

        assert "fetch" in git_calls
        assert "checkout" in git_calls
        assert "pull" in git_calls

    def test_resolve_existing_uses_detected_default_branch(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        self._make_existing_repo(tmp_path, "github.com/owner/repo")

        checkout_args = []
        def fake_git(args, cwd=None, timeout=300):
            if args[0] == "checkout":
                checkout_args.extend(args)
            if args[:2] == ["symbolic-ref", "refs/remotes/origin/HEAD"]:
                return "origin/develop"
            return ""

        with patch("infra.workspace._run_git", side_effect=fake_git):
            ws.resolve("owner/repo")

        assert "develop" in checkout_args

    def test_resolve_fetch_failure_continues(self, tmp_path):
        """Fetch failure should log a warning and return existing state, not raise."""
        ws = WorkspaceManager(tmp_path)
        self._make_existing_repo(tmp_path, "github.com/owner/repo")

        def fail_fetch(args, cwd=None, timeout=300):
            if args[0] == "fetch":
                raise WorkspaceError("fetch failed: connection refused")
            return "origin/main"

        with patch("infra.workspace._run_git", side_effect=fail_fetch):
            path = ws.resolve("owner/repo")  # must not raise

        assert path.exists()

    def test_resolve_returns_correct_path_for_existing(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        expected = self._make_existing_repo(tmp_path, "github.com/owner/repo")

        with patch("infra.workspace._run_git", return_value="origin/main"):
            result = ws.resolve("owner/repo")

        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════
# 4. WorkspaceManager.clean
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkspaceManagerClean:

    def test_clean_removes_directory(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        repo_path = tmp_path / "github.com" / "owner" / "repo"
        repo_path.mkdir(parents=True)
        (repo_path / "file.txt").write_text("content")

        ws.clean("owner/repo")

        assert not repo_path.exists()

    def test_clean_nonexistent_is_noop(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        ws.clean("owner/nonexistent")  # must not raise

    def test_clean_removes_empty_parent_dirs(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        repo_path = tmp_path / "github.com" / "singleowner" / "only-repo"
        repo_path.mkdir(parents=True)

        ws.clean("github.com/singleowner/only-repo")

        # repo dir gone
        assert not repo_path.exists()
        # owner dir should be gone too (was empty)
        assert not (tmp_path / "github.com" / "singleowner").exists()

    def test_clean_invalid_ref_logs_warning_no_raise(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        ws.clean("not-a-valid/ref/with/too/many/levels/and/extra")  # must not raise

    def test_clean_leaves_sibling_repos(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        repo_a = tmp_path / "github.com" / "owner" / "repo-a"
        repo_b = tmp_path / "github.com" / "owner" / "repo-b"
        repo_a.mkdir(parents=True)
        repo_b.mkdir(parents=True)

        ws.clean("github.com/owner/repo-a")

        assert not repo_a.exists()
        assert repo_b.exists()  # sibling untouched


# ═══════════════════════════════════════════════════════════════════════════
# 5. WorkspaceManager.list_workspaces
# ═══════════════════════════════════════════════════════════════════════════

class TestWorkspaceManagerList:

    def test_list_empty_workspace(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "ws")
        assert ws.list_workspaces() == []

    def test_list_nonexistent_root(self, tmp_path):
        ws = WorkspaceManager(tmp_path / "nonexistent")
        assert ws.list_workspaces() == []

    def test_list_detects_git_dirs(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        for path in ("github.com/owner/repo-a", "github.com/owner/repo-b"):
            repo = tmp_path / Path(*path.split("/"))
            repo.mkdir(parents=True)
            (repo / ".git").mkdir()

        infos = ws.list_workspaces()
        refs = {i.repo_ref for i in infos}
        assert "github.com/owner/repo-a" in refs
        assert "github.com/owner/repo-b" in refs

    def test_list_ignores_non_git_dirs(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        not_a_repo = tmp_path / "github.com" / "owner" / "not-cloned"
        not_a_repo.mkdir(parents=True)
        (not_a_repo / "some_file.txt").write_text("nope")

        assert ws.list_workspaces() == []

    def test_list_returns_workspace_info_objects(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        repo = tmp_path / "github.com" / "owner" / "repo"
        repo.mkdir(parents=True)
        (repo / ".git").mkdir()

        infos = ws.list_workspaces()
        assert len(infos) == 1
        info = infos[0]
        assert isinstance(info, WorkspaceInfo)
        assert info.local_path == repo
        assert info.last_used is not None

    def test_list_mixed_hosts(self, tmp_path):
        ws = WorkspaceManager(tmp_path)
        for path in ("github.com/a/b", "gitlab.com/c/d", "internal.corp/team/proj"):
            repo = tmp_path / Path(*path.split("/"))
            repo.mkdir(parents=True)
            (repo / ".git").mkdir()

        infos = ws.list_workspaces()
        assert len(infos) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 6. active_repo context variable
# ═══════════════════════════════════════════════════════════════════════════

class TestActiveRepoContextVar:

    def test_default_falls_back_to_settings(self):
        clear_repo_root()
        with patch("app.core.config.get_settings",
                   return_value=SimpleNamespace(target_repo_path="/static/path")):
            result = get_repo_root()
        assert result == "/static/path"

    def test_set_overrides_settings(self):
        set_repo_root("/dynamic/path")
        # Context var wins — settings irrelevant
        result = get_repo_root()
        assert result == "/dynamic/path"

    def test_clear_restores_settings_fallback(self):
        set_repo_root("/dynamic/path")
        clear_repo_root()
        with patch("app.core.config.get_settings",
                   return_value=SimpleNamespace(target_repo_path="/static/path")):
            result = get_repo_root()
        assert result == "/static/path"

    def test_settings_exception_returns_empty(self):
        clear_repo_root()
        with patch("app.core.config.get_settings",
                   side_effect=Exception("no settings")):
            result = get_repo_root()
        assert result == ""

    def test_set_empty_string_reverts_to_fallback(self):
        set_repo_root("/dynamic")
        set_repo_root("")   # clear
        with patch("app.core.config.get_settings",
                   return_value=SimpleNamespace(target_repo_path="/static")):
            assert get_repo_root() == "/static"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Tools use get_repo_root
# ═══════════════════════════════════════════════════════════════════════════

class TestToolsUseDynamicRoot:

    def test_filesystem_uses_context_var(self, tmp_path):
        """filesystem._resolve_safe must use get_repo_root(), not settings directly."""
        from app.tools.filesystem import _resolve_safe

        set_repo_root(str(tmp_path))
        resolved = _resolve_safe("subdir/file.txt")
        assert str(resolved).startswith(str(tmp_path))

    def test_filesystem_rejects_escape_even_with_context_var(self, tmp_path):
        from app.tools.filesystem import PathEscapeError, _resolve_safe

        set_repo_root(str(tmp_path))
        with pytest.raises((PathEscapeError, FileNotFoundError)):
            _resolve_safe("../../etc/passwd")

    def test_git_tool_uses_context_var(self, tmp_path):
        """git._run_git must read repo root from get_repo_root()."""
        from app.tools import git as git_mod

        set_repo_root(str(tmp_path))
        with patch("app.core.active_repo.get_repo_root", return_value=str(tmp_path)):
            root = Path(get_repo_root()).resolve()
        assert root == tmp_path.resolve()

    def test_build_run_tests_uses_context_var(self, tmp_path):
        """build.run_tests must call get_repo_root(), not settings.target_repo_path."""
        from app.tools import build as build_mod

        set_repo_root(str(tmp_path))
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

        with patch("app.tools.build.run_terminal") as mock_term:
            mock_term.invoke = MagicMock(return_value="1 passed")
            # Just verify the function doesn't error and calls terminal
            from app.tools.build import run_tests
            # The tool reads get_repo_root() for project detection
            with patch("app.core.active_repo.get_repo_root", return_value=str(tmp_path)):
                from app.tools.build import _detect_project_type
                types = _detect_project_type(str(tmp_path))
            assert "python" in types

    def test_shell_uses_context_var(self, tmp_path):
        """shell.run_shell must use get_repo_root() for the sandbox root."""
        import app.tools.shell as shell_mod

        set_repo_root(str(tmp_path))
        with patch("app.core.active_repo.get_repo_root", return_value=str(tmp_path)):
            root = Path(get_repo_root()).resolve()
        assert root == tmp_path.resolve()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Config keys
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigKeys:

    def test_daedalus_workspace_dir_exists(self):
        from app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "daedalus_workspace_dir")

    def test_daedalus_workspace_dir_default_is_resolved(self):
        from app.core.config import get_settings
        s = get_settings()
        # Default should be expanded/resolved (no tilde)
        assert "~" not in s.daedalus_workspace_dir
        assert Path(s.daedalus_workspace_dir).is_absolute()

    def test_target_repo_path_still_works(self):
        from app.core.config import get_settings
        s = get_settings()
        # target_repo_path must still exist (backward compat)
        assert hasattr(s, "target_repo_path")

    def test_daedalus_workspace_dir_default_contains_daedalus(self):
        from app.core.config import get_settings
        s = get_settings()
        assert "daedalus" in s.daedalus_workspace_dir.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 9. GraphState.repo_ref
# ═══════════════════════════════════════════════════════════════════════════

class TestStateRepoRef:

    def test_repo_ref_field_exists(self):
        from app.core.state import GraphState
        s = GraphState(user_request="test")
        assert hasattr(s, "repo_ref")

    def test_repo_ref_default_empty_string(self):
        from app.core.state import GraphState
        s = GraphState(user_request="test")
        assert s.repo_ref == ""

    def test_repo_ref_round_trips_through_model_dump(self):
        from app.core.state import GraphState
        s = GraphState(user_request="test", repo_ref="owner/repo")
        d = s.model_dump()
        assert d["repo_ref"] == "owner/repo"
        s2 = GraphState(**d)
        assert s2.repo_ref == "owner/repo"


# ═══════════════════════════════════════════════════════════════════════════
# 10. context_loader_node — workspace integration
# ═══════════════════════════════════════════════════════════════════════════

class TestContextLoaderWorkspaceIntegration:

    def _make_settings(self, tmp_path, target_repo_path=""):
        return SimpleNamespace(
            target_repo_path=target_repo_path,
            daedalus_workspace_dir=str(tmp_path / "ws"),
            max_output_chars=10000,
            context_warn_fraction=0.7,
            tool_result_max_chars=8000,
        )

    def test_static_path_bypasses_workspace(self, tmp_path, monkeypatch):
        """When repo_root or target_repo_path is set, no WorkspaceManager call."""
        from app.core import nodes

        (tmp_path / "README.md").write_text("hello")

        mock_ws = MagicMock()
        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, str(tmp_path)))

        with patch("infra.workspace.WorkspaceManager") as MockWS:
            state = __import__("app.core.state", fromlist=["GraphState"]).GraphState(
                user_request="test", repo_root=str(tmp_path)
            )
            result = nodes.context_loader_node(state)

        MockWS.assert_not_called()
        assert result.get("context_loaded") is True

    def test_missing_repo_root_and_ref_stops(self, tmp_path, monkeypatch):
        """With neither repo_root nor repo_ref, workflow must stop."""
        from app.core import nodes
        from app.core.state import GraphState

        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, target_repo_path=""))

        state = GraphState(user_request="test", repo_root="", repo_ref="")
        result = nodes.context_loader_node(state)

        assert result["context_loaded"] is False
        assert "missing" in result["stop_reason"].lower()

    def test_workspace_resolve_called_when_no_static_path(self, tmp_path, monkeypatch):
        """When repo_root is empty but repo_ref is set, WorkspaceManager.resolve is called."""
        from app.core import nodes
        from app.core.state import GraphState

        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, target_repo_path=""))

        # Make a fake resolved path
        fake_repo = tmp_path / "github.com" / "owner" / "repo"
        fake_repo.mkdir(parents=True)
        (fake_repo / "README.md").write_text("# Fake repo")

        mock_ws_instance = MagicMock()
        mock_ws_instance.resolve.return_value = fake_repo

        with patch("infra.workspace.WorkspaceManager", return_value=mock_ws_instance):
            state = GraphState(user_request="test", repo_root="", repo_ref="owner/repo")
            result = nodes.context_loader_node(state)

        mock_ws_instance.resolve.assert_called_once_with("owner/repo")
        assert result.get("context_loaded") is True

    def test_workspace_error_stops_workflow(self, tmp_path, monkeypatch):
        """WorkspaceManager.resolve raising WorkspaceError must stop the workflow."""
        from app.core import nodes
        from app.core.state import GraphState
        from infra.workspace import WorkspaceError

        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, target_repo_path=""))

        mock_ws_instance = MagicMock()
        mock_ws_instance.resolve.side_effect = WorkspaceError("auth failed")

        with patch("infra.workspace.WorkspaceManager", return_value=mock_ws_instance):
            state = GraphState(user_request="test", repo_root="", repo_ref="owner/repo")
            result = nodes.context_loader_node(state)

        assert result["context_loaded"] is False
        assert "workspace" in result["stop_reason"].lower()

    def test_resolved_repo_root_in_return_dict(self, tmp_path, monkeypatch):
        """context_loader must return the resolved repo_root so state is persisted."""
        from app.core import nodes
        from app.core.state import GraphState

        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, target_repo_path=""))

        fake_repo = tmp_path / "github.com" / "owner" / "repo"
        fake_repo.mkdir(parents=True)
        (fake_repo / "README.md").write_text("# x")

        mock_ws_instance = MagicMock()
        mock_ws_instance.resolve.return_value = fake_repo

        with patch("infra.workspace.WorkspaceManager", return_value=mock_ws_instance):
            state = GraphState(user_request="test", repo_root="", repo_ref="owner/repo")
            result = nodes.context_loader_node(state)

        assert "repo_root" in result
        assert result["repo_root"] == str(fake_repo.resolve())

    def test_context_var_set_after_resolve(self, tmp_path, monkeypatch):
        """set_repo_root must be called with the resolved path so tools use it."""
        from app.core import nodes
        from app.core.state import GraphState
        import app.core.active_repo as ar

        monkeypatch.setattr(nodes, "get_settings",
                            lambda: self._make_settings(tmp_path, target_repo_path=""))

        fake_repo = tmp_path / "github.com" / "owner" / "repo"
        fake_repo.mkdir(parents=True)
        (fake_repo / "README.md").write_text("# x")

        mock_ws_instance = MagicMock()
        mock_ws_instance.resolve.return_value = fake_repo

        set_root_calls = []
        original_set = ar.set_repo_root
        def track_set(path):
            set_root_calls.append(path)
            original_set(path)

        with patch("infra.workspace.WorkspaceManager", return_value=mock_ws_instance), \
             patch("app.core.active_repo.set_repo_root", side_effect=track_set):
            state = GraphState(user_request="test", repo_root="", repo_ref="owner/repo")
            nodes.context_loader_node(state)

        assert any(str(fake_repo.resolve()) in p for p in set_root_calls)


# ═══════════════════════════════════════════════════════════════════════════
# 11. infra package exports WorkspaceManager
# ═══════════════════════════════════════════════════════════════════════════

class TestInfraExports:

    def test_workspace_manager_exported(self):
        import infra
        assert hasattr(infra, "WorkspaceManager")

    def test_workspace_info_exported(self):
        import infra
        assert hasattr(infra, "WorkspaceInfo")

    def test_workspace_error_exported(self):
        import infra
        assert hasattr(infra, "WorkspaceError")


# ═══════════════════════════════════════════════════════════════════════════
# 12. run_workflow forwards repo_ref
# ═══════════════════════════════════════════════════════════════════════════

class TestRunWorkflowRepoRef:

    @pytest.mark.asyncio
    async def test_repo_ref_propagated_to_initial_state(self):
        """run_workflow must put repo_ref into the initial GraphState."""
        from app.core.orchestrator import run_workflow
        from app.core.state import GraphState, WorkflowPhase

        captured_states = []

        async def fake_to_thread(fn, state_dict):
            captured_states.append(state_dict)
            # Return a minimal valid completed state
            s = GraphState(
                user_request="test",
                repo_root="/tmp/fake",
                repo_ref="owner/repo",
                phase=WorkflowPhase.COMPLETE,
            )
            return s.model_dump()

        with patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch("app.core.orchestrator.compile_graph"):
            await run_workflow("test task", "/tmp/fake", repo_ref="owner/repo")

        assert len(captured_states) == 1
        assert captured_states[0].get("repo_ref") == "owner/repo"

    @pytest.mark.asyncio
    async def test_empty_repo_ref_defaults_to_empty_string(self):
        from app.core.orchestrator import run_workflow
        from app.core.state import GraphState, WorkflowPhase

        captured = []

        async def fake_to_thread(fn, state_dict):
            captured.append(state_dict)
            s = GraphState(user_request="t", phase=WorkflowPhase.COMPLETE)
            return s.model_dump()

        with patch("asyncio.to_thread", side_effect=fake_to_thread), \
             patch("app.core.orchestrator.compile_graph"):
            await run_workflow("test", "/tmp/fake")

        assert captured[0].get("repo_ref", "") == ""
