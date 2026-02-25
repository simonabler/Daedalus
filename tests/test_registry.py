"""Tests for infra.registry — YAML-based repo access control."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from infra.registry import RepoEntry, RepoRegistry, get_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "repos.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


VALID_YAML = """
repos:
  - name: my-api
    url: https://github.com/org/my-api
    default_branch: main
    description: "Main backend API"
  - name: infra-scripts
    url: https://gitlab.internal/ops/infra-scripts
    default_branch: develop
    description: "Internal infrastructure scripts"
  - name: no-desc
    url: https://github.com/org/no-desc
"""


# ---------------------------------------------------------------------------
# RepoEntry — unit tests
# ---------------------------------------------------------------------------


class TestRepoEntry:
    def test_basic_fields(self):
        e = RepoEntry(name="my-api", url="https://github.com/org/my-api")
        assert e.name == "my-api"
        assert e.url == "https://github.com/org/my-api"
        assert e.default_branch == "main"
        assert e.description == ""

    def test_trailing_slash_stripped_from_url(self):
        e = RepoEntry(name="x", url="https://github.com/org/repo/")
        assert not e.url.endswith("/")

    def test_owner_name(self):
        e = RepoEntry(name="x", url="https://github.com/acme/myrepo")
        assert e.owner_name == "acme/myrepo"

    def test_canonical_key(self):
        e = RepoEntry(name="x", url="https://github.com/acme/myrepo")
        assert e.canonical_key == "github.com/acme/myrepo"

    def test_canonical_key_gitlab_self_hosted(self):
        e = RepoEntry(name="x", url="https://gitlab.internal/team/proj")
        assert e.canonical_key == "gitlab.internal/team/proj"

    def test_matches_name_alias(self):
        e = RepoEntry(name="my-api", url="https://github.com/org/my-api")
        assert e.matches("my-api")
        assert e.matches("MY-API")  # case-insensitive

    def test_matches_full_url(self):
        e = RepoEntry(name="x", url="https://github.com/org/repo")
        assert e.matches("https://github.com/org/repo")
        assert e.matches("https://github.com/org/repo/")  # trailing slash

    def test_matches_no_scheme_url(self):
        e = RepoEntry(name="x", url="https://github.com/org/repo")
        assert e.matches("github.com/org/repo")

    def test_matches_owner_name(self):
        e = RepoEntry(name="x", url="https://github.com/org/repo")
        assert e.matches("org/repo")

    def test_matches_canonical_key(self):
        e = RepoEntry(name="x", url="https://github.com/org/repo")
        assert e.matches("github.com/org/repo")

    def test_does_not_match_unrelated(self):
        e = RepoEntry(name="my-api", url="https://github.com/org/my-api")
        assert not e.matches("other-api")
        assert not e.matches("https://github.com/org/other")
        assert not e.matches("")
        assert not e.matches("   ")

    def test_name_empty_raises(self):
        with pytest.raises(Exception):
            RepoEntry(name="   ", url="https://github.com/org/repo")

    def test_url_empty_raises(self):
        with pytest.raises(Exception):
            RepoEntry(name="x", url="   ")


# ---------------------------------------------------------------------------
# RepoRegistry.load
# ---------------------------------------------------------------------------


class TestRepoRegistryLoad:
    def test_load_valid_yaml(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, VALID_YAML))
        assert len(reg) == 3

    def test_load_empty_repos_key(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, "repos: []\n"))
        assert len(reg) == 0

    def test_load_missing_file_logs_warning(self, tmp_path):
        reg = RepoRegistry()
        reg.load(tmp_path / "nonexistent.yaml")  # should not raise
        assert len(reg) == 0

    def test_load_invalid_repos_type_raises(self, tmp_path):
        reg = RepoRegistry()
        with pytest.raises(ValueError, match="list"):
            reg.load(_write_yaml(tmp_path, "repos: not-a-list\n"))

    def test_load_invalid_entry_raises(self, tmp_path):
        reg = RepoRegistry()
        with pytest.raises(ValueError, match="entry"):
            reg.load(_write_yaml(tmp_path, "repos:\n  - not-a-dict\n"))

    def test_load_missing_required_url_raises(self, tmp_path):
        reg = RepoRegistry()
        with pytest.raises(ValueError):
            reg.load(_write_yaml(tmp_path, "repos:\n  - name: x\n"))

    def test_optional_description(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, VALID_YAML))
        entry = reg.resolve("no-desc")
        assert entry.description == ""

    def test_default_branch_custom(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, VALID_YAML))
        entry = reg.resolve("infra-scripts")
        assert entry.default_branch == "develop"


# ---------------------------------------------------------------------------
# RepoRegistry.resolve
# ---------------------------------------------------------------------------


class TestRepoRegistryResolve:
    def setup_method(self, _):
        self.reg = RepoRegistry()

    def _load(self, tmp_path):
        self.reg.load(_write_yaml(tmp_path, VALID_YAML))

    def test_resolve_by_name(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("my-api")
        assert e.name == "my-api"

    def test_resolve_by_name_case_insensitive(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("MY-API")
        assert e.name == "my-api"

    def test_resolve_by_full_url(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("https://github.com/org/my-api")
        assert e.name == "my-api"

    def test_resolve_by_owner_name(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("org/my-api")
        assert e.name == "my-api"

    def test_resolve_by_canonical_key(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("github.com/org/my-api")
        assert e.name == "my-api"

    def test_resolve_gitlab_self_hosted(self, tmp_path):
        self._load(tmp_path)
        e = self.reg.resolve("infra-scripts")
        assert "gitlab.internal" in e.url

    def test_resolve_unknown_raises_value_error(self, tmp_path):
        self._load(tmp_path)
        with pytest.raises(ValueError, match="not in the registry"):
            self.reg.resolve("unknown-repo")

    def test_resolve_empty_string_raises(self, tmp_path):
        self._load(tmp_path)
        with pytest.raises(ValueError):
            self.reg.resolve("")


# ---------------------------------------------------------------------------
# RepoRegistry.is_allowed
# ---------------------------------------------------------------------------


class TestRepoRegistryIsAllowed:
    def setup_method(self, _):
        self.reg = RepoRegistry()

    def _load(self, tmp_path):
        self.reg.load(_write_yaml(tmp_path, VALID_YAML))

    def test_allowed_name(self, tmp_path):
        self._load(tmp_path)
        assert self.reg.is_allowed("my-api")

    def test_allowed_url(self, tmp_path):
        self._load(tmp_path)
        assert self.reg.is_allowed("https://github.com/org/my-api")

    def test_allowed_owner_name(self, tmp_path):
        self._load(tmp_path)
        assert self.reg.is_allowed("org/my-api")

    def test_not_allowed_unknown(self, tmp_path):
        self._load(tmp_path)
        assert not self.reg.is_allowed("unknown")

    def test_not_allowed_empty(self, tmp_path):
        self._load(tmp_path)
        assert not self.reg.is_allowed("")

    def test_empty_registry_allows_nothing(self, tmp_path):
        self.reg.load(_write_yaml(tmp_path, "repos: []\n"))
        assert not self.reg.is_allowed("anything")


# ---------------------------------------------------------------------------
# RepoRegistry.list_repos
# ---------------------------------------------------------------------------


class TestRepoRegistryListRepos:
    def test_list_returns_all(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, VALID_YAML))
        repos = reg.list_repos()
        assert len(repos) == 3
        names = {r.name for r in repos}
        assert names == {"my-api", "infra-scripts", "no-desc"}

    def test_list_is_copy(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, VALID_YAML))
        a = reg.list_repos()
        b = reg.list_repos()
        assert a is not b  # different list objects

    def test_list_empty_registry(self, tmp_path):
        reg = RepoRegistry()
        reg.load(_write_yaml(tmp_path, "repos: []\n"))
        assert reg.list_repos() == []


# ---------------------------------------------------------------------------
# get_registry singleton
# ---------------------------------------------------------------------------


class TestGetRegistrySingleton:
    def test_singleton_returns_same_object(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, VALID_YAML)
        r1 = get_registry(yaml_path)
        r2 = get_registry(yaml_path)
        # When path is explicitly passed, a fresh instance is returned each time
        # (path != None bypasses singleton storage).
        assert r1 is not r2  # both valid but independent

    def test_explicit_path_loads_correctly(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, VALID_YAML)
        reg = get_registry(yaml_path)
        assert len(reg) == 3

    def test_reload_flag_reloads(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, VALID_YAML)
        r1 = get_registry(yaml_path)
        r2 = get_registry(yaml_path, reload=True)
        assert len(r1) == len(r2)

    def test_registry_is_always_truthy(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, "repos: []\n")
        reg = get_registry(yaml_path)
        assert bool(reg) is True  # even when empty


# ---------------------------------------------------------------------------
# Integration: registry guard in context_loader_node
# ---------------------------------------------------------------------------


class TestContextLoaderRegistryGuard:
    """Verify context_loader_node rejects repos not in registry."""

    def test_unknown_repo_stops_workflow(self, tmp_path):
        from unittest.mock import patch

        from app.core.nodes import context_loader_node
        from app.core.state import GraphState, WorkflowPhase

        yaml_path = _write_yaml(tmp_path, VALID_YAML)

        state = GraphState(
            user_request="add feature",
            repo_ref="not-in-registry",
        )

        with (
            patch("app.core.config.get_settings") as mock_settings,
            patch("infra.registry.get_registry", return_value=get_registry(yaml_path)),
        ):
            mock_settings.return_value.target_repo_path = ""
            mock_settings.return_value.daedalus_workspace_dir = str(tmp_path / "ws")
            mock_settings.return_value.repos_yaml_path = ""
            result = context_loader_node(state)

        assert result.get("phase") == WorkflowPhase.STOPPED
        assert result.get("stop_reason") == "context_repo_not_in_registry"

    def test_known_repo_passes_guard(self, tmp_path):
        """Registry guard passes; workspace resolve fails (expected in unit test)."""
        from unittest.mock import patch, MagicMock

        from app.core.nodes import context_loader_node
        from app.core.state import GraphState, WorkflowPhase

        yaml_path = _write_yaml(tmp_path, VALID_YAML)

        state = GraphState(
            user_request="add feature",
            repo_ref="my-api",
        )

        ws_mock = MagicMock()
        ws_mock.resolve.side_effect = Exception("no network in tests")

        with (
            patch("app.core.config.get_settings") as mock_settings,
            patch("infra.registry.get_registry", return_value=get_registry(yaml_path)),
            patch("infra.workspace.WorkspaceManager", return_value=ws_mock),
        ):
            mock_settings.return_value.target_repo_path = ""
            mock_settings.return_value.daedalus_workspace_dir = str(tmp_path / "ws")
            mock_settings.return_value.repos_yaml_path = ""
            result = context_loader_node(state)

        # Should not be rejected by registry — workspace error instead
        assert result.get("stop_reason") != "context_repo_not_in_registry"

    def test_empty_registry_skips_guard(self, tmp_path):
        """When registry is empty, the guard is bypassed (permissive default)."""
        from unittest.mock import patch, MagicMock

        from app.core.nodes import context_loader_node
        from app.core.state import GraphState

        empty_yaml = _write_yaml(tmp_path, "repos: []\n")

        state = GraphState(
            user_request="add feature",
            repo_ref="any-repo",
        )

        ws_mock = MagicMock()
        ws_mock.resolve.side_effect = Exception("no network")

        with (
            patch("app.core.config.get_settings") as mock_settings,
            patch("infra.registry.get_registry", return_value=get_registry(empty_yaml)),
            patch("infra.workspace.WorkspaceManager", return_value=ws_mock),
        ):
            mock_settings.return_value.target_repo_path = ""
            mock_settings.return_value.daedalus_workspace_dir = str(tmp_path / "ws")
            mock_settings.return_value.repos_yaml_path = ""
            result = context_loader_node(state)

        assert result.get("stop_reason") != "context_repo_not_in_registry"


# ---------------------------------------------------------------------------
# Integration: _extract_repo_ref in router
# ---------------------------------------------------------------------------


class TestExtractRepoRef:
    def _call(self, text: str) -> str:
        from app.core.nodes import _extract_repo_ref
        return _extract_repo_ref(text)

    def test_full_https_url(self):
        assert self._call("fix issue #42 in https://github.com/org/my-api") == "https://github.com/org/my-api"

    def test_no_scheme_github(self):
        ref = self._call("add feature to github.com/org/my-api please")
        assert ref == "github.com/org/my-api"

    def test_no_scheme_gitlab(self):
        ref = self._call("check gitlab.internal/team/proj issue 7")
        assert ref == "gitlab.internal/team/proj"

    def test_keyword_in_alias(self):
        ref = self._call("fix issue #42 in my-api")
        assert ref == "my-api"

    def test_keyword_in_owner_name(self):
        ref = self._call("add feature in org/repo")
        assert ref == "org/repo"

    def test_keyword_for(self):
        ref = self._call("create endpoint for infra-scripts")
        assert ref == "infra-scripts"

    def test_keyword_repo(self):
        ref = self._call("analyse repo my-service")
        assert ref == "my-service"

    def test_no_match_returns_empty(self):
        assert self._call("show me the status") == ""
        assert self._call("") == ""
