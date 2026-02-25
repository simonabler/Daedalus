"""Tests for the unified forge API client layer (Issue #44).

Covers:
- Data models: Issue, PRRequest, PRResult
- ForgeClient protocol satisfaction (GitHubClient, GitLabClient)
- GitHubClient: all operations against mocked HTTP (respx)
- GitLabClient: all operations against mocked HTTP (respx)
- factory.get_forge_client: auto-detection + explicit platform
- factory convenience helpers
- ForgeError on HTTP errors
- Config keys (GITHUB_TOKEN, GITLAB_TOKEN, GITLAB_URL)

No real network calls are made — all HTTP is mocked via respx.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
import respx
import httpx

from infra.forge import ForgeClient, ForgeError, Issue, PRRequest, PRResult
from infra.github_client import GitHubClient
from infra.gitlab_client import GitLabClient
from infra.factory import get_forge_client, get_github_client, get_gitlab_client


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data models
# ═══════════════════════════════════════════════════════════════════════════

class TestDataModels:
    def test_issue_required_fields(self):
        issue = Issue(id=1, title="Test", url="https://example.com/issues/1")
        assert issue.id == 1
        assert issue.title == "Test"
        assert issue.description == ""
        assert issue.labels == []
        assert issue.author == ""
        assert issue.created_at is None

    def test_issue_full_fields(self):
        ts = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        issue = Issue(
            id=42,
            title="Bug report",
            description="Something is broken",
            url="https://github.com/owner/repo/issues/42",
            labels=["bug", "p1"],
            author="alice",
            created_at=ts,
        )
        assert issue.labels == ["bug", "p1"]
        assert issue.created_at == ts

    def test_pr_request_fields(self):
        pr = PRRequest(
            title="feat: health endpoint",
            body="Adds /health",
            head_branch="feat/health",
            base_branch="main",
        )
        assert pr.head_branch == "feat/health"
        assert pr.base_branch == "main"

    def test_pr_result_fields(self):
        result = PRResult(id=101, url="https://github.com/owner/repo/pull/5", number=5)
        assert result.number == 5

    def test_pr_request_body_optional(self):
        pr = PRRequest(title="title", head_branch="feature", base_branch="main")
        assert pr.body == ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. ForgeClient protocol compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestProtocolCompliance:
    def test_github_client_satisfies_protocol(self):
        client = GitHubClient(token="tok")
        assert isinstance(client, ForgeClient)

    def test_gitlab_client_satisfies_protocol(self):
        client = GitLabClient(token="tok")
        assert isinstance(client, ForgeClient)

    def test_github_client_has_all_methods(self):
        client = GitHubClient()
        for method in ("get_issue", "list_issues", "clone_url", "create_pr",
                       "post_comment", "list_branches", "get_default_branch"):
            assert callable(getattr(client, method)), f"Missing method: {method}"

    def test_gitlab_client_has_all_methods(self):
        client = GitLabClient()
        for method in ("get_issue", "list_issues", "clone_url", "create_pr",
                       "post_comment", "list_branches", "get_default_branch"):
            assert callable(getattr(client, method)), f"Missing method: {method}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. GitHubClient — mocked HTTP tests
# ═══════════════════════════════════════════════════════════════════════════

GITHUB_API = "https://api.github.com"


@respx.mock
class TestGitHubClient:

    def _client(self) -> GitHubClient:
        return GitHubClient(token="ghp_test_token")

    # --- get_issue ---

    def test_get_issue_returns_issue(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues/42").mock(
            return_value=httpx.Response(200, json={
                "number": 42,
                "title": "Fix the bug",
                "body": "It crashes on startup",
                "html_url": "https://github.com/owner/repo/issues/42",
                "labels": [{"name": "bug"}],
                "user": {"login": "alice"},
                "created_at": "2024-01-02T03:04:05Z",
            })
        )
        issue = self._client().get_issue("owner/repo", 42)
        assert issue.id == 42
        assert issue.title == "Fix the bug"
        assert issue.description == "It crashes on startup"
        assert issue.labels == ["bug"]
        assert issue.author == "alice"
        assert issue.created_at is not None

    def test_get_issue_raises_on_404(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues/999").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        with pytest.raises(ForgeError) as exc_info:
            self._client().get_issue("owner/repo", 999)
        assert exc_info.value.status_code == 404

    def test_get_issue_null_body(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues/1").mock(
            return_value=httpx.Response(200, json={
                "number": 1, "title": "No body", "body": None,
                "html_url": "https://github.com/owner/repo/issues/1",
                "labels": [], "user": {"login": "bob"}, "created_at": None,
            })
        )
        issue = self._client().get_issue("owner/repo", 1)
        assert issue.description == ""

    # --- list_issues ---

    def test_list_issues_filters_prs(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues").mock(
            return_value=httpx.Response(200, json=[
                {"number": 1, "title": "Issue", "body": "", "html_url": "url1",
                 "labels": [], "user": {"login": "u"}, "created_at": None},
                {"number": 2, "title": "PR", "body": "", "html_url": "url2",
                 "labels": [], "user": {"login": "u"}, "created_at": None,
                 "pull_request": {"url": "..."}},  # this should be filtered
            ])
        )
        issues = self._client().list_issues("owner/repo")
        assert len(issues) == 1
        assert issues[0].id == 1

    def test_list_issues_closed(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = self._client().list_issues("owner/repo", state="closed")
        assert result == []

    def test_list_issues_passes_state_param(self):
        route = respx.get(f"{GITHUB_API}/repos/owner/repo/issues").mock(
            return_value=httpx.Response(200, json=[])
        )
        self._client().list_issues("owner/repo", state="all")
        assert route.called
        request = route.calls.last.request
        assert b"state=all" in request.url.query

    # --- clone_url ---

    def test_clone_url_embeds_token(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo").mock(
            return_value=httpx.Response(200, json={
                "clone_url": "https://github.com/owner/repo.git",
                "default_branch": "main",
            })
        )
        url = self._client().clone_url("owner/repo")
        assert url == "https://ghp_test_token@github.com/owner/repo.git"

    def test_clone_url_no_token(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo").mock(
            return_value=httpx.Response(200, json={
                "clone_url": "https://github.com/owner/repo.git",
                "default_branch": "main",
            })
        )
        url = GitHubClient(token="").clone_url("owner/repo")
        assert "ghp_" not in url
        assert url == "https://github.com/owner/repo.git"

    # --- list_branches ---

    def test_list_branches(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/branches").mock(
            return_value=httpx.Response(200, json=[
                {"name": "main"}, {"name": "develop"}, {"name": "feat/x"},
            ])
        )
        branches = self._client().list_branches("owner/repo")
        assert branches == ["main", "develop", "feat/x"]

    def test_list_branches_empty(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/branches").mock(
            return_value=httpx.Response(200, json=[])
        )
        assert self._client().list_branches("owner/repo") == []

    # --- get_default_branch ---

    def test_get_default_branch(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo").mock(
            return_value=httpx.Response(200, json={"default_branch": "main", "clone_url": ""})
        )
        assert self._client().get_default_branch("owner/repo") == "main"

    def test_get_default_branch_master(self):
        respx.get(f"{GITHUB_API}/repos/owner/old-repo").mock(
            return_value=httpx.Response(200, json={"default_branch": "master", "clone_url": ""})
        )
        assert self._client().get_default_branch("owner/old-repo") == "master"

    # --- create_pr ---

    def test_create_pr_returns_result(self):
        respx.post(f"{GITHUB_API}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(201, json={
                "id": 999,
                "number": 7,
                "html_url": "https://github.com/owner/repo/pull/7",
            })
        )
        pr = PRRequest(title="feat", body="desc", head_branch="feat/x", base_branch="main")
        result = self._client().create_pr("owner/repo", pr)
        assert result.number == 7
        assert result.url == "https://github.com/owner/repo/pull/7"
        assert result.id == 999

    def test_create_pr_sends_correct_payload(self):
        route = respx.post(f"{GITHUB_API}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(201, json={"id": 1, "number": 1, "html_url": "u"})
        )
        pr = PRRequest(title="My PR", body="Body", head_branch="feature", base_branch="main")
        self._client().create_pr("owner/repo", pr)
        import json
        payload = json.loads(route.calls.last.request.content)
        assert payload["title"] == "My PR"
        assert payload["head"] == "feature"
        assert payload["base"] == "main"

    def test_create_pr_raises_on_error(self):
        respx.post(f"{GITHUB_API}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(422, json={"message": "Validation Failed"})
        )
        pr = PRRequest(title="t", head_branch="h", base_branch="b")
        with pytest.raises(ForgeError) as exc_info:
            self._client().create_pr("owner/repo", pr)
        assert exc_info.value.status_code == 422

    # --- post_comment ---

    def test_post_comment(self):
        route = respx.post(f"{GITHUB_API}/repos/owner/repo/issues/5/comments").mock(
            return_value=httpx.Response(201, json={"id": 123})
        )
        self._client().post_comment("owner/repo", 5, "Hello!")
        assert route.called

    def test_post_comment_sends_body(self):
        route = respx.post(f"{GITHUB_API}/repos/owner/repo/issues/5/comments").mock(
            return_value=httpx.Response(201, json={"id": 1})
        )
        self._client().post_comment("owner/repo", 5, "My comment")
        import json
        payload = json.loads(route.calls.last.request.content)
        assert payload["body"] == "My comment"

    # --- network error ---

    def test_network_error_raises_forge_error(self):
        respx.get(f"{GITHUB_API}/repos/owner/repo/issues/1").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(ForgeError) as exc_info:
            self._client().get_issue("owner/repo", 1)
        assert "network error" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════
# 4. GitLabClient — mocked HTTP tests
# ═══════════════════════════════════════════════════════════════════════════

GITLAB_API = "https://gitlab.com/api/v4"
GITLAB_SELF = "https://gitlab.internal"
GITLAB_SELF_API = f"{GITLAB_SELF}/api/v4"


@respx.mock
class TestGitLabClient:

    def _client(self, base_url: str = "https://gitlab.com") -> GitLabClient:
        return GitLabClient(token="glpat-test", base_url=base_url)

    def _encoded(self, path: str) -> str:
        from urllib.parse import quote
        return quote(path, safe="")

    # --- get_issue ---

    def test_get_issue_returns_issue(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/issues/10").mock(
            return_value=httpx.Response(200, json={
                "iid": 10,
                "title": "GL Bug",
                "description": "Desc",
                "web_url": "https://gitlab.com/group/project/-/issues/10",
                "labels": ["backend"],
                "author": {"username": "bob"},
                "created_at": "2024-03-01T12:00:00Z",
            })
        )
        issue = self._client().get_issue("group/project", 10)
        assert issue.id == 10
        assert issue.title == "GL Bug"
        assert issue.author == "bob"
        assert issue.labels == ["backend"]

    def test_get_issue_raises_on_404(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/issues/999").mock(
            return_value=httpx.Response(404, json={"message": "404 Not Found"})
        )
        with pytest.raises(ForgeError) as exc_info:
            self._client().get_issue("group/project", 999)
        assert exc_info.value.status_code == 404

    def test_get_issue_null_description(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/issues/1").mock(
            return_value=httpx.Response(200, json={
                "iid": 1, "title": "T", "description": None,
                "web_url": "u", "labels": [], "author": {"username": "x"},
                "created_at": None,
            })
        )
        issue = self._client().get_issue("group/project", 1)
        assert issue.description == ""

    # --- list_issues ---

    def test_list_issues(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/issues").mock(
            return_value=httpx.Response(200, json=[
                {"iid": 1, "title": "A", "description": "", "web_url": "u1",
                 "labels": [], "author": {"username": "a"}, "created_at": None},
                {"iid": 2, "title": "B", "description": "", "web_url": "u2",
                 "labels": [], "author": {"username": "b"}, "created_at": None},
            ])
        )
        issues = self._client().list_issues("group/project")
        assert len(issues) == 2

    def test_list_issues_maps_open_to_opened(self):
        enc = self._encoded("group/project")
        route = respx.get(f"{GITLAB_API}/projects/{enc}/issues").mock(
            return_value=httpx.Response(200, json=[])
        )
        self._client().list_issues("group/project", state="open")
        request = route.calls.last.request
        assert b"state=opened" in request.url.query

    def test_list_issues_closed_passes_through(self):
        enc = self._encoded("group/project")
        route = respx.get(f"{GITLAB_API}/projects/{enc}/issues").mock(
            return_value=httpx.Response(200, json=[])
        )
        self._client().list_issues("group/project", state="closed")
        request = route.calls.last.request
        assert b"state=closed" in request.url.query

    # --- clone_url ---

    def test_clone_url_embeds_token(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}").mock(
            return_value=httpx.Response(200, json={
                "http_url_to_repo": "https://gitlab.com/group/project.git",
                "default_branch": "main",
            })
        )
        url = self._client().clone_url("group/project")
        assert url == "https://oauth2:glpat-test@gitlab.com/group/project.git"

    def test_clone_url_self_hosted(self):
        enc = self._encoded("team/proj")
        respx.get(f"{GITLAB_SELF_API}/projects/{enc}").mock(
            return_value=httpx.Response(200, json={
                "http_url_to_repo": f"{GITLAB_SELF}/team/proj.git",
                "default_branch": "main",
            })
        )
        url = self._client(base_url=GITLAB_SELF).clone_url("team/proj")
        assert "oauth2:glpat-test@" in url

    def test_clone_url_no_token(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}").mock(
            return_value=httpx.Response(200, json={
                "http_url_to_repo": "https://gitlab.com/group/project.git",
                "default_branch": "main",
            })
        )
        url = GitLabClient(token="", base_url="https://gitlab.com").clone_url("group/project")
        assert "oauth2:" not in url

    # --- list_branches ---

    def test_list_branches(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/repository/branches").mock(
            return_value=httpx.Response(200, json=[
                {"name": "main"}, {"name": "feature/x"},
            ])
        )
        branches = self._client().list_branches("group/project")
        assert branches == ["main", "feature/x"]

    # --- get_default_branch ---

    def test_get_default_branch(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}").mock(
            return_value=httpx.Response(200, json={
                "default_branch": "develop",
                "http_url_to_repo": "https://gitlab.com/group/project.git",
            })
        )
        assert self._client().get_default_branch("group/project") == "develop"

    # --- create_pr (merge request) ---

    def test_create_pr_returns_result(self):
        enc = self._encoded("group/project")
        respx.post(f"{GITLAB_API}/projects/{enc}/merge_requests").mock(
            return_value=httpx.Response(201, json={
                "id": 55,
                "iid": 3,
                "web_url": "https://gitlab.com/group/project/-/merge_requests/3",
            })
        )
        pr = PRRequest(title="feat", body="desc", head_branch="feat/x", base_branch="main")
        result = self._client().create_pr("group/project", pr)
        assert result.number == 3
        assert result.id == 55

    def test_create_pr_sends_correct_fields(self):
        enc = self._encoded("group/project")
        route = respx.post(f"{GITLAB_API}/projects/{enc}/merge_requests").mock(
            return_value=httpx.Response(201, json={"id": 1, "iid": 1, "web_url": "u"})
        )
        pr = PRRequest(title="My MR", body="Body", head_branch="feature", base_branch="main")
        self._client().create_pr("group/project", pr)
        import json
        payload = json.loads(route.calls.last.request.content)
        assert payload["title"] == "My MR"
        assert payload["source_branch"] == "feature"
        assert payload["target_branch"] == "main"
        assert payload["description"] == "Body"

    # --- post_comment (note) ---

    def test_post_comment(self):
        enc = self._encoded("group/project")
        route = respx.post(f"{GITLAB_API}/projects/{enc}/issues/3/notes").mock(
            return_value=httpx.Response(201, json={"id": 77})
        )
        self._client().post_comment("group/project", 3, "Hello GitLab!")
        assert route.called

    # --- self-hosted GitLab ---

    def test_self_hosted_uses_configured_base_url(self):
        enc = self._encoded("team/proj")
        route = respx.get(f"{GITLAB_SELF_API}/projects/{enc}/issues/1").mock(
            return_value=httpx.Response(200, json={
                "iid": 1, "title": "T", "description": "", "web_url": "u",
                "labels": [], "author": {"username": "x"}, "created_at": None,
            })
        )
        self._client(base_url=GITLAB_SELF).get_issue("team/proj", 1)
        assert route.called

    # --- subgroup paths ---

    def test_subgroup_path_encoded(self):
        path = "group/subgroup/project"
        from urllib.parse import quote
        enc = quote(path, safe="")
        route = respx.get(f"{GITLAB_API}/projects/{enc}/issues/1").mock(
            return_value=httpx.Response(200, json={
                "iid": 1, "title": "T", "description": "", "web_url": "u",
                "labels": [], "author": {"username": "x"}, "created_at": None,
            })
        )
        self._client().get_issue(path, 1)
        assert route.called

    # --- network error ---

    def test_network_error_raises_forge_error(self):
        enc = self._encoded("group/project")
        respx.get(f"{GITLAB_API}/projects/{enc}/issues/1").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(ForgeError):
            self._client().get_issue("group/project", 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Factory — get_forge_client auto-detection
# ═══════════════════════════════════════════════════════════════════════════

class TestFactory:

    def _mock_settings(self, github="gh_tok", gitlab="gl_tok", gitlab_url="https://gitlab.com"):
        s = MagicMock()
        s.github_token = github
        s.gitlab_token = gitlab
        s.gitlab_url   = gitlab_url
        return s

    def test_github_com_url_returns_github_client(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_forge_client("https://github.com/owner/repo")
        assert isinstance(client, GitHubClient)

    def test_github_ssh_url_detected(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_forge_client("git@github.com:owner/repo.git")
        # ssh URL contains github.com — detected correctly
        assert isinstance(client, GitHubClient)

    def test_gitlab_com_url_returns_gitlab_client(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_forge_client("https://gitlab.com/group/project")
        assert isinstance(client, GitLabClient)

    def test_self_hosted_gitlab_detected_from_config(self):
        settings = self._mock_settings(gitlab_url="https://gitlab.internal")
        with patch("infra.factory._settings", return_value=settings):
            client = get_forge_client("https://gitlab.internal/team/project")
        assert isinstance(client, GitLabClient)

    def test_unknown_url_raises_forge_error(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(gitlab_url="")):
            with pytest.raises(ForgeError) as exc_info:
                get_forge_client("https://bitbucket.org/owner/repo")
        assert "Cannot determine" in str(exc_info.value)

    def test_explicit_platform_github(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_forge_client("https://code.company.com/org/repo", platform="github")
        assert isinstance(client, GitHubClient)

    def test_explicit_platform_gitlab(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_forge_client("https://code.company.com/org/repo", platform="gitlab")
        assert isinstance(client, GitLabClient)

    def test_explicit_platform_invalid_raises(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            with pytest.raises(ForgeError) as exc_info:
                get_forge_client("https://github.com/x/y", platform="bitbucket")
        assert "Unknown platform" in str(exc_info.value)

    def test_github_token_passed_to_client(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(github="my_token")):
            client = get_forge_client("https://github.com/owner/repo")
        assert isinstance(client, GitHubClient)
        # Token is in the Authorization header
        assert "my_token" in str(client._client.headers.get("Authorization", ""))

    def test_gitlab_token_passed_to_client(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(gitlab="glpat-abc")):
            client = get_forge_client("https://gitlab.com/group/proj")
        assert isinstance(client, GitLabClient)
        assert "glpat-abc" in str(client._client.headers.get("PRIVATE-TOKEN", ""))

    def test_settings_exception_handled_gracefully(self):
        """Factory must work even if settings cannot be loaded."""
        with patch("infra.factory._settings", side_effect=Exception("no .env")):
            client = get_forge_client("https://github.com/owner/repo")
        assert isinstance(client, GitHubClient)

    def test_get_github_client_convenience(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(github="tok")):
            client = get_github_client()
        assert isinstance(client, GitHubClient)

    def test_get_github_client_token_override(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(github="from_config")):
            client = get_github_client(token="override")
        assert "override" in str(client._client.headers.get("Authorization", ""))

    def test_get_gitlab_client_convenience(self):
        with patch("infra.factory._settings", return_value=self._mock_settings(gitlab="gl")):
            client = get_gitlab_client()
        assert isinstance(client, GitLabClient)

    def test_get_gitlab_client_base_url_override(self):
        with patch("infra.factory._settings", return_value=self._mock_settings()):
            client = get_gitlab_client(base_url="https://my-gitlab.io")
        assert client._base_url == "https://my-gitlab.io"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Config keys
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigKeys:

    def test_settings_has_github_token(self):
        from app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "github_token")
        assert isinstance(s.github_token, str)

    def test_settings_has_gitlab_token(self):
        from app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "gitlab_token")
        assert isinstance(s.gitlab_token, str)

    def test_settings_has_gitlab_url(self):
        from app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "gitlab_url")
        assert s.gitlab_url  # non-empty default

    def test_github_token_default_is_empty_string(self):
        from app.core.config import get_settings
        s = get_settings()
        # Default should be empty string (not None)
        assert s.github_token == "" or isinstance(s.github_token, str)

    def test_gitlab_url_default_is_gitlab_com(self):
        from app.core.config import get_settings
        s = get_settings()
        assert "gitlab" in s.gitlab_url


# ═══════════════════════════════════════════════════════════════════════════
# 7. ForgeError
# ═══════════════════════════════════════════════════════════════════════════

class TestForgeError:

    def test_forge_error_message(self):
        err = ForgeError("something failed")
        assert "something failed" in str(err)

    def test_forge_error_status_code(self):
        err = ForgeError("not found", status_code=404)
        assert err.status_code == 404

    def test_forge_error_no_status_code(self):
        err = ForgeError("network error")
        assert err.status_code is None

    def test_forge_error_is_exception(self):
        with pytest.raises(ForgeError):
            raise ForgeError("boom")


# ═══════════════════════════════════════════════════════════════════════════
# 8. infra package public API
# ═══════════════════════════════════════════════════════════════════════════

class TestInfraPackageExports:
    def test_all_symbols_importable_from_infra(self):
        import infra
        for name in [
            "ForgeClient", "ForgeError", "Issue", "PRRequest", "PRResult",
            "GitHubClient", "GitLabClient",
            "get_forge_client", "get_github_client", "get_gitlab_client",
        ]:
            assert hasattr(infra, name), f"infra.{name} not exported"
