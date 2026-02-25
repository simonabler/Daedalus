"""GitLab forge client.

Implements :class:`~infra.forge.ForgeClient` against the GitLab REST API v4.
Works with both gitlab.com and self-hosted GitLab instances — pass the full
base URL (e.g. ``https://gitlab.internal``) via the ``GITLAB_URL`` config key.

Authentication uses a personal access token (PAT) via the ``PRIVATE-TOKEN``
HTTP header.

Usage::

    from infra.factory import get_forge_client
    client = get_forge_client("https://gitlab.com/group/project")
    issue = client.get_issue("group/project", 42)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from infra.forge import ForgeClient, ForgeError, Issue, PRRequest, PRResult


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp returned by GitLab."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.rstrip("Z")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _encode_project(project_path: str) -> str:
    """URL-encode a project path (``group/subgroup/project`` → ``group%2Fsubgroup%2Fproject``)."""
    return quote(project_path, safe="")


class GitLabClient:
    """GitLab REST API v4 client.

    Args:
        token:    GitLab personal access token.
        base_url: GitLab instance URL, e.g. ``"https://gitlab.com"`` or
                  ``"https://gitlab.internal"``.  Trailing slash is stripped.
        timeout:  HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        token: str = "",
        base_url: str = "https://gitlab.com",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_base = f"{self._base_url}/api/v4"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["PRIVATE-TOKEN"] = token
        self._client = httpx.Client(headers=headers, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._api_base}{path}"
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ForgeError(
                f"GitLab GET {path} failed: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ForgeError(f"GitLab GET {path} network error: {exc}") from exc

    def _post(self, path: str, json: dict[str, Any]) -> Any:
        url = f"{self._api_base}{path}"
        try:
            response = self._client.post(url, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ForgeError(
                f"GitLab POST {path} failed: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ForgeError(f"GitLab POST {path} network error: {exc}") from exc

    def _project_path(self, repo: str) -> str:
        """Return ``/projects/encoded-path``."""
        return f"/projects/{_encode_project(repo)}"

    def _issue_from_dict(self, data: dict[str, Any]) -> Issue:
        return Issue(
            id=data.get("iid", data.get("id", 0)),
            title=data.get("title", ""),
            description=data.get("description") or "",
            url=data.get("web_url", ""),
            labels=data.get("labels", []),
            author=data.get("author", {}).get("username", ""),
            created_at=_parse_dt(data.get("created_at")),
        )

    # ------------------------------------------------------------------
    # ForgeClient implementation
    # ------------------------------------------------------------------

    def get_issue(self, repo: str, issue_id: int) -> Issue:
        """Fetch a single GitLab issue by IID.

        Args:
            repo:     Project path (e.g. ``"group/project"``).
            issue_id: Issue IID (project-scoped number shown in the UI).
        """
        data = self._get(f"{self._project_path(repo)}/issues/{issue_id}")
        return self._issue_from_dict(data)

    def list_issues(self, repo: str, state: str = "open") -> list[Issue]:
        """List GitLab issues for a project.

        Args:
            repo:  Project path.
            state: ``"open"``, ``"closed"``, or ``"all"``.
        """
        # GitLab uses "opened" not "open"
        gl_state = "opened" if state == "open" else state
        data = self._get(
            f"{self._project_path(repo)}/issues",
            params={"state": gl_state, "per_page": 100},
        )
        return [self._issue_from_dict(item) for item in data]

    def clone_url(self, repo: str) -> str:
        """Return an authenticated HTTPS clone URL.

        Embeds the token as a URL credential using the ``oauth2`` username
        convention supported by all GitLab versions.

        Args:
            repo: Project path.
        """
        data = self._get(self._project_path(repo))
        clone = data.get("http_url_to_repo", f"{self._base_url}/{repo}.git")
        token = self._client.headers.get("PRIVATE-TOKEN", "")
        if token:
            clone = clone.replace("https://", f"https://oauth2:{token}@")
        return clone

    def list_branches(self, repo: str) -> list[str]:
        """Return branch names for a GitLab project.

        Args:
            repo: Project path.
        """
        data = self._get(
            f"{self._project_path(repo)}/repository/branches",
            params={"per_page": 100},
        )
        return [branch["name"] for branch in data]

    def get_default_branch(self, repo: str) -> str:
        """Return the default branch name for a GitLab project.

        Args:
            repo: Project path.
        """
        data = self._get(self._project_path(repo))
        return data.get("default_branch", "main")

    def create_pr(self, repo: str, pr: PRRequest) -> PRResult:
        """Open a GitLab merge request.

        Args:
            repo: Project path.
            pr:   :class:`~infra.forge.PRRequest` with title, body, branches.
        """
        data = self._post(
            f"{self._project_path(repo)}/merge_requests",
            json={
                "title": pr.title,
                "description": pr.body,
                "source_branch": pr.head_branch,
                "target_branch": pr.base_branch,
            },
        )
        return PRResult(
            id=data["id"],
            url=data["web_url"],
            number=data["iid"],
        )

    def post_comment(self, repo: str, issue_id: int, body: str) -> None:
        """Post a note on a GitLab issue.

        Args:
            repo:     Project path.
            issue_id: Issue IID.
            body:     Comment / note body.
        """
        self._post(
            f"{self._project_path(repo)}/issues/{issue_id}/notes",
            json={"body": body},
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"GitLabClient(base_url={self._base_url!r})"
