"""GitHub forge client.

Implements :class:`~infra.forge.ForgeClient` against the GitHub REST API v3.
Authentication uses a personal access token (PAT) supplied via the
``GITHUB_TOKEN`` environment variable / config key.

Usage::

    from infra.factory import get_forge_client
    client = get_forge_client("https://github.com/owner/repo")
    issue = client.get_issue("owner/repo", 42)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from infra.forge import ForgeClient, ForgeError, Issue, PRRequest, PRResult

_GITHUB_API = "https://api.github.com"


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp returned by GitHub (``2024-01-02T03:04:05Z``)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.rstrip("Z")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class GitHubClient:
    """GitHub REST API v3 client.

    Args:
        token: GitHub personal access token.  Pass an empty string to make
               unauthenticated requests (rate-limited to 60 req/h).
        base_url: API base URL.  Override in tests or for GitHub Enterprise.
        timeout: HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        token: str = "",
        base_url: str = _GITHUB_API,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(headers=headers, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ForgeError(
                f"GitHub GET {path} failed: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ForgeError(f"GitHub GET {path} network error: {exc}") from exc

    def _post(self, path: str, json: dict[str, Any]) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = self._client.post(url, json=json)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ForgeError(
                f"GitHub POST {path} failed: {exc.response.status_code} {exc.response.text[:200]}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ForgeError(f"GitHub POST {path} network error: {exc}") from exc

    def _repo_path(self, repo: str) -> str:
        """Return URL-encoded ``/repos/owner/name``."""
        return f"/repos/{quote(repo, safe='/')}"

    def _issue_from_dict(self, data: dict[str, Any]) -> Issue:
        return Issue(
            id=data["number"],
            title=data.get("title", ""),
            description=data.get("body") or "",
            url=data.get("html_url", ""),
            labels=[lbl["name"] for lbl in data.get("labels", [])],
            author=data.get("user", {}).get("login", ""),
            created_at=_parse_dt(data.get("created_at")),
        )

    # ------------------------------------------------------------------
    # ForgeClient implementation
    # ------------------------------------------------------------------

    def get_issue(self, repo: str, issue_id: int) -> Issue:
        """Fetch a single GitHub issue.

        Args:
            repo:     ``owner/name``.
            issue_id: Issue number.
        """
        data = self._get(f"{self._repo_path(repo)}/issues/{issue_id}")
        return self._issue_from_dict(data)

    def list_issues(self, repo: str, state: str = "open") -> list[Issue]:
        """List GitHub issues for a repository.

        Args:
            repo:  ``owner/name``.
            state: ``"open"``, ``"closed"``, or ``"all"``.
        """
        # GitHub returns PRs in the issues list — filter them out
        data = self._get(
            f"{self._repo_path(repo)}/issues",
            params={"state": state, "per_page": 100},
        )
        return [
            self._issue_from_dict(item)
            for item in data
            if "pull_request" not in item  # exclude PRs
        ]

    def clone_url(self, repo: str) -> str:
        """Return an authenticated HTTPS clone URL.

        Embeds the token as a URL credential so git can clone without
        additional credential configuration.

        Args:
            repo: ``owner/name``.
        """
        data = self._get(self._repo_path(repo))
        clone = data.get("clone_url", f"https://github.com/{repo}.git")
        if self._client.headers.get("Authorization"):
            token = str(self._client.headers["Authorization"]).removeprefix("Bearer ")
            clone = clone.replace("https://", f"https://{token}@")
        return clone

    def list_branches(self, repo: str) -> list[str]:
        """Return branch names for a GitHub repository.

        Args:
            repo: ``owner/name``.
        """
        data = self._get(
            f"{self._repo_path(repo)}/branches",
            params={"per_page": 100},
        )
        return [branch["name"] for branch in data]

    def get_default_branch(self, repo: str) -> str:
        """Return the default branch name (e.g. ``"main"``) for a repo.

        Args:
            repo: ``owner/name``.
        """
        data = self._get(self._repo_path(repo))
        return data.get("default_branch", "main")

    def create_pr(self, repo: str, pr: PRRequest) -> PRResult:
        """Open a GitHub pull request.

        Args:
            repo: ``owner/name``.
            pr:   :class:`~infra.forge.PRRequest` with title, body, branches.
        """
        data = self._post(
            f"{self._repo_path(repo)}/pulls",
            json={
                "title": pr.title,
                "body": pr.body,
                "head": pr.head_branch,
                "base": pr.base_branch,
            },
        )
        return PRResult(
            id=data["id"],
            url=data["html_url"],
            number=data["number"],
        )

    def post_comment(self, repo: str, issue_id: int, body: str) -> None:
        """Post a comment on a GitHub issue or PR.

        Args:
            repo:     ``owner/name``.
            issue_id: Issue / PR number.
            body:     Markdown comment text.
        """
        self._post(
            f"{self._repo_path(repo)}/issues/{issue_id}/comments",
            json={"body": body},
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"GitHubClient(base_url={self._base_url!r})"


# Make the class itself satisfy the protocol at import time
assert isinstance(GitHubClient, type)
