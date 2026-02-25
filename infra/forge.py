"""Unified forge interface — abstract protocol and shared data models.

All code that needs to talk to a source-code forge (GitHub or GitLab)
must go through a ``ForgeClient`` implementation.  Direct HTTP calls to
forge APIs outside this package are not allowed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Issue(BaseModel):
    """A forge issue (GitHub issue or GitLab issue)."""

    id: int
    title: str
    description: str = ""
    url: str
    labels: list[str] = Field(default_factory=list)
    author: str = ""
    created_at: datetime | None = None


class PRRequest(BaseModel):
    """Payload for creating a pull / merge request."""

    title: str
    body: str = ""
    head_branch: str
    base_branch: str


class PRResult(BaseModel):
    """Result returned after a PR/MR is created."""

    id: int
    url: str
    number: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ForgeClient(Protocol):
    """Minimal forge operations required by Daedalus.

    Both ``GitHubClient`` and ``GitLabClient`` implement this protocol.
    Callers should type-hint against ``ForgeClient``, not against a concrete
    implementation class.

    All methods are synchronous.  Async callers should run them in a thread
    pool (e.g. ``asyncio.to_thread``).
    """

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def get_issue(self, repo: str, issue_id: int) -> Issue:
        """Fetch a single issue by numeric ID.

        Args:
            repo:     ``owner/name`` for GitHub; project path for GitLab.
            issue_id: Issue number (GitHub) or IID (GitLab).

        Returns:
            Populated :class:`Issue` instance.

        Raises:
            ForgeError: on any HTTP or parsing error.
        """
        ...

    def list_issues(self, repo: str, state: str = "open") -> list[Issue]:
        """List issues for a repository.

        Args:
            repo:  ``owner/name`` (GitHub) or project path (GitLab).
            state: ``"open"``, ``"closed"``, or ``"all"``.

        Returns:
            List of :class:`Issue` instances (may be empty).
        """
        ...

    # ------------------------------------------------------------------
    # Repository helpers
    # ------------------------------------------------------------------

    def clone_url(self, repo: str) -> str:
        """Return an authenticated HTTPS clone URL.

        The token is embedded in the URL so callers can pass it directly to
        ``git clone`` without additional credential configuration.

        Args:
            repo: ``owner/name`` (GitHub) or project path (GitLab).
        """
        ...

    def list_branches(self, repo: str) -> list[str]:
        """Return branch names for a repository.

        Args:
            repo: ``owner/name`` (GitHub) or project path (GitLab).
        """
        ...

    def get_default_branch(self, repo: str) -> str:
        """Return the name of the default branch (e.g. ``"main"``).

        Args:
            repo: ``owner/name`` (GitHub) or project path (GitLab).
        """
        ...

    # ------------------------------------------------------------------
    # Pull / Merge requests
    # ------------------------------------------------------------------

    def create_pr(self, repo: str, pr: PRRequest) -> PRResult:
        """Open a pull request (GitHub) or merge request (GitLab).

        Args:
            repo: ``owner/name`` (GitHub) or project path (GitLab).
            pr:   :class:`PRRequest` with title, body, head/base branches.

        Returns:
            :class:`PRResult` with the new PR id, url, and number.
        """
        ...

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    def post_comment(self, repo: str, issue_id: int, body: str) -> None:
        """Post a comment on an issue or pull/merge request.

        Args:
            repo:     ``owner/name`` (GitHub) or project path (GitLab).
            issue_id: Issue number (GitHub) or IID (GitLab).
            body:     Markdown comment body.
        """
        ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ForgeError(Exception):
    """Raised for any forge API error (HTTP errors, missing fields, …)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover
        return f"ForgeError({self.args[0]!r}, status_code={self.status_code})"
