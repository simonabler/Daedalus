"""Daedalus infrastructure layer — forge API clients.

All external forge communication (GitHub, GitLab) goes through this package.
Use :func:`~infra.factory.get_forge_client` to obtain a client instance.

Quick start::

    from infra.factory import get_forge_client

    client = get_forge_client("https://github.com/owner/repo")
    issue  = client.get_issue("owner/repo", 42)
    pr     = client.create_pr("owner/repo", PRRequest(
        title="feat: add health endpoint",
        body="Implements /api/health as discussed in #42.",
        head_branch="feat/health",
        base_branch="main",
    ))
"""

from infra.factory import get_forge_client, get_github_client, get_gitlab_client
from infra.forge import ForgeClient, ForgeError, Issue, PRRequest, PRResult
from infra.github_client import GitHubClient
from infra.gitlab_client import GitLabClient
from infra.workspace import WorkspaceError, WorkspaceInfo, WorkspaceManager

__all__ = [
    # Protocol & models
    "ForgeClient",
    "ForgeError",
    "Issue",
    "PRRequest",
    "PRResult",
    # Clients
    "GitHubClient",
    "GitLabClient",
    # Factory
    "get_forge_client",
    "get_github_client",
    "get_gitlab_client",
    # Workspace
    "WorkspaceManager",
    "WorkspaceInfo",
    "WorkspaceError",
]
