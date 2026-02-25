"""Forge client factory.

:func:`get_forge_client` is the single entry-point for obtaining a
``ForgeClient`` instance.  It auto-detects the platform from the repository
URL and reads credentials from the application config.

Usage::

    from infra.factory import get_forge_client

    # GitHub (any github.com URL)
    client = get_forge_client("https://github.com/owner/repo")

    # GitLab.com
    client = get_forge_client("https://gitlab.com/group/project")

    # Self-hosted GitLab (URL must match GITLAB_URL in config)
    client = get_forge_client("https://gitlab.internal/team/project")

    # Explicit platform override
    client = get_forge_client("https://code.company.com/org/repo", platform="gitlab")
"""

from __future__ import annotations

from infra.forge import ForgeClient, ForgeError
from infra.github_client import GitHubClient
from infra.gitlab_client import GitLabClient


def _settings():  # pragma: no cover — thin wrapper, tested via integration
    """Lazy import to avoid circular imports and allow test overrides."""
    from app.core.config import get_settings
    return get_settings()


def get_forge_client(
    url: str,
    platform: str | None = None,
) -> ForgeClient:
    """Return a :class:`~infra.forge.ForgeClient` for *url*.

    Platform detection rules (first match wins):

    1. If *platform* is ``"github"`` → return :class:`GitHubClient`.
    2. If *platform* is ``"gitlab"`` → return :class:`GitLabClient`.
    3. If *url* contains ``"github.com"`` → GitHub.
    4. If *url* contains ``"gitlab.com"`` → GitLab (gitlab.com instance).
    5. If the configured ``GITLAB_URL`` is a prefix of *url* → GitLab
       (self-hosted instance).
    6. Otherwise raise :class:`~infra.forge.ForgeError`.

    Args:
        url:      Any URL that identifies the forge — can be the repository
                  clone URL, web URL, or just the forge base URL.
        platform: Optional override: ``"github"`` or ``"gitlab"``.

    Returns:
        A :class:`~infra.forge.ForgeClient` instance ready to make API calls.

    Raises:
        ForgeError: If the platform cannot be determined from *url*.
    """
    try:
        settings = _settings()
        github_token: str = getattr(settings, "github_token", "") or ""
        gitlab_token: str = getattr(settings, "gitlab_token", "") or ""
        gitlab_url: str   = getattr(settings, "gitlab_url", "https://gitlab.com") or "https://gitlab.com"
    except Exception:  # settings not available (e.g. unit tests without .env)
        github_token = ""
        gitlab_token = ""
        gitlab_url   = "https://gitlab.com"

    # ── Explicit override ────────────────────────────────────────────────
    if platform is not None:
        platform = platform.lower().strip()
        if platform == "github":
            return GitHubClient(token=github_token)
        if platform == "gitlab":
            return GitLabClient(token=gitlab_token, base_url=gitlab_url)
        raise ForgeError(
            f"Unknown platform {platform!r}. Must be 'github' or 'gitlab'."
        )

    # ── Auto-detection from URL ──────────────────────────────────────────
    url_lower = url.lower()

    if "github.com" in url_lower:
        return GitHubClient(token=github_token)

    if "gitlab.com" in url_lower:
        return GitLabClient(token=gitlab_token, base_url="https://gitlab.com")

    # Self-hosted GitLab: check if url starts with the configured GITLAB_URL
    gitlab_url_lower = gitlab_url.rstrip("/").lower()
    if gitlab_url_lower and url_lower.startswith(gitlab_url_lower):
        return GitLabClient(token=gitlab_token, base_url=gitlab_url)

    raise ForgeError(
        f"Cannot determine forge platform from URL {url!r}. "
        "Set GITLAB_URL in config for self-hosted GitLab instances, "
        "or pass platform='github'/'gitlab' explicitly."
    )


def get_github_client(token: str = "") -> GitHubClient:
    """Convenience factory for a GitHub client.

    Uses ``GITHUB_TOKEN`` from config if *token* is not provided.

    Args:
        token: Optional PAT override.
    """
    if not token:
        try:
            token = getattr(_settings(), "github_token", "") or ""
        except Exception:
            token = ""
    return GitHubClient(token=token)


def get_gitlab_client(token: str = "", base_url: str = "") -> GitLabClient:
    """Convenience factory for a GitLab client.

    Uses ``GITLAB_TOKEN`` and ``GITLAB_URL`` from config as defaults.

    Args:
        token:    Optional PAT override.
        base_url: Optional base URL override (for self-hosted instances).
    """
    if not token or not base_url:
        try:
            s = _settings()
            token    = token    or getattr(s, "gitlab_token", "") or ""
            base_url = base_url or getattr(s, "gitlab_url",   "") or "https://gitlab.com"
        except Exception:
            base_url = base_url or "https://gitlab.com"
    return GitLabClient(token=token, base_url=base_url)
