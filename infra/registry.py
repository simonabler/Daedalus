"""Repo registry — YAML-based access control for Daedalus.

Only repositories listed in ``repos.yaml`` (project root) may be cloned or
modified by Daedalus.  Any task targeting an unknown repo is rejected before
the workflow starts.

Schema example (``repos.yaml``)::

    repos:
      - name: my-api
        url: https://github.com/org/my-api
        default_branch: main
        description: "Main backend API"

      - name: infra-scripts
        url: https://gitlab.internal/ops/infra-scripts
        default_branch: main
        description: "Internal infrastructure scripts"

Typical usage::

    from infra.registry import get_registry

    registry = get_registry()          # loads repos.yaml on first call
    entry    = registry.resolve("my-api")
    if not registry.is_allowed("https://github.com/org/my-api"):
        raise ValueError("Repo not in registry")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, field_validator

from app.core.logging import get_logger

logger = get_logger("infra.registry")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_DEFAULT_REPOS_YAML = Path(__file__).parent.parent / "repos.yaml"


class RepoEntry(BaseModel):
    """A single repository entry in the registry."""

    name: str
    """Short alias used in Telegram tasks and the Web UI (e.g. ``my-api``)."""

    url: str
    """Full forge URL (e.g. ``https://github.com/org/my-api``)."""

    default_branch: str = "main"
    """Branch that Daedalus checks out and targets for PRs/MRs."""

    description: str = ""
    """Human-readable description shown in ``/status`` and the Web UI."""

    @field_validator("name")
    @classmethod
    def _name_no_spaces(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()

    @field_validator("url")
    @classmethod
    def _url_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("url must not be empty")
        return v.strip().rstrip("/")

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def owner_name(self) -> str:
        """``owner/name`` shorthand extracted from the URL, e.g. ``org/my-api``."""
        parsed = urlparse(self.url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return ""

    @property
    def canonical_key(self) -> str:
        """``host/owner/name`` key, e.g. ``github.com/org/my-api``."""
        parsed = urlparse(self.url)
        return f"{parsed.netloc}/{parsed.path.strip('/')}"

    def matches(self, ref: str) -> bool:
        """Return True if *ref* refers to this entry.

        Accepted forms:
        - Name alias      → ``"my-api"``
        - Full URL        → ``"https://github.com/org/my-api"``
        - No-scheme URL   → ``"github.com/org/my-api"``
        - owner/name      → ``"org/my-api"``
        """
        ref = ref.strip().rstrip("/")
        if not ref:
            return False

        normalised = ref.lower()

        # 1. Name alias (case-insensitive)
        if normalised == self.name.lower():
            return True

        # 2. Full URL (normalise scheme differences)
        url_no_scheme = re.sub(r"^https?://", "", self.url.lower())
        ref_no_scheme = re.sub(r"^https?://", "", normalised)
        if ref_no_scheme == url_no_scheme:
            return True

        # 3. owner/name shorthand
        if self.owner_name and normalised == self.owner_name.lower():
            return True

        # 4. host/owner/name canonical key
        if normalised == self.canonical_key.lower():
            return True

        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RepoRegistry:
    """In-memory registry loaded from ``repos.yaml``.

    Thread-safe for reads after :meth:`load` completes.
    """

    def __init__(self) -> None:
        self._entries: list[RepoEntry] = []
        self._loaded_path: Path | None = None

    # ── Loading ──────────────────────────────────────────────────────────

    def load(self, path: Path | None = None) -> None:
        """Load (or reload) the registry from a YAML file.

        Args:
            path: Path to the YAML file.  Defaults to ``repos.yaml`` in the
                  project root next to this package.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is malformed or contains invalid entries.
        """
        resolved = Path(path) if path else _DEFAULT_REPOS_YAML
        if not resolved.exists():
            logger.warning("repos.yaml not found at %s — registry is empty", resolved)
            self._entries = []
            self._loaded_path = resolved
            return

        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        raw_repos = raw.get("repos", [])
        if not isinstance(raw_repos, list):
            raise ValueError(f"repos.yaml: 'repos' must be a list, got {type(raw_repos)}")

        entries: list[RepoEntry] = []
        for i, item in enumerate(raw_repos):
            if not isinstance(item, dict):
                raise ValueError(f"repos.yaml: entry {i} must be a mapping, got {type(item)}")
            try:
                entries.append(RepoEntry(**item))
            except Exception as exc:
                raise ValueError(f"repos.yaml: invalid entry {i} ({item!r}): {exc}") from exc

        self._entries = entries
        self._loaded_path = resolved
        logger.info(
            "Registry loaded %d repo(s) from %s",
            len(self._entries),
            resolved,
        )

    def _ensure_loaded(self) -> None:
        """Auto-load from the default path if not yet loaded."""
        if self._loaded_path is None:
            self.load()

    # ── Queries ──────────────────────────────────────────────────────────

    def resolve(self, ref: str) -> RepoEntry:
        """Return the :class:`RepoEntry` that matches *ref*.

        Matching is tried in this order: name alias → full URL →
        no-scheme URL → ``owner/name`` shorthand → canonical key.

        Args:
            ref: Any of the accepted reference forms.

        Returns:
            The matching :class:`RepoEntry`.

        Raises:
            ValueError: If no entry matches *ref*.
        """
        self._ensure_loaded()
        for entry in self._entries:
            if entry.matches(ref):
                return entry
        raise ValueError(
            f"Repository {ref!r} is not in the registry.  "
            f"Known repos: {[e.name for e in self._entries]}"
        )

    def is_allowed(self, ref: str) -> bool:
        """Return True if *ref* matches any registered repository.

        Args:
            ref: Name alias, URL, or ``owner/name`` shorthand.
        """
        self._ensure_loaded()
        return any(entry.matches(ref) for entry in self._entries)

    def list_repos(self) -> list[RepoEntry]:
        """Return all registered repositories (read-only copy)."""
        self._ensure_loaded()
        return list(self._entries)

    def __iter__(self) -> Iterator[RepoEntry]:
        self._ensure_loaded()
        return iter(self._entries)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._entries)

    def __bool__(self) -> bool:
        return True  # always truthy even when empty


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_registry: RepoRegistry | None = None


def get_registry(path: Path | None = None, *, reload: bool = False) -> RepoRegistry:
    """Return the singleton :class:`RepoRegistry`, loading it on first call.

    Resolution order for the YAML file:
    1. ``path`` argument (tests / explicit override)
    2. ``REPOS_YAML_PATH`` env / config setting
    3. ``repos.yaml`` next to the project root (default)

    Args:
        path:   Override the default ``repos.yaml`` path (mainly for tests).
        reload: Force a reload even if already loaded.
    """
    global _registry
    if _registry is None or reload or path is not None:
        resolved_path = path
        if resolved_path is None:
            # Respect config setting if provided
            try:
                from app.core.config import get_settings
                cfg_path = get_settings().repos_yaml_path
                if cfg_path:
                    resolved_path = Path(cfg_path).expanduser().resolve()
            except Exception:
                pass

        r = RepoRegistry()
        r.load(resolved_path)
        if path is None and not reload:
            _registry = r
        else:
            return r
    return _registry
