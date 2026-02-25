"""Dynamic workspace manager — clone and maintain per-repo working directories.

:class:`WorkspaceManager` replaces the static ``TARGET_REPO_PATH`` env var.
Given any repo reference (URL, ``owner/name`` shorthand, or fully-qualified
forge path), it either clones the repo on first use or pulls the latest
changes on subsequent uses.

Workspace layout::

    ~/daedalus-workspace/          ← DAEDALUS_WORKSPACE_DIR
      github.com/owner/repo-a/     ← one dir per repo
      gitlab.com/group/project/
      gitlab.internal/team/repo/

All git operations are done via ``subprocess`` directly (not the ``git_command``
LangChain tool) because this code runs *before* the workflow state has a
``repo_root``, which is exactly what we are establishing here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from app.core.logging import get_logger

logger = get_logger("infra.workspace")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WorkspaceError(Exception):
    """Raised when a workspace operation cannot be completed."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceInfo:
    """Metadata about a locally cloned repository workspace."""

    repo_ref: str
    """Canonical forge reference, e.g. ``github.com/owner/repo``."""

    local_path: Path
    """Absolute path to the local clone."""

    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """Approximate time of last use (mtime of the directory)."""


# ---------------------------------------------------------------------------
# Reference normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_ref(repo_ref: str) -> tuple[str, str]:
    """Normalise *repo_ref* to ``(canonical_key, clone_url)``.

    Returns
    -------
    canonical_key
        A filesystem-safe identifier like ``github.com/owner/repo`` that
        mirrors the workspace directory structure.
    clone_url
        The HTTPS URL that git should clone from.  May be the same as
        *repo_ref* if it was already a full URL.

    Accepted input formats
    ----------------------
    - ``https://github.com/owner/repo``
    - ``https://github.com/owner/repo.git``
    - ``https://gitlab.com/group/subgroup/project``
    - ``https://gitlab.internal/team/proj.git``
    - ``owner/repo``  → treated as GitHub (github.com/owner/repo)
    - ``github.com/owner/repo``
    - ``gitlab.com/group/project``

    Raises
    ------
    WorkspaceError
        If the reference cannot be parsed into a valid forge path.
    """
    ref = repo_ref.strip()
    if not ref:
        raise WorkspaceError("repo_ref must not be empty")

    # ── Full HTTPS URL ────────────────────────────────────────────────
    if ref.startswith("http://") or ref.startswith("https://"):
        parsed = urlparse(ref)
        host = parsed.hostname or ""
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            raise WorkspaceError(
                f"URL {ref!r} does not contain at least owner/repo components"
            )
        canonical = f"{host}/{'/'.join(parts)}"
        # Reconstruct a clean https URL (without .git suffix for the key)
        clone_url = f"https://{host}/{'/'.join(parts)}.git"
        return canonical, clone_url

    # ── host/owner/repo (no scheme) ───────────────────────────────────
    # e.g. "github.com/owner/repo" or "gitlab.internal/team/project"
    if re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/", ref):
        path = ref.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = path.split("/")
        if len(parts) < 3:
            raise WorkspaceError(
                f"Forge path {ref!r} must be host/owner/repo (at least 3 segments)"
            )
        canonical = path
        clone_url = f"https://{path}.git"
        return canonical, clone_url

    # ── Short-form: owner/repo — assume GitHub ────────────────────────
    parts = [p for p in ref.split("/") if p]
    if len(parts) == 2:
        owner, name = parts
        canonical = f"github.com/{owner}/{name}"
        clone_url = f"https://github.com/{owner}/{name}.git"
        return canonical, clone_url

    raise WorkspaceError(
        f"Cannot parse repo reference {ref!r}. "
        "Expected a full URL, 'host/owner/repo', or 'owner/repo' (GitHub shorthand)."
    )


def _authenticated_clone_url(canonical_key: str, plain_url: str) -> str:
    """Return an authenticated clone URL by querying the forge client.

    Falls back to the unauthenticated *plain_url* if the forge client cannot
    be constructed (e.g. no token configured).

    The forge client's ``clone_url`` method returns a URL with the token
    embedded as a URL credential, e.g.
    ``https://<token>@github.com/owner/repo.git``.
    """
    try:
        from infra.factory import get_forge_client
        from infra.forge import ForgeError

        # Use the plain https URL for platform detection
        client = get_forge_client(plain_url)

        # The repo path for the forge API is everything after host/
        host_slash = canonical_key.index("/")
        repo_path = canonical_key[host_slash + 1:]

        return client.clone_url(repo_path)
    except Exception as exc:
        logger.warning(
            "workspace: could not get authenticated URL for %s: %s — using plain URL",
            canonical_key,
            exc,
        )
        return plain_url


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 300) -> str:
    """Run a git sub-command and return stdout.

    Raises
    ------
    WorkspaceError
        If git exits with a non-zero return code.
    """
    cmd = ["git"] + args
    logger.debug("workspace git | cwd=%s | %s", cwd, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise WorkspaceError("git executable not found on PATH") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise WorkspaceError(
            f"git {' '.join(args[:2])} failed (exit {result.returncode}): {stderr[:400]}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """Manage per-repo clone directories under a shared workspace root.

    Args:
        workspace_root: Root directory that will contain all cloned repos.
                        Created automatically if it does not exist.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root.expanduser().resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, repo_ref: str) -> Path:
        """Return the local path for *repo_ref*, cloning or pulling as needed.

        Behaviour
        ---------
        - **Fresh clone**: If the local directory does not exist, the repo is
          cloned using an authenticated URL from the forge client.
        - **Existing clone**: ``git fetch origin`` → ``git checkout <default>``
          → ``git pull --ff-only`` so the working tree is up-to-date.

        Args:
            repo_ref: URL, ``owner/name``, or ``host/owner/name``.

        Returns:
            Absolute :class:`~pathlib.Path` to the local clone.

        Raises:
            WorkspaceError: On any git failure or unparseable reference.
        """
        canonical, plain_url = _normalise_ref(repo_ref)
        local_path = self._local_path(canonical)

        if local_path.exists():
            self._pull(local_path, canonical)
        else:
            self._clone(canonical, plain_url, local_path)

        return local_path

    def clean(self, repo_ref: str) -> None:
        """Remove the local clone for *repo_ref*.

        No-op if the directory does not exist.

        Args:
            repo_ref: Same formats accepted by :meth:`resolve`.
        """
        try:
            canonical, _ = _normalise_ref(repo_ref)
        except WorkspaceError:
            logger.warning("workspace.clean: cannot parse ref %r — skipping", repo_ref)
            return

        local_path = self._local_path(canonical)
        if local_path.exists():
            shutil.rmtree(local_path)
            logger.info("workspace.clean: removed %s", local_path)
            # Remove empty parent dirs (owner/ and host/) if possible
            for parent in (local_path.parent, local_path.parent.parent):
                try:
                    parent.rmdir()  # only succeeds if empty
                except OSError:
                    pass

    def list_workspaces(self) -> list[WorkspaceInfo]:
        """Return information about all locally cloned workspaces.

        Scans ``workspace_root`` for directories that contain a ``.git``
        sub-directory (depth ≤ 4 to match ``host/owner/name``).

        Returns:
            List of :class:`WorkspaceInfo` instances, sorted by ``local_path``.
        """
        if not self._root.exists():
            return []

        results: list[WorkspaceInfo] = []
        # Walk up to 4 levels deep: host / owner / repo
        for candidate in sorted(self._root.rglob(".git")):
            repo_dir = candidate.parent
            if not repo_dir.is_dir():
                continue
            try:
                rel = repo_dir.relative_to(self._root)
                ref = str(rel).replace("\\", "/")
                mtime = datetime.fromtimestamp(repo_dir.stat().st_mtime, tz=timezone.utc)
                results.append(WorkspaceInfo(
                    repo_ref=ref,
                    local_path=repo_dir,
                    last_used=mtime,
                ))
            except (ValueError, OSError):
                continue
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_path(self, canonical: str) -> Path:
        """Translate a canonical key to an absolute local path."""
        # canonical is like "github.com/owner/repo" — replace / with OS sep
        rel = Path(*canonical.split("/"))
        return self._root / rel

    def _clone(self, canonical: str, plain_url: str, local_path: Path) -> None:
        """Clone *plain_url* (with auth) into *local_path*."""
        auth_url = _authenticated_clone_url(canonical, plain_url)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("workspace.clone: %s → %s", canonical, local_path)
        _run_git(["clone", auth_url, str(local_path)])
        logger.info("workspace.clone: done")

    def _pull(self, local_path: Path, canonical: str) -> None:
        """Bring an existing clone up-to-date (fetch + checkout default + pull)."""
        logger.info("workspace.pull: updating %s", local_path)

        # 1. Fetch all remote changes
        try:
            _run_git(["fetch", "origin"], cwd=local_path)
        except WorkspaceError as exc:
            logger.warning("workspace.pull: fetch failed: %s — continuing with local state", exc)
            return

        # 2. Determine the default branch (origin/HEAD if available)
        default_branch = self._detect_default_branch(local_path)

        # 3. Checkout default branch and fast-forward
        try:
            _run_git(["checkout", default_branch], cwd=local_path)
            _run_git(["pull", "--ff-only", "origin", default_branch], cwd=local_path)
            logger.info("workspace.pull: updated %s to latest %s", canonical, default_branch)
        except WorkspaceError as exc:
            logger.warning(
                "workspace.pull: pull failed for %s: %s — using existing state",
                canonical, exc,
            )

    def _detect_default_branch(self, local_path: Path) -> str:
        """Return the remote default branch name, falling back to 'main'."""
        # Try to read from origin/HEAD symbolic ref
        for strategy in (
            ["symbolic-ref", "refs/remotes/origin/HEAD", "--short"],
            ["rev-parse", "--abbrev-ref", "origin/HEAD"],
        ):
            try:
                out = _run_git(strategy, cwd=local_path)
                # e.g. "origin/main" → "main"
                return out.split("/", 1)[-1].strip() or "main"
            except WorkspaceError:
                continue

        # Inspect remote branches for common default names
        try:
            branches_out = _run_git(
                ["branch", "-r", "--format=%(refname:short)"], cwd=local_path
            )
            remote_branches = [b.strip() for b in branches_out.splitlines()]
            for candidate in ("origin/main", "origin/master", "origin/develop"):
                if candidate in remote_branches:
                    return candidate.split("/", 1)[-1]
        except WorkspaceError:
            pass

        return "main"

    def __repr__(self) -> str:  # pragma: no cover
        return f"WorkspaceManager(root={self._root!r})"
