"""Per-task repository root context variable.

All tools (filesystem, git, shell, build) call :func:`get_repo_root` instead
of reading ``settings.target_repo_path`` directly.  This makes the sandbox
root dynamic: each workflow task can operate on a different repository without
mutating the global Settings singleton.

The :class:`contextvars.ContextVar` is correctly isolated per async task, so
future parallel task execution will also be safe.

Typical call-site pattern
-------------------------
In ``context_loader_node`` (or any node that establishes the repo root)::

    from app.core.active_repo import set_repo_root
    set_repo_root("/path/to/cloned/repo")

In tools::

    from app.core.active_repo import get_repo_root
    root = Path(get_repo_root()).resolve()
"""

from __future__ import annotations

from contextvars import ContextVar

# ---------------------------------------------------------------------------
# Module-level ContextVar
# ---------------------------------------------------------------------------

_repo_root_var: ContextVar[str] = ContextVar("daedalus_repo_root", default="")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_repo_root(path: str) -> None:
    """Set the repository root for the current async-task context.

    Args:
        path: Absolute path to the repository root directory.
              An empty string resets to the settings fallback.
    """
    _repo_root_var.set(path)


def get_repo_root() -> str:
    """Return the repository root for the current context.

    Resolution order:

    1. Value set by :func:`set_repo_root` in the current async context.
    2. ``settings.target_repo_path`` (static env-var, backward-compat).
    3. Empty string (callers must guard against this).

    Returns:
        Absolute path string, or ``""`` if not configured.
    """
    value = _repo_root_var.get("")
    if value:
        return value

    # Lazy import to avoid circular dependency at module load time.
    try:
        from app.core.config import get_settings
        return get_settings().target_repo_path or ""
    except Exception:
        return ""


def clear_repo_root() -> None:
    """Reset the context variable to the empty default.

    Useful in tests to ensure isolation between test cases.
    """
    _repo_root_var.set("")
