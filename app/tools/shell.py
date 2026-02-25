"""Safe POSIX shell execution with sandbox and blocklist enforcement."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from app.core.active_repo import get_repo_root
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.shell")

IS_WINDOWS = platform.system().lower().startswith("win")
SHELL_ENABLED = not IS_WINDOWS

# Patterns that must never be executed.
BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[rRf]+\s+)*/((?!home)|$)", re.IGNORECASE),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*:\|:.*\}"),
    re.compile(r"\bchmod\s+(-R\s+)?777\s+/"),
    re.compile(r"\bcurl\s+.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\bwget\s+.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\s+"),
    re.compile(r"\bpasswd\b"),
    re.compile(r"\buseradd\b"),
    re.compile(r"\buserdel\b"),
    re.compile(r"\bsystemctl\b"),
    re.compile(r"\bservice\s+"),
    re.compile(r"\biptables\b"),
    re.compile(r"\bmount\b"),
    re.compile(r"\bumount\b"),
]


def _is_blocked(command: str) -> str | None:
    """Return a reason if command is blocked, else None."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"Blocked by safety rule: {pattern.pattern}"
    return None


def _is_within_root(path: Path, root: Path) -> bool:
    """Return True only if *path* is inside *root*."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _truncate(text: str) -> str:
    limit = get_settings().max_output_chars
    if len(text) > limit:
        half = limit // 2
        return text[:half] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-half:]
    return text


@tool
def run_shell(command: str, working_dir: str = ".") -> str:
    """Execute a shell command inside the target repository.

    This tool is only enabled on non-Windows systems.
    """
    if not SHELL_ENABLED:
        return "DISABLED: run_shell is only enabled on non-Windows systems."

    settings = get_settings()
    root = Path(get_repo_root()).resolve()
    cwd = (root / working_dir).resolve()

    if not _is_within_root(cwd, root):
        msg = f"BLOCKED: working_dir escapes repo root: {working_dir}"
        logger.warning(msg)
        return msg

    if not cwd.is_dir():
        return f"ERROR: directory does not exist: {working_dir}"

    reason = _is_blocked(command)
    if reason:
        logger.warning("BLOCKED shell | %s | reason: %s", command, reason)
        return f"BLOCKED: {reason}"

    logger.info("run_shell  | cwd=%s | cmd=%s", cwd, command)
    env = {
        **dict(os.environ),
        "GIT_AUTHOR_NAME": settings.git_author_name,
        "GIT_AUTHOR_EMAIL": settings.git_author_email,
    }

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=settings.shell_timeout_seconds,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        msg = f"TIMEOUT after {settings.shell_timeout_seconds}s: {command}"
        logger.error(msg)
        return msg
    except Exception as exc:
        msg = f"ERROR executing command: {exc}"
        logger.error(msg)
        return msg

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += ("\n--- stderr ---\n" if output else "") + result.stderr

    output = _truncate(output.strip())
    logger.info("run_shell  | exit=%d | output_len=%d", result.returncode, len(output))

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return f"[{status}]\n{output}" if output else f"[{status}]"


ALL_SHELL_TOOLS = [run_shell] if SHELL_ENABLED else []
