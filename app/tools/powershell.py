"""Safe PowerShell execution with sandbox and blocklist enforcement."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.powershell")

IS_WINDOWS = platform.system().lower().startswith("win")
POWERSHELL_ENABLED = IS_WINDOWS

BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\brestart-computer\b", re.IGNORECASE),
    re.compile(r"\bstop-computer\b", re.IGNORECASE),
    re.compile(r"\bformat-volume\b", re.IGNORECASE),
    re.compile(r"\bclear-disk\b", re.IGNORECASE),
    re.compile(r"\binitialize-disk\b", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bbcdedit\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b.*-recurse.*-force", re.IGNORECASE),
    re.compile(r"\bcurl\s+.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\bwget\s+.*\|\s*(ba)?sh", re.IGNORECASE),
    re.compile(r"\b(iwr|invoke-webrequest)\b.*\|\s*(iex|invoke-expression)", re.IGNORECASE),
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


def _resolve_ps_executable() -> str | None:
    """Find an available PowerShell executable."""
    return shutil.which("pwsh") or shutil.which("powershell")


@tool
def run_powershell(command: str, working_dir: str = ".") -> str:
    """Execute a PowerShell command inside the target repository.

    This tool is only enabled on Windows systems.
    """
    if not POWERSHELL_ENABLED:
        return "DISABLED: run_powershell is only enabled on Windows systems."

    settings = get_settings()
    root = Path(settings.target_repo_path).resolve()
    cwd = (root / working_dir).resolve()

    if not _is_within_root(cwd, root):
        msg = f"BLOCKED: working_dir escapes repo root: {working_dir}"
        logger.warning(msg)
        return msg

    if not cwd.is_dir():
        return f"ERROR: directory does not exist: {working_dir}"

    reason = _is_blocked(command)
    if reason:
        logger.warning("BLOCKED powershell | %s | reason: %s", command, reason)
        return f"BLOCKED: {reason}"

    ps_exe = _resolve_ps_executable()
    if not ps_exe:
        return "ERROR: no PowerShell executable found (pwsh/powershell)."

    logger.info("run_powershell | cwd=%s | cmd=%s", cwd, command)
    env = {
        **dict(os.environ),
        "GIT_AUTHOR_NAME": settings.git_author_name,
        "GIT_AUTHOR_EMAIL": settings.git_author_email,
    }

    try:
        result = subprocess.run(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", command],
            shell=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=settings.shell_timeout_seconds,
            env=env,
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
    logger.info("run_powershell | exit=%d | output_len=%d", result.returncode, len(output))

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return f"[{status}]\n{output}" if output else f"[{status}]"


ALL_POWERSHELL_TOOLS = [run_powershell] if POWERSHELL_ENABLED else []
