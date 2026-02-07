"""Safe shell execution — runs only inside repo root with blocklist enforcement.

Every command is logged with cwd, exit code, and truncated output.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.shell")

# ── Blocklist ────────────────────────────────────────────────────────────
# Patterns that must NEVER be executed, regardless of context.
BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+(-[rRf]+\s+)*/((?!home)|$)", re.IGNORECASE),  # rm -rf /
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*:\|:.*\}"),  # fork bomb
    re.compile(r"\bchmod\s+(-R\s+)?777\s+/"),  # chmod 777 /
    re.compile(r"\bcurl\s+.*\|\s*(ba)?sh", re.IGNORECASE),  # curl | sh
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

# Allowed command prefixes for common dev tasks
ALLOWED_PREFIXES = {
    "python", "python3", "pip", "pip3", "uv",
    "node", "npm", "npx", "pnpm", "yarn", "tsc",
    "dotnet",
    "pytest", "ruff", "mypy", "black", "isort",
    "git",  # git is handled separately by the git tool but allowed here for status/diff
    "docker", "docker-compose",
    "cat", "head", "tail", "wc", "grep", "find", "ls", "echo", "pwd", "env",
    "mkdir", "cp", "mv", "touch", "rm",
    "cargo", "rustc",
    "make", "cmake",
    "curl", "wget",  # allowed standalone (blocked only in pipe-to-sh)
    "tar", "unzip", "zip",
    "sed", "awk", "sort", "uniq", "diff", "patch",
    "sh", "bash",
}


def _is_blocked(command: str) -> str | None:
    """Return a reason string if the command is blocked, else None."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"Blocked by safety rule: {pattern.pattern}"
    return None


def _truncate(text: str) -> str:
    limit = get_settings().max_output_chars
    if len(text) > limit:
        half = limit // 2
        return text[:half] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-half:]
    return text


@tool
def run_shell(command: str, working_dir: str = ".") -> str:
    """Execute a shell command inside the target repository.

    `working_dir` is relative to the repo root (default ".").
    Output (stdout + stderr) is returned, truncated if too large.
    Dangerous commands are blocked.
    """
    settings = get_settings()
    root = Path(settings.target_repo_path).resolve()
    cwd = (root / working_dir).resolve()

    # Sandbox check
    if not str(cwd).startswith(str(root)):
        msg = f"BLOCKED: working_dir escapes repo root: {working_dir}"
        logger.warning(msg)
        return msg

    if not cwd.is_dir():
        return f"ERROR: directory does not exist: {working_dir}"

    # Blocklist check
    reason = _is_blocked(command)
    if reason:
        logger.warning("BLOCKED shell | %s | reason: %s", command, reason)
        return f"BLOCKED: {reason}"

    logger.info("run_shell  | cwd=%s | cmd=%s", cwd, command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=settings.shell_timeout_seconds,
            env={
                **dict(__import__("os").environ),
                "GIT_AUTHOR_NAME": settings.git_author_name,
                "GIT_AUTHOR_EMAIL": settings.git_author_email,
            },
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
    logger.info(
        "run_shell  | exit=%d | output_len=%d", result.returncode, len(output)
    )

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return f"[{status}]\n{output}" if output else f"[{status}]"


ALL_SHELL_TOOLS = [run_shell]
