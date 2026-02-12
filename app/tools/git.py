"""Safe Git operations with strict allow/blocklist enforcement.

Git commands run inside the target repo root. Merge/rebase/reset are forbidden.
Commits require explicit planner approval (enforced at the orchestrator level).
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.git")

# Allowed git sub-commands.
ALLOWED_SUBCOMMANDS = {
    "status",
    "diff",
    "add",
    "commit",
    "checkout",
    "switch",
    "push",
    "pull",
    "fetch",
    "log",
    "branch",
    "show",
    "stash",
    "tag",
    "remote",
    "config",
    "rev-parse",
}

# Explicitly blocked patterns.
BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bgit\s+merge\b"),
    re.compile(r"\bgit\s+rebase\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-fd\b"),
    re.compile(r"\bgit\s+push\s+.*--force"),
    re.compile(r"\bgit\s+push\s+.*-f\b"),
]

# Block shell control operators explicitly.
BLOCKED_SHELL_TOKENS = {"&&", "||", "|", ";", "`"}


def _split_command(command: str) -> list[str]:
    """Split a git command string into argv."""
    posix = not platform.system().lower().startswith("win")
    return shlex.split(command, posix=posix)


def _validate_git_command(command: str) -> str | None:
    """Return error message if blocked, else None."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            return f"BLOCKED: forbidden git operation - {pattern.pattern}"

    try:
        parts = _split_command(command.strip())
    except ValueError as exc:
        return f"ERROR: invalid git command syntax: {exc}"

    if not parts or parts[0] != "git":
        return "ERROR: command must start with 'git'"
    if len(parts) < 2:
        return "ERROR: incomplete git command"
    if any(token in BLOCKED_SHELL_TOKENS for token in parts[1:]):
        return "BLOCKED: shell control operators are not allowed in git commands"

    sub = parts[1].lstrip("-")
    if sub not in ALLOWED_SUBCOMMANDS:
        allowed = ", ".join(sorted(ALLOWED_SUBCOMMANDS))
        return f"BLOCKED: git subcommand '{sub}' is not allowed. Permitted: {allowed}"

    return None


def _run_git(command: str) -> str:
    """Execute a validated git command and return output."""
    settings = get_settings()
    root = Path(settings.target_repo_path).resolve()

    if not root.is_dir():
        return f"ERROR: repo root does not exist: {root}"

    logger.info("git_tool   | %s", command)
    args = _split_command(command)

    try:
        result = subprocess.run(
            args,
            shell=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
            env={
                **dict(os.environ),
                "GIT_AUTHOR_NAME": settings.git_author_name,
                "GIT_AUTHOR_EMAIL": settings.git_author_email,
                "GIT_COMMITTER_NAME": settings.git_author_name,
                "GIT_COMMITTER_EMAIL": settings.git_author_email,
            },
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT: git command took > 60s"
    except Exception as exc:
        return f"ERROR: {exc}"

    output = ""
    if result.stdout:
        output += result.stdout.strip()
    if result.stderr:
        output += ("\n--- stderr ---\n" if output else "") + result.stderr.strip()

    limit = settings.max_output_chars
    if len(output) > limit:
        half = limit // 2
        output = output[:half] + "\n...[truncated]...\n" + output[-half:]

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    logger.info("git_tool   | exit=%d | output_len=%d", result.returncode, len(output))
    return f"[{status}]\n{output}" if output else f"[{status}]"


@tool
def git_command(command: str) -> str:
    """Run a git command inside the target repository.

    Allowed: status, diff, add, commit, checkout, switch, push, pull,
    fetch, log, branch, show, stash, tag, remote, config, rev-parse.
    Blocked: merge, rebase, reset --hard, clean -fd, push --force.
    """
    command = command.strip()
    if not command.startswith("git "):
        command = f"git {command}"

    error = _validate_git_command(command)
    if error:
        logger.warning("git_tool   | %s | %s", command, error)
        return error

    return _run_git(command)


@tool
def git_create_branch(branch_name: str) -> str:
    """Create and switch to a new feature branch."""
    if not branch_name.startswith("feature/"):
        branch_name = f"feature/{branch_name}"
    safe = re.sub(r"[^a-zA-Z0-9/_-]", "-", branch_name)
    return _run_git(f"git checkout -b {safe}")


@tool
def git_commit_and_push(message: str, push: bool = True) -> str:
    """Stage all changes, commit with a message, and optionally push.

    The commit message should follow Conventional Commits format.
    This tool should ONLY be called after planner approval + tests pass.
    """
    results: list[str] = []

    results.append(_run_git("git add -A"))

    status = _run_git("git status --porcelain")
    if "[OK]" in status and status.strip().endswith("[OK]"):
        return "Nothing to commit - working tree clean."

    safe_msg = message.replace('"', '\\"')
    results.append(_run_git(f'git commit -m "{safe_msg}"'))

    if push:
        branch_result = _run_git("git rev-parse --abbrev-ref HEAD")
        branch = branch_result.split("\n")[-1].strip() if "[OK]" in branch_result else "HEAD"
        results.append(_run_git(f"git push -u origin {branch}"))

    return "\n---\n".join(results)


@tool
def git_status() -> str:
    """Show git status and current branch."""
    branch = _run_git("git rev-parse --abbrev-ref HEAD")
    status = _run_git("git status --short")
    return f"Branch: {branch}\n{status}"


ALL_GIT_TOOLS = [git_command, git_create_branch, git_commit_and_push, git_status]
