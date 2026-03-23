"""Committer node — git commit, PR creation, and issue linking."""
from __future__ import annotations

import platform
import re
from contextlib import suppress
from pathlib import Path

from app.core.config import get_settings
from app.core.events import (
    emit,
    emit_error,
    emit_commit,
    emit_node_end,
    emit_node_start,
    emit_pr_created,
    emit_status,
)
from app.core.memory import get_memory_stats
from app.core.logging import get_logger
from app.core.state import GraphState, ItemStatus, WorkflowPhase
from app.tools.git import git_command, git_commit_and_push

from ._helpers import (
    _assign_coder_pair,
    _coder_label,
    _progress_meta,
    _reviewer_for_worker,
    _save_checkpoint_snapshot,
)

logger = get_logger("core.nodes.committer")

def _extract_commit_message(peer_notes: str, planner_notes: str, fallback_desc: str) -> str:
    for source in [planner_notes, peer_notes]:
        for line in source.split("\n"):
            stripped = line.strip()
            for prefix in ["Suggested commit:", "Commit message:", "Commit:", "Suggested Conventional Commit message:"]:
                if prefix.lower() in stripped.lower():
                    stripped = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
                    break
            stripped = stripped.strip("`").strip('"').strip("'").strip()
            if any(stripped.startswith(p) for p in ["feat(", "fix(", "docs:", "test:", "refactor(", "chore("]):
                return stripped
    return f"feat: {fallback_desc[:50].lower()}"


# ---------------------------------------------------------------------------
# PR/MR creation helper
# ---------------------------------------------------------------------------

def _build_pr_repo_path(repo_ref: str) -> tuple[str, str]:
    """Return ``(forge_url_for_detection, api_repo_path)`` from a repo_ref.

    * forge_url_for_detection — passed to ``get_forge_client()`` for platform
      auto-detection (e.g. ``https://github.com/owner/repo``).
    * api_repo_path — the path fragment used in API calls (``owner/repo``).
    """
    ref = (repo_ref or "").strip()
    if not ref:
        return "", ""

    if ref.startswith("http://") or ref.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(ref)
        path = parsed.path.strip("/")
        return ref, path

    parts = ref.split("/")
    if len(parts) >= 3:
        # host/owner/repo  →  forge_url = https://host/owner/repo
        forge_url = f"https://{ref}"
        api_path  = "/".join(parts[1:])
    else:
        # owner/repo  →  assume GitHub
        forge_url = f"https://github.com/{ref}"
        api_path  = ref
    return forge_url, api_path


def _create_pr_for_branch(state: GraphState) -> "PRResult | None":
    """Open a PR/MR for the current branch and return the result.

    Called at the end of ``committer_node`` when all items are done and
    ``settings.auto_create_pr`` is True.

    Returns ``None`` when:
    - ``auto_create_pr`` is False
    - ``repo_ref`` is not set (no forge info available)
    - Any forge API error (logged as warning, never raises)
    """
    from app.core.config import get_settings
    settings = get_settings()

    if not settings.auto_create_pr:
        logger.info("PR creation skipped: DAEDALUS_AUTO_CREATE_PR=false")
        return None

    repo_ref = state.repo_ref or ""
    if not repo_ref:
        logger.info("PR creation skipped: repo_ref not set")
        return None

    branch = state.branch_name
    if not branch:
        logger.info("PR creation skipped: branch_name not set")
        return None

    try:
        from infra.factory import get_forge_client
        from infra.forge import PRRequest

        forge_url, api_path = _build_pr_repo_path(repo_ref)
        if not forge_url or not api_path:
            logger.warning("PR creation skipped: cannot parse repo_ref %r", repo_ref)
            return None

        client = get_forge_client(forge_url)

        # Determine base branch
        base_branch = "main"
        try:
            from infra.workspace import _run_git
            from app.core.active_repo import get_repo_root
            from pathlib import Path as _Path
            _root = _Path(get_repo_root())
            if _root.is_dir():
                out = _run_git(
                    ["symbolic-ref", "refs/remotes/origin/HEAD", "--short"],
                    cwd=_root,
                )
                base_branch = out.split("/", 1)[-1].strip() or "main"
        except Exception:
            pass

        # Build title from first completed item or user request
        first_done = next(
            (i for i in state.todo_items if i.commit_message),
            None,
        )
        if first_done and first_done.commit_message:
            title = first_done.commit_message
        else:
            request_snippet = (state.user_request or "automated task")[:72]
            title = f"feat: {request_snippet}"

        # Build body
        task_lines = ["## Task\n", state.user_request or "(automated task)", ""]
        files_changed: list[str] = []
        try:
            raw = git_command.invoke({"command": "git diff --name-only HEAD~1 HEAD"})
            files_changed = [f.strip() for f in raw.splitlines() if f.strip()]
        except Exception:
            pass
        if files_changed:
            task_lines += ["## Files changed\n"]
            task_lines += [f"- `{f}`" for f in files_changed[:30]]
            if len(files_changed) > 30:
                task_lines.append(f"- …and {len(files_changed) - 30} more")
            task_lines.append("")

        # Link to issue if task was triggered by one
        if state.issue_ref:
            task_lines.append(f"Closes #{state.issue_ref.issue_id}")
            task_lines.append("")

        task_lines.append("---")
        task_lines.append("*Opened automatically by [Daedalus](https://github.com/simonabler/Daedalus)*")

        body = "\n".join(task_lines)

        pr_request = PRRequest(
            title=title,
            body=body,
            head_branch=branch,
            base_branch=base_branch,
        )

        pr = client.create_pr(api_path, pr_request)

        # Detect platform from forge URL for the result label
        platform = "gitlab" if "gitlab" in forge_url.lower() else "github"

        from app.core.state import PRResult
        result = PRResult(url=pr.url, number=pr.number, platform=platform)

        emit_pr_created(url=pr.url, number=pr.number, platform=platform, branch=branch)
        emit_status(
            "system",
            f"🔗 {('MR' if platform == 'gitlab' else 'PR')} #{pr.number} opened: {pr.url}",
            **_progress_meta(state, "complete"),
        )

        # If task was issue-triggered, reply to the issue with the PR link
        if state.issue_ref:
            _try_post_pr_link_on_issue(client, api_path, state.issue_ref.issue_id, pr.url, pr.number, platform)

        return result

    except Exception as exc:
        logger.warning("PR creation failed (continuing): %s", exc)
        emit_status(
            "system",
            f"⚠️ Could not create PR/MR: {exc}",
            **_progress_meta(state, "complete"),
        )
        return None


def _try_post_pr_link_on_issue(
    client: "Any",
    api_path: str,
    issue_id: int,
    pr_url: str,
    pr_number: int,
    platform: str,
) -> None:
    """Best-effort: comment on the issue with the PR/MR link."""
    label = "MR" if platform == "gitlab" else "PR"
    body = f"🔗 {label} #{pr_number} has been opened: {pr_url}"
    try:
        client.post_comment(api_path, issue_id, body)
        logger.info("Posted %s link on issue #%d", label, issue_id)
    except Exception as exc:
        logger.warning("Could not post %s link on issue #%d: %s", label, issue_id, exc)


# ---------------------------------------------------------------------------
# NODE: committer
# ---------------------------------------------------------------------------

def committer_node(state: GraphState) -> dict:
    """Commit, push, and advance to next item."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to commit")
        return {"error_message": "No item to commit", "phase": WorkflowPhase.STOPPED}

    emit_status("system", f"📦 Committing: {item.commit_message}", **_progress_meta(state, "committing"))

    result = git_commit_and_push.invoke({"message": item.commit_message, "push": True})
    emit_commit(item.commit_message, item_id=item.id)
    logger.info("commit result: %s", result[:200])
    _save_checkpoint_snapshot(state, {"phase": WorkflowPhase.COMMITTING}, "commit_success")

    next_index = state.current_item_index + 1
    has_more = next_index < len(state.todo_items)

    if has_more:
        next_item = state.todo_items[next_index]
        next_coder = next_item.assigned_agent or _assign_coder_pair(next_index)[0]
        next_reviewer = next_item.assigned_reviewer or _reviewer_for_worker(next_coder)
        emit_status(
            "planner",
            f"Moving to item {next_index + 1}/{len(state.todo_items)}: "
            f"{next_item.description} -> {_coder_label(next_coder)}",
            **_progress_meta(state, "coding"),
        )
        return {
            "current_item_index": next_index,
            "phase": WorkflowPhase.CODING,
            "active_coder": next_coder,
            "active_reviewer": next_reviewer,
            "peer_review_notes": "",
            "peer_review_verdict": "",
            "needs_human_approval": False,
            "pending_approval": {},
            "stop_reason": "",
        }
    else:
        # Log final memory stats
        stats = get_memory_stats()
        total = sum(s["chars"] for s in stats.values())
        emit_status(
            "planner",
            f"🧠 Final memory: {total} chars across {len(stats)} files",
            **_progress_meta(state, "complete"),
        )
        emit_status(
            "planner",
            f"🎉 All {len(state.todo_items)} items completed! Branch: {state.branch_name}",
            **_progress_meta(state, "complete"),
        )

        # Attempt to open a PR/MR automatically
        pr_result = _create_pr_for_branch(state)

        completion: dict = {
            "phase": WorkflowPhase.COMPLETE,
            "needs_human_approval": False,
            "pending_approval": {},
            "stop_reason": "",
        }
        if pr_result is not None:
            completion["pr_result"] = pr_result
        return completion

# ---------------------------------------------------------------------------
# Documenter Node
# ---------------------------------------------------------------------------

# Patterns in a git diff that indicate documentation should be updated.
# Matched against the raw diff text (additions + context lines).
_DOCS_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\+\s*def [a-z]", re.MULTILINE),         # new public function
    re.compile(r"^\+\s*async def [a-z]", re.MULTILINE),   # new public async function
    re.compile(r"^\+\s*class [A-Z]", re.MULTILINE),        # new public class
    re.compile(r"^\+.*@(app|router)\.(get|post|put|delete|patch|head)\b", re.MULTILINE),  # new API endpoint
    re.compile(r"^\+.*settings\.\w+\s*=", re.MULTILINE),  # new settings assignment
    re.compile(r"^\+\s*[A-Z_]{3,}\s*=\s*", re.MULTILINE), # new constant / env var
    re.compile(r"^\+.*argparse\|add_argument", re.MULTILINE),  # new CLI arg
]


