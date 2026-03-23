"""Documenter node — post-commit documentation updates."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import load_system_prompt
from app.core.events import emit_node_end, emit_node_start, emit_status
from app.core.logging import get_logger
from app.core.state import GraphState, WorkflowPhase
from app.core.token_budget import BudgetExceededException
from app.tools.git import git_command

from ._helpers import DOCUMENTER_TOOLS, _invoke_with_budget, _progress_meta

logger = get_logger("core.nodes.documenter")

_DOCS_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\+\s*def [a-z]", re.MULTILINE),         # new public function
    re.compile(r"^\+\s*async def [a-z]", re.MULTILINE),   # new public async function
    re.compile(r"^\+\s*class [A-Z]", re.MULTILINE),        # new public class
    re.compile(r"^\+.*@(app|router)\.(get|post|put|delete|patch|head)\b", re.MULTILINE),  # new API endpoint
    re.compile(r"^\+.*settings\.\w+\s*=", re.MULTILINE),  # new settings assignment
    re.compile(r"^\+\s*[A-Z_]{3,}\s*=\s*", re.MULTILINE), # new constant / env var
    re.compile(r"^\+.*argparse\|add_argument", re.MULTILINE),  # new CLI arg
]

def _diff_needs_docs(diff: str) -> bool:
    """Return True if the git diff contains documentation-worthy changes.

    Uses lightweight regex heuristics so no LLM call is made for trivial
    commits (test-only changes, typo fixes, pure refactors).
    """
    return any(pat.search(diff) for pat in _DOCS_TRIGGER_PATTERNS)


def documenter_node(state: GraphState) -> dict:
    """Update project documentation after a successful commit.

    Runs after every commit. Uses a heuristic diff scan to decide whether
    an LLM call is warranted. If the diff contains no documentation-worthy
    changes the node exits immediately without an LLM call.

    The documenter writes documentation changes to disk but does NOT commit
    them — they will be picked up by the next commit cycle or can be
    committed manually.
    """
    emit_node_start("documenter", "Documenting", item_desc="Checking diff for documentation updates")
    emit_status(
        "documenter",
        "📝 Documenter: scanning commit diff…",
        **_progress_meta(state, "documenting"),
    )

    # -- 1. Get diff of the last commit -----------------------------------
    try:
        diff = git_command.invoke({"command": "git diff HEAD~1 HEAD"})
    except Exception as exc:
        logger.warning("documenter | git diff failed: %s", exc)
        diff = ""

    if not diff:
        emit_node_end("documenter", "Documenting", "No diff available — skipping documentation update")
        return {}

    # -- 2. Heuristic gate — skip LLM if diff is not doc-worthy ----------
    if not _diff_needs_docs(diff):
        emit_status(
            "documenter",
            "📝 Documenter: no documentation-worthy changes detected — skipping",
            **_progress_meta(state, "documenting"),
        )
        emit_node_end("documenter", "Documenting", "Skipped — diff contains no public API or config changes")
        return {}

    # -- 3. LLM call with full diff context ------------------------------
    emit_status(
        "documenter",
        "📝 Documenter: documentation-worthy changes detected — updating docs…",
        **_progress_meta(state, "documenting"),
    )

    prompt = (
        "## Documentation Task\n\n"
        "A commit was just made. Review the diff below and update the project documentation "
        "following your system instructions.\n\n"
        f"### Git Diff (last commit)\n\n```diff\n{diff[:8000]}\n```\n\n"
        "Start by reading any existing CHANGELOG.md and README.md with the available tools, "
        "then make the necessary updates. Output your structured summary when done."
    )

    try:
        result, budget_update = _invoke_with_budget(
            state, "documenter", [HumanMessage(content=prompt)],
            DOCUMENTER_TOOLS, node="documenter",
        )
    except BudgetExceededException:
        emit_node_end("documenter", "Documenting", "Budget limit exceeded — skipping documentation")
        return {}

    emit_node_end("documenter", "Documenting", result[:400] if result else "Documentation updated")
    return {**budget_update}
