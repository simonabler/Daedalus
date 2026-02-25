"""Task routing helpers with history-aware Thompson Sampling.

The planner can route work to specialized agents based on task type.
Routing uses a lightweight multi-armed bandit per task type and persists
its history in the target repository.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.state import IssueRef

logger = get_logger("core.task_routing")

ROUTING_HISTORY_PATH = "memory/task-routing-history.json"


def classify_task_type(description: str) -> str:
    """Classify a work item into a coarse task type."""
    text = description.lower()

    doc_keywords = ("readme", "documentation", "docs", "changelog", "comment", "markdown", "md")
    test_keywords = ("test", "pytest", "coverage", "integration test", "unit test", "lint")
    ops_keywords = ("ci", "pipeline", "docker", "deploy", "infrastructure")

    if any(k in text for k in doc_keywords):
        return "documentation"
    if any(k in text for k in test_keywords):
        return "testing"
    if any(k in text for k in ops_keywords):
        return "ops"
    return "coding"


def is_programming_request(user_request: str) -> bool:
    """Heuristic to decide whether a request is primarily programming work."""
    text = user_request.lower()
    programming_markers = (
        "code",
        "implement",
        "fix",
        "bug",
        "feature",
        "api",
        "endpoint",
        "test",
        "refactor",
        "python",
        "javascript",
        "typescript",
        "backend",
        "frontend",
    )
    return any(marker in text for marker in programming_markers)


def load_routing_history(repo_root: str) -> dict:
    """Load routing history from repo, or return an empty default."""
    if not repo_root:
        return {"task_types": {}}

    path = Path(repo_root) / ROUTING_HISTORY_PATH
    if not path.exists():
        return {"task_types": {}}

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("Failed to read routing history from %s: %s", path, exc)

    return {"task_types": {}}


def save_routing_history(repo_root: str, history: dict) -> None:
    """Persist routing history into the repo."""
    if not repo_root:
        return

    path = Path(repo_root) / ROUTING_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")


def _agent_stats(history: dict, task_type: str, agent: str) -> dict:
    task_bucket = history.setdefault("task_types", {}).setdefault(task_type, {})
    return task_bucket.setdefault(agent, {"alpha": 1.0, "beta": 1.0, "trials": 0, "wins": 0})


def select_agent_thompson(
    repo_root: str,
    task_type: str,
    candidates: list[str],
) -> tuple[str, dict]:
    """Pick an agent for a task via Thompson Sampling."""
    if not candidates:
        raise ValueError("No candidate agents provided for routing")

    history = load_routing_history(repo_root)
    best_agent = candidates[0]
    best_score = -1.0

    for agent in candidates:
        stats = _agent_stats(history, task_type, agent)
        score = random.betavariate(float(stats["alpha"]), float(stats["beta"]))
        if score > best_score:
            best_score = score
            best_agent = agent

    return best_agent, history


def record_agent_outcome(repo_root: str, task_type: str, agent: str, success: bool) -> None:
    """Update Thompson Sampling posterior after an item outcome."""
    if not repo_root or not task_type or not agent:
        return

    history = load_routing_history(repo_root)
    stats = _agent_stats(history, task_type, agent)
    stats["trials"] = int(stats.get("trials", 0)) + 1
    if success:
        stats["alpha"] = float(stats.get("alpha", 1.0)) + 1.0
        stats["wins"] = int(stats.get("wins", 0)) + 1
    else:
        stats["beta"] = float(stats.get("beta", 1.0)) + 1.0

    save_routing_history(repo_root, history)


def history_summary(repo_root: str) -> str:
    """Compact text summary to inject into planner context."""
    history = load_routing_history(repo_root)
    task_types = history.get("task_types", {})
    if not task_types:
        return "No routing history yet."

    lines: list[str] = []
    for task_type, agents in task_types.items():
        if not isinstance(agents, dict):
            continue
        sorted_agents = sorted(
            agents.items(),
            key=lambda kv: (
                (kv[1].get("wins", 0) / kv[1].get("trials", 1)) if kv[1].get("trials", 0) else 0.0
            ),
            reverse=True,
        )
        snippets = []
        for agent, stats in sorted_agents[:3]:
            trials = int(stats.get("trials", 0))
            wins = int(stats.get("wins", 0))
            rate = (wins / trials * 100.0) if trials else 0.0
            snippets.append(f"{agent}: {wins}/{trials} ({rate:.0f}%)")
        lines.append(f"- {task_type}: " + ", ".join(snippets))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Issue reference parsing
# ---------------------------------------------------------------------------

# Matches GitHub/GitLab issue URLs:
#   https://github.com/owner/repo/issues/42
#   https://gitlab.com/group/project/-/issues/42
#   https://gitlab.internal/team/proj/issues/42
_ISSUE_URL_RE = re.compile(
    r"https?://(?P<host>[^\s/]+)/(?P<path>[^\s]+?)/-?/?issues/(?P<id>\d+)",
    re.IGNORECASE,
)

# Matches #42 or issue #42 or issue 42 when a repo ref can be found alongside it
_HASH_ISSUE_RE = re.compile(r"(?:issue\s+)?#(?P<id>\d+)\b", re.IGNORECASE)

# Matches "issue 42 in owner/repo" or "issue 42 for owner/repo"
_BARE_ISSUE_RE = re.compile(
    r"\bissue\s+(?P<id>\d+)\s+(?:in|for|on|at)\s+(?P<repo>[A-Za-z0-9_./-]+)",
    re.IGNORECASE,
)


def _platform_from_host(host: str) -> str:
    """Guess platform string from a forge hostname."""
    h = host.lower()
    if "github" in h:
        return "github"
    if "gitlab" in h:
        return "gitlab"
    return ""


def parse_issue_ref(user_request: str, fallback_repo_ref: str = "") -> "IssueRef | None":
    """Detect and parse a forge issue reference from free-form user text.

    Detection strategies (first match wins):

    1. Full issue URL:
       ``https://github.com/owner/repo/issues/42``
       ``https://gitlab.com/group/proj/-/issues/7``

    2. Bare issue + inline repo:
       ``fix issue 42 in owner/repo``

    3. ``#N`` shorthand when a *fallback_repo_ref* is available:
       ``fix #42`` (with repo_ref already set to "owner/repo")

    Args:
        user_request:     Raw text from the user.
        fallback_repo_ref: repo_ref already extracted by _extract_repo_ref
                           (used for #N shorthand that has no embedded URL).

    Returns:
        :class:`~app.core.state.IssueRef` or ``None`` if nothing detected.
    """
    from app.core.state import IssueRef  # local to avoid circular import at module load

    # ── Strategy 1: full issue URL ────────────────────────────────────
    m = _ISSUE_URL_RE.search(user_request)
    if m:
        host = m.group("host")
        raw_path = m.group("path").rstrip("/-")
        # path is like "owner/repo" or "group/sub/project"
        # reconstruct a clean repo_ref = host/path
        repo_ref = f"{host}/{raw_path}"
        return IssueRef(
            repo_ref=repo_ref,
            issue_id=int(m.group("id")),
            platform=_platform_from_host(host),
        )

    # ── Strategy 2: "issue 42 in owner/repo" ─────────────────────────
    m2 = _BARE_ISSUE_RE.search(user_request)
    if m2:
        return IssueRef(
            repo_ref=m2.group("repo").rstrip(".,;"),
            issue_id=int(m2.group("id")),
            platform="",
        )

    # ── Strategy 3: #N shorthand with known repo_ref ─────────────────
    m3 = _HASH_ISSUE_RE.search(user_request)
    if m3 and fallback_repo_ref:
        return IssueRef(
            repo_ref=fallback_repo_ref,
            issue_id=int(m3.group("id")),
            platform="",
        )

    return None
