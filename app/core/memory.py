"""Shared long-term memory for coders.

Three memory files live in the target repo under `memory/`:
  - coding-style.md       — naming, patterns, error handling, import order, etc.
  - architecture-decisions.md — ADRs (why we chose X over Y)
  - shared-insights.md    — codebase quirks, gotchas, undocumented behaviors

Both coders read ALL memory before each task. After each peer review
a learning step extracts new insights and appends them.

Periodically the planner compresses memory files to stay within token
budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("core.memory")

# ── Memory file names (relative to repo root) ────────────────────────────

MEMORY_DIR = "memory"

MEMORY_FILES = {
    "coding_style": f"{MEMORY_DIR}/coding-style.md",
    "architecture": f"{MEMORY_DIR}/architecture-decisions.md",
    "insights": f"{MEMORY_DIR}/shared-insights.md",
}

# Maximum chars per memory file before compression is triggered
MAX_MEMORY_CHARS = 8_000

# ── Seed content for fresh memory files ───────────────────────────────────

_SEEDS = {
    "coding_style": """\
# Coding Style Guide
> Shared by Coder A (Claude) and Coder B (GPT-5.3).
> Both coders read this before every task and add entries after reviews.

## Rules
<!-- Add rules as: - **Rule**: description (learned from item-XXX) -->
""",
    "architecture": """\
# Architecture Decisions
> Shared ADR log. Records WHY decisions were made, not just WHAT.

## Decisions
<!-- Add as: ### ADR-N: Title\\nContext: ...\\nDecision: ...\\nConsequences: ... -->
""",
    "insights": """\
# Shared Insights
> Codebase knowledge that isn't obvious from reading the code.
> Gotchas, quirks, undocumented behaviors, useful helpers, test tips.

## Entries
<!-- Add as: - **Topic**: insight (discovered during item-XXX) -->
""",
}


# ── Core API ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    settings = get_settings()
    return Path(settings.target_repo_path)


def _memory_path(key: str) -> Path:
    return _repo_root() / MEMORY_FILES[key]


def ensure_memory_files() -> None:
    """Create memory directory and seed files if they don't exist."""
    mem_dir = _repo_root() / MEMORY_DIR
    mem_dir.mkdir(parents=True, exist_ok=True)

    for key, seed in _SEEDS.items():
        path = _memory_path(key)
        if not path.exists():
            path.write_text(seed, encoding="utf-8")
            logger.info("Created memory file: %s", path)


def load_memory(key: str) -> str:
    """Load a single memory file. Returns empty string if missing."""
    path = _memory_path(key)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_all_memory() -> str:
    """Load all memory files into a single context block for LLM injection."""
    sections = []
    for key in MEMORY_FILES:
        content = load_memory(key)
        if content.strip():
            sections.append(content)
    if not sections:
        return ""
    return (
        "# ═══ SHARED LONG-TERM MEMORY ═══\n"
        "The following is shared knowledge built up by both coders across sessions.\n"
        "Use this to stay consistent with established patterns and decisions.\n\n"
        + "\n\n---\n\n".join(sections)
    )


def append_memory(key: str, entry: str, item_id: str = "") -> None:
    """Append a new entry to a memory file."""
    path = _memory_path(key)
    ensure_memory_files()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    source_tag = f" (from {item_id})" if item_id else ""
    formatted = f"\n- [{timestamp}{source_tag}] {entry}"

    current = path.read_text(encoding="utf-8")
    path.write_text(current + formatted, encoding="utf-8")
    logger.info("Memory appended [%s]: %s", key, entry[:80])


def memory_needs_compression(key: str) -> bool:
    """Check if a memory file has grown beyond the compression threshold."""
    content = load_memory(key)
    return len(content) > MAX_MEMORY_CHARS


def get_memory_stats() -> dict[str, dict]:
    """Return size stats for each memory file."""
    stats = {}
    for key in MEMORY_FILES:
        content = load_memory(key)
        stats[key] = {
            "chars": len(content),
            "lines": content.count("\n"),
            "needs_compression": len(content) > MAX_MEMORY_CHARS,
        }
    return stats


# ── Compression prompt builder ────────────────────────────────────────────

def build_compression_prompt(key: str) -> str | None:
    """Build a prompt for the planner to compress a memory file.

    Returns None if compression is not needed.
    """
    content = load_memory(key)
    if len(content) <= MAX_MEMORY_CHARS:
        return None

    friendly_name = {
        "coding_style": "Coding Style Guide",
        "architecture": "Architecture Decisions",
        "insights": "Shared Insights",
    }.get(key, key)

    return (
        f"## Memory Compression Task\n\n"
        f"The **{friendly_name}** memory file has grown to {len(content)} characters "
        f"(limit: {MAX_MEMORY_CHARS}).\n\n"
        f"Current content:\n```\n{content}\n```\n\n"
        f"Compress this file:\n"
        f"1. Remove duplicate or redundant entries.\n"
        f"2. Merge related entries into concise rules.\n"
        f"3. Remove entries that are too project-specific to be useful long-term.\n"
        f"4. Keep the most important, actionable rules.\n"
        f"5. Preserve the file header and format.\n"
        f"6. Target: under {MAX_MEMORY_CHARS // 2} characters.\n\n"
        f"Return ONLY the compressed file content, nothing else."
    )


def save_compressed(key: str, content: str) -> None:
    """Overwrite a memory file with compressed content."""
    path = _memory_path(key)
    path.write_text(content, encoding="utf-8")
    logger.info("Memory compressed [%s]: %d → %d chars", key, len(load_memory(key)), len(content))


# ── Learning extraction prompt builder ────────────────────────────────────

LEARNING_EXTRACTION_PROMPT = """\
You are the **Learning Extractor** for a dual-coder AI system.

After each peer review, you analyze the review to extract reusable knowledge
that BOTH coders should remember for future tasks.

## Input
You will receive:
- The review text (from the peer reviewer)
- The item description
- The item ID

## What to Extract
Look for these categories of learnable knowledge:

### Coding Style (→ memory/coding-style.md)
- Naming conventions discovered or enforced
- Error handling patterns preferred in this project
- Import ordering or module structure rules
- Testing patterns (how tests are organized, what's tested)
- Code formatting preferences

### Architecture Decisions (→ memory/architecture-decisions.md)
- Patterns chosen (Repository, Factory, Strategy, etc.)
- Library choices and why
- Module boundaries and dependency rules
- API design conventions

### Shared Insights (→ memory/shared-insights.md)
- Undocumented codebase behaviors
- Useful existing helpers or utilities discovered
- Test environment quirks
- Build/deploy gotchas
- Common pitfalls in this specific codebase

## Output Format
Respond with a JSON object. Use empty arrays for categories with no learnings.
```json
{
  "coding_style": ["rule 1", "rule 2"],
  "architecture": ["decision 1"],
  "insights": ["insight 1", "insight 2"]
}
```

If the review contains NO useful generalizable learnings (e.g. it's purely
about a one-off bug fix), return:
```json
{"coding_style": [], "architecture": [], "insights": []}
```

Be selective — only extract things that will be useful for FUTURE tasks,
not things specific to this one fix. Quality over quantity.
"""
