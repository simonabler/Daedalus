"""Safe repository text search tool."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.search")

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}

SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".pdf",
    ".zip",
}


def _resolve_repo_root(repo_path: str | None = None) -> Path:
    settings = get_settings()
    root = Path(repo_path or settings.target_repo_path).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    return root


@tool
def search_in_repo(
    pattern: str,
    file_pattern: str = "*",
    max_hits: int = 50,
    case_sensitive: bool = False,
    repo_path: str = "",
) -> str:
    """Search text across repository files without shell execution.

    Returns lines in format: `path:line_number: matched_line`.
    """
    if not pattern.strip():
        return "Error: pattern must not be empty."

    try:
        root = _resolve_repo_root(repo_path)
    except FileNotFoundError as exc:
        return f"Error: {exc}"

    max_hits = max(1, min(int(max_hits), 200))
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: Invalid regex pattern: {exc}"

    results: list[str] = []
    files_searched = 0
    hits_found = 0

    for file_path in root.rglob(file_pattern):
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        if file_path.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_size > 1_000_000:
                continue
        except OSError:
            continue

        files_searched += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_num, line in enumerate(content.splitlines(), start=1):
            if not regex.search(line):
                continue
            rel_path = file_path.relative_to(root)
            results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
            hits_found += 1
            if hits_found >= max_hits:
                results.append(f"\n... (max hits {max_hits} reached, {files_searched} files searched)")
                logger.info("search_in_repo | pattern=%r | hits=%d", pattern, hits_found)
                return "\n".join(results)

    if not results:
        return f"No matches found for '{pattern}' in {files_searched} files."

    results.append(f"\n({hits_found} matches in {files_searched} files)")
    logger.info("search_in_repo | pattern=%r | hits=%d", pattern, hits_found)
    return "\n".join(results)


ALL_SEARCH_TOOLS = [search_in_repo]
