"""Safe filesystem operations — sandboxed to TARGET_REPO_PATH.

Every path is resolved and verified to stay inside the repo root before any I/O.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.core.active_repo import get_repo_root
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("tools.filesystem")


class PathEscapeError(Exception):
    """Raised when a path would escape the repo sandbox."""


def _resolve_safe(relative_path: str) -> Path:
    """Resolve *relative_path* against repo root and verify it stays inside."""
    root = Path(get_repo_root()).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Target repo root does not exist: {root}")

    # Reject obvious escapes early
    if ".." in relative_path or relative_path.startswith("/"):
        # Still resolve to catch creative tricks, but warn
        pass

    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PathEscapeError(
            f"Path escapes repo sandbox: {relative_path!r} resolved to {target}"
        ) from exc
    return target


def _truncate(text: str, max_chars: int | None = None) -> str:
    limit = max_chars or get_settings().max_output_chars
    if len(text) > limit:
        return text[: limit // 2] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-limit // 2 :]
    return text


# ── LangChain Tools ──────────────────────────────────────────────────────


@tool
def read_file(path: str) -> str:
    """Read a file inside the target repository. `path` is relative to repo root."""
    target = _resolve_safe(path)
    if not target.is_file():
        return f"ERROR: Not a file: {path}"
    # Read raw bytes first so we can detect and strip any BOM before decoding.
    raw = target.read_bytes()
    if raw.startswith(b"\xff\xfe\x00\x00") or raw.startswith(b"\x00\x00\xfe\xff"):
        # UTF-32 BOM (LE or BE)
        content = raw.decode("utf-32", errors="replace").lstrip("\ufeff")
    elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        # UTF-16 BOM (LE or BE)
        content = raw.decode("utf-16", errors="replace").lstrip("\ufeff")
    elif raw.startswith(b"\xef\xbb\xbf"):
        # UTF-8 BOM — decode and strip
        content = raw[3:].decode("utf-8", errors="replace")
    else:
        content = raw.decode("utf-8", errors="replace")
    logger.info("read_file  | %s (%d chars)", path, len(content))
    return _truncate(content)


@tool
def write_file(path: str, content: str) -> str:
    """Write (create or overwrite) a file. `path` is relative to repo root."""
    target = _resolve_safe(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Strip any Unicode BOM the LLM may have included in the content string.
    content = content.lstrip("\ufeff")
    target.write_text(content, encoding="utf-8")
    logger.info("write_file | %s (%d chars)", path, len(content))
    return f"OK: wrote {len(content)} chars to {path}"


@tool
def patch_file(path: str, old: str, new: str) -> str:
    """Replace the first occurrence of `old` with `new` in a file (surgical edit)."""
    target = _resolve_safe(path)
    if not target.is_file():
        return f"ERROR: File not found: {path}"
    raw = target.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace").lstrip("\ufeff")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw[3:].decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    # Also strip BOM from search/replace strings if agent passes them
    old = old.lstrip("\ufeff")
    new = new.lstrip("\ufeff")
    if old not in text:
        return f"ERROR: Search string not found in {path}"
    updated = text.replace(old, new, 1)
    target.write_text(updated, encoding="utf-8")
    logger.info("patch_file | %s (replaced %d chars → %d chars)", path, len(old), len(new))
    return f"OK: patched {path}"


@tool
def list_directory(path: str = ".", max_depth: int = 2) -> str:
    """List files and directories. `path` is relative to repo root."""
    target = _resolve_safe(path)
    if not target.is_dir():
        return f"ERROR: Not a directory: {path}"

    lines: list[str] = []

    def _walk(p: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "node_modules" or entry.name == "__pycache__":
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                _walk(entry, depth + 1, prefix + "  ")
            else:
                lines.append(f"{prefix}{entry.name}")

    _walk(target, 0)
    result = "\n".join(lines) if lines else "(empty)"
    logger.info("list_dir   | %s (%d entries)", path, len(lines))
    return _truncate(result)


@tool
def delete_file(path: str) -> str:
    """Delete a single file inside the repo. Directories are not deletable."""
    target = _resolve_safe(path)
    if not target.is_file():
        return f"ERROR: Not a file or does not exist: {path}"
    target.unlink()
    logger.info("delete_file| %s", path)
    return f"OK: deleted {path}"


# Convenience: export all tools as a list
ALL_FS_TOOLS = [read_file, write_file, patch_file, list_directory, delete_file]
