"""Persistent cache for code intelligence results, keyed by git commit hash.

Cache files live in ``<repo>/.daedalus/intelligence_cache/``.
Each entry is a JSON file named ``<commit_hash>.json`` containing the
serialised output of all four analysis tools.

Public API
----------
``get_commit_hash(repo_path)``    → str | None
``load_cache(repo_path, key)``    → dict | None
``save_cache(repo_path, key, data)``
``cache_path(repo_path, key)``    → Path
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger("analysis.intelligence_cache")

_CACHE_DIR = ".daedalus/intelligence_cache"


def get_commit_hash(repo_path: str | Path) -> str | None:
    """Return the current HEAD commit hash for *repo_path*, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        logger.debug("intelligence_cache: could not get commit hash: %s", exc)
    return None


def cache_path(repo_path: str | Path, key: str) -> Path:
    return Path(repo_path) / _CACHE_DIR / f"{key}.json"


def load_cache(repo_path: str | Path, key: str) -> dict | None:
    """Return cached intelligence data for *key*, or None if not found / stale."""
    if not key:
        return None
    path = cache_path(repo_path, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("intelligence_cache: cache HIT for key=%s", key)
        return data
    except Exception as exc:
        logger.warning("intelligence_cache: failed to load cache %s: %s", path, exc)
        return None


def save_cache(repo_path: str | Path, key: str, data: dict) -> None:
    """Persist *data* under *key* in the repo's intelligence cache directory."""
    if not key:
        return
    path = cache_path(repo_path, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("intelligence_cache: saved cache for key=%s (%d bytes)", key, path.stat().st_size)
    except Exception as exc:
        logger.warning("intelligence_cache: failed to save cache %s: %s", path, exc)
