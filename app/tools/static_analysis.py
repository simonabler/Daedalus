"""Static analysis integration for Python (ruff, mypy) and JS/TS (eslint).

Runs the detected linter/type-checker for a repository and returns a
structured list of ``StaticIssue`` objects.  Every analysis step is
best-effort: if a tool is not installed or exits with an unexpected error the
step is skipped and a warning is logged — the workflow is never blocked.

Provider routing mirrors the existing CodebaseAnalyzer language detection:
- Python repos  → ruff (lint) + mypy (types)
- JS/TS repos   → eslint
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("tools.static_analysis")

Severity = Literal["error", "warning", "info"]

# How long (seconds) a single analysis tool may run before we give up.
_TOOL_TIMEOUT = 60


class StaticIssue(BaseModel):
    """A single issue reported by a static analysis tool."""

    file: str
    line: int = 0
    col: int = 0
    severity: Severity = "warning"
    rule_id: str = ""
    message: str
    tool: str  # "ruff" | "mypy" | "eslint"

    def one_line(self) -> str:
        loc = f"{self.file}:{self.line}"
        if self.col:
            loc += f":{self.col}"
        rule = f" [{self.rule_id}]" if self.rule_id else ""
        return f"[{self.severity.upper()}]{rule} {loc} — {self.message}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_static_analysis(repo_path: str | Path, language: str) -> list[StaticIssue]:
    """Run all applicable static analysis tools for *language* in *repo_path*.

    Returns a list of ``StaticIssue`` objects sorted by severity (errors
    first) then by file/line.  Returns an empty list — never raises — when no
    tools are available or all tools fail.
    """
    root = Path(repo_path).resolve()
    lang = (language or "").lower()

    issues: list[StaticIssue] = []

    if lang == "python":
        issues.extend(_run_ruff(root))
        issues.extend(_run_mypy(root))
    elif lang in {"javascript", "typescript"}:
        issues.extend(_run_eslint(root))
        if lang == "typescript":
            issues.extend(_run_tsc(root))
    else:
        logger.debug("static_analysis: unsupported language '%s', skipping", lang)

    return _sort_issues(issues)


def format_issues_for_prompt(issues: list[StaticIssue], max_issues: int = 20) -> str:
    """Return a compact, prompt-ready summary of *issues*.

    Errors are shown first (up to *max_issues* total).
    """
    if not issues:
        return "No static analysis issues detected."

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    selected: list[StaticIssue] = []
    selected.extend(errors)
    selected.extend(warnings)
    selected.extend(infos)
    selected = selected[:max_issues]

    total = len(issues)
    shown = len(selected)
    header = (
        f"## Static Analysis — {len(errors)} error(s), {len(warnings)} warning(s)"
        f", {len(infos)} info(s)"
    )
    if total > shown:
        header += f" (showing top {shown} of {total})"

    lines = [header, ""]
    for issue in selected:
        lines.append(f"- {issue.one_line()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ruff
# ---------------------------------------------------------------------------

def _run_ruff(root: Path) -> list[StaticIssue]:
    """Run ``ruff check --output-format=json`` and parse its output."""
    if not _tool_available("ruff"):
        logger.debug("ruff not found, skipping")
        return []

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", ".", "--output-format=json",
             "--no-cache"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.debug("ruff module not available via python -m ruff")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("ruff timed out after %ds", _TOOL_TIMEOUT)
        return []
    except Exception as exc:
        logger.warning("ruff failed unexpectedly: %s", exc)
        return []

    # ruff exits 1 when issues are found — that is expected, not an error.
    raw = (proc.stdout or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse ruff JSON output: %s", exc)
        return []

    issues: list[StaticIssue] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        location = item.get("location") or {}
        end_location = item.get("end_location") or {}
        severity = _ruff_severity(item.get("code", ""))
        issues.append(StaticIssue(
            file=_rel(root, item.get("filename", "")),
            line=location.get("row", 0),
            col=location.get("column", 0),
            severity=severity,
            rule_id=item.get("code", ""),
            message=item.get("message", ""),
            tool="ruff",
        ))

    logger.info("ruff: %d issues found", len(issues))
    return issues


def _ruff_severity(code: str) -> Severity:
    """Map ruff rule codes to our 3-level severity."""
    if not code:
        return "warning"
    prefix = code[0].upper()
    # E = pycodestyle errors, F = pyflakes, N = naming → treat as errors
    if prefix in {"E", "F"}:
        return "error"
    # W = warnings
    if prefix == "W":
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# mypy
# ---------------------------------------------------------------------------

def _run_mypy(root: Path) -> list[StaticIssue]:
    """Run ``mypy`` and parse its line-by-line output."""
    if not _tool_available("mypy"):
        logger.debug("mypy not found, skipping")
        return []

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", ".", "--no-error-summary",
             "--show-column-numbers", "--show-error-codes",
             "--ignore-missing-imports"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.debug("mypy module not available")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("mypy timed out after %ds", _TOOL_TIMEOUT)
        return []
    except Exception as exc:
        logger.warning("mypy failed unexpectedly: %s", exc)
        return []

    issues: list[StaticIssue] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        parsed = _parse_mypy_line(root, line)
        if parsed:
            issues.append(parsed)

    logger.info("mypy: %d issues found", len(issues))
    return issues


def _parse_mypy_line(root: Path, line: str) -> StaticIssue | None:
    """Parse a single mypy output line.

    Format: ``path/to/file.py:10:5: error: message  [error-code]``
    """
    # Minimum: "file:line: severity: message"
    parts = line.split(":", 3)
    if len(parts) < 4:
        return None

    file_part = parts[0].strip()
    if not file_part or file_part.startswith(" "):
        return None

    # parts[1] might be line number or (on Windows) a drive letter
    try:
        line_no = int(parts[1])
    except ValueError:
        return None

    rest = parts[2] + ":" + parts[3] if len(parts) > 3 else parts[2]
    # rest looks like " col: severity: message [code]"
    col = 0
    rest = rest.strip()
    subparts = rest.split(":", 2)

    # Try to extract column
    severity_raw = subparts[0].strip()
    message = ""
    if len(subparts) >= 2:
        # Could be "col: severity: message"
        try:
            col = int(severity_raw)
            severity_raw = subparts[1].strip()
            message = subparts[2].strip() if len(subparts) > 2 else ""
        except ValueError:
            message = ": ".join(subparts[1:]).strip()

    sev_map = {"error": "error", "warning": "warning", "note": "info"}
    severity = sev_map.get(severity_raw.lower(), "info")
    if severity == "info":
        return None  # skip mypy notes

    # Extract rule code from trailing "[code]"
    rule_id = ""
    if message.endswith("]") and "[" in message:
        bracket_start = message.rfind("[")
        rule_id = message[bracket_start + 1:-1]
        message = message[:bracket_start].strip()

    return StaticIssue(
        file=_rel(root, file_part),
        line=line_no,
        col=col,
        severity=severity,
        rule_id=rule_id,
        message=message,
        tool="mypy",
    )


# ---------------------------------------------------------------------------
# eslint
# ---------------------------------------------------------------------------

def _run_eslint(root: Path) -> list[StaticIssue]:
    """Run ``eslint --format=json`` and parse its output."""
    eslint_bin = _find_eslint(root)
    if not eslint_bin:
        logger.debug("eslint not found, skipping")
        return []

    try:
        proc = subprocess.run(
            [eslint_bin, ".", "--format=json", "--max-warnings=-1"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.debug("eslint binary not executable: %s", eslint_bin)
        return []
    except subprocess.TimeoutExpired:
        logger.warning("eslint timed out after %ds", _TOOL_TIMEOUT)
        return []
    except Exception as exc:
        logger.warning("eslint failed unexpectedly: %s", exc)
        return []

    raw = (proc.stdout or "").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse eslint JSON output: %s", exc)
        return []

    issues: list[StaticIssue] = []
    for file_result in data:
        if not isinstance(file_result, dict):
            continue
        file_path = _rel(root, file_result.get("filePath", ""))
        for msg in file_result.get("messages", []):
            if not isinstance(msg, dict):
                continue
            sev_int = msg.get("severity", 1)
            severity: Severity = "error" if sev_int == 2 else "warning"
            issues.append(StaticIssue(
                file=file_path,
                line=msg.get("line", 0),
                col=msg.get("column", 0),
                severity=severity,
                rule_id=msg.get("ruleId") or "",
                message=msg.get("message", ""),
                tool="eslint",
            ))

    logger.info("eslint: %d issues found", len(issues))
    return issues


def _find_eslint(root: Path) -> str | None:
    """Locate the eslint binary: local node_modules first, then PATH."""
    local = root / "node_modules" / ".bin" / "eslint"
    if local.exists():
        return str(local)
    import shutil
    return shutil.which("eslint")


# ---------------------------------------------------------------------------
# tsc (TypeScript compiler — type checking only)
# ---------------------------------------------------------------------------

def _run_tsc(root: Path) -> list[StaticIssue]:
    """Run ``tsc --noEmit`` and parse its diagnostic output.

    Only runs when a ``tsconfig.json`` is present in *root*.  Requires
    ``typescript`` to be installed as a local dev dependency or globally.
    """
    tsconfig = root / "tsconfig.json"
    if not tsconfig.exists():
        logger.debug("tsc: no tsconfig.json found, skipping")
        return []

    tsc_bin = _find_tsc(root)
    if not tsc_bin:
        logger.debug("tsc: binary not found, skipping")
        return []

    try:
        proc = subprocess.run(
            [tsc_bin, "--noEmit", "--pretty", "false"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.debug("tsc binary not executable: %s", tsc_bin)
        return []
    except subprocess.TimeoutExpired:
        logger.warning("tsc timed out after %ds", _TOOL_TIMEOUT)
        return []
    except Exception as exc:
        logger.warning("tsc failed unexpectedly: %s", exc)
        return []

    issues: list[StaticIssue] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        parsed = _parse_tsc_line(root, line)
        if parsed:
            issues.append(parsed)

    logger.info("tsc: %d issues found", len(issues))
    return issues


def _parse_tsc_line(root: Path, line: str) -> StaticIssue | None:
    """Parse a single tsc diagnostic line.

    Format: ``path/to/file.ts(10,5): error TS2345: message``
    """
    # Must contain the TS error pattern
    if "): error TS" not in line and "): warning TS" not in line:
        return None

    # Split on the first "): "
    paren_idx = line.find("): ")
    if paren_idx == -1:
        return None

    location_part = line[:paren_idx]   # "path/to/file.ts(10,5"
    rest = line[paren_idx + 3:]        # "error TS2345: Argument of type..."

    # Extract file and line/col from location_part
    paren_open = location_part.rfind("(")
    if paren_open == -1:
        return None

    file_part = location_part[:paren_open]
    coords = location_part[paren_open + 1:]  # "10,5"
    line_no = 0
    col_no = 0
    if "," in coords:
        parts = coords.split(",", 1)
        try:
            line_no = int(parts[0])
            col_no = int(parts[1])
        except ValueError:
            pass
    else:
        try:
            line_no = int(coords)
        except ValueError:
            pass

    # Parse "error TS2345: message"
    severity: Severity = "error"
    rule_id = ""
    message = rest

    if rest.startswith("error "):
        severity = "error"
        rest_body = rest[len("error "):]
    elif rest.startswith("warning "):
        severity = "warning"
        rest_body = rest[len("warning "):]
    else:
        rest_body = rest

    # Extract TS error code
    if rest_body.startswith("TS") and ": " in rest_body:
        code_end = rest_body.index(": ")
        rule_id = rest_body[:code_end]
        message = rest_body[code_end + 2:]
    else:
        message = rest_body

    return StaticIssue(
        file=_rel(root, file_part.strip()),
        line=line_no,
        col=col_no,
        severity=severity,
        rule_id=rule_id,
        message=message.strip(),
        tool="tsc",
    )


def _find_tsc(root: Path) -> str | None:
    """Locate the tsc binary: local node_modules first, then PATH."""
    local = root / "node_modules" / ".bin" / "tsc"
    if local.exists():
        return str(local)
    import shutil
    return shutil.which("tsc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_available(tool: str) -> bool:
    """Return True if *tool* is importable as a Python module."""
    import importlib.util
    return importlib.util.find_spec(tool) is not None


def _rel(root: Path, path: str) -> str:
    """Return *path* relative to *root*, or the original string if not possible."""
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return path


def _sort_issues(issues: list[StaticIssue]) -> list[StaticIssue]:
    """Sort: errors first, then warnings, then info; then file+line."""
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: (order.get(i.severity, 9), i.file, i.line))
