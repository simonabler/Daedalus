"""Call graph analysis via Python AST and JS/TS regex parsing.

Builds a function-level directed call graph for Python, JavaScript, and
TypeScript repositories **without** invoking any shell tools or external
processes.  Everything runs in-process using the standard ``ast`` module
(Python) or lightweight regex parsing (JS/TS).

Public API
----------
``CallGraphAnalyzer(repo_path)``
    Main entry point.  Call ``analyze()`` to build the graph.

``CallGraph``
    Result model with helper methods:
    - ``get_callers(func)``  — who calls *func*
    - ``get_callees(func)``  — what *func* calls
    - ``get_impact_radius(func, depth=2)``  — transitive blast-radius
    - ``most_called(n=10)``  — hottest functions by in-degree
    - ``to_dict()``  — JSON-serialisable representation
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("analysis.call_graph")

# File extensions we analyse per language.
_PY_EXTS = {".py"}
_JS_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS = {".ts", ".tsx"}

# Directories we always skip.
_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".tox", "dist", "build",
    ".daedalus",
}

# Maximum files to analyse (safety limit for very large repos).
_MAX_FILES = 500


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class CallGraph(BaseModel):
    """Directed call graph for a repository.

    ``callers``: func_name → list of functions that call it  (in-edges)
    ``callees``: func_name → list of functions it calls      (out-edges)
    ``file_map``: func_name → source file (relative path)
    ``language``: detected primary language
    ``files_analysed``: number of source files processed
    ``parse_errors``: files skipped due to syntax errors
    """

    callers: dict[str, list[str]] = Field(default_factory=dict)
    callees: dict[str, list[str]] = Field(default_factory=dict)
    file_map: dict[str, str] = Field(default_factory=dict)
    language: str = "unknown"
    files_analysed: int = 0
    parse_errors: int = 0

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_callers(self, func: str) -> list[str]:
        """Return all functions that directly call *func*."""
        return list(self.callers.get(func, []))

    def get_callees(self, func: str) -> list[str]:
        """Return all functions directly called by *func*."""
        return list(self.callees.get(func, []))

    def get_impact_radius(self, func: str, depth: int = 2) -> set[str]:
        """Return all functions transitively affected when *func* changes.

        "Affected" means: functions that call *func* (direct or indirect)
        up to *depth* hops.  Depth 1 = direct callers only.
        """
        all_found: set[str] = set()
        frontier = {func}
        visited: set[str] = {func}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for f in frontier:
                for caller in self.callers.get(f, []):
                    if caller not in visited:
                        next_frontier.add(caller)
                        all_found.add(caller)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return all_found

    def most_called(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-*n* functions ranked by number of callers."""
        ranked = sorted(
            ((fn, len(callers)) for fn, callers in self.callers.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]

    def all_functions(self) -> set[str]:
        """Return every function name present in the graph."""
        return set(self.callees.keys()) | set(self.callers.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (suitable for GraphState storage)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "CallGraph":
        return cls(**data)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class CallGraphAnalyzer:
    """Build a ``CallGraph`` for the repository at *repo_path*."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Not a directory: {self.repo_path}")

    def analyze(self) -> CallGraph:
        """Walk the repository and return a populated ``CallGraph``."""
        language = self._detect_language()
        logger.info("call_graph: analysing %s repo at %s", language, self.repo_path)

        if language == "python":
            return self._analyse_python()
        if language in {"javascript", "typescript"}:
            return self._analyse_js_ts(language)

        logger.debug("call_graph: unsupported language '%s'", language)
        return CallGraph(language=language)

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _detect_language(self) -> str:
        if (self.repo_path / "pyproject.toml").exists() or \
           (self.repo_path / "setup.py").exists():
            return "python"
        if (self.repo_path / "tsconfig.json").exists():
            return "typescript"
        if (self.repo_path / "package.json").exists():
            return "javascript"
        # Fall back to file extension count
        py_count = sum(1 for _ in self._iter_files(_PY_EXTS, limit=10))
        js_count = sum(1 for _ in self._iter_files(_JS_EXTS | _TS_EXTS, limit=10))
        if py_count >= js_count:
            return "python"
        return "javascript"

    # ------------------------------------------------------------------
    # Python AST analysis
    # ------------------------------------------------------------------

    def _analyse_python(self) -> CallGraph:
        """Walk .py files and extract caller → callee relationships."""
        # Pass 1: collect all defined function names with their file
        defined: dict[str, str] = {}  # qualified_name → rel_path
        trees: list[tuple[str, ast.Module]] = []

        files_ok = 0
        files_err = 0

        for path in self._iter_files(_PY_EXTS):
            rel = str(path.relative_to(self.repo_path))
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(path))
                trees.append((rel, tree))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defined[node.name] = rel
                files_ok += 1
            except SyntaxError as exc:
                logger.debug("call_graph: syntax error in %s: %s", rel, exc)
                files_err += 1
            except Exception as exc:
                logger.debug("call_graph: could not read %s: %s", rel, exc)
                files_err += 1

        # Pass 2: for each function body, find calls to known functions
        callees_raw: dict[str, set[str]] = defaultdict(set)

        for rel, tree in trees:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                caller_name = node.name
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    callee_name = _extract_call_name(child)
                    if callee_name and callee_name in defined and callee_name != caller_name:
                        callees_raw[caller_name].add(callee_name)

        return self._build_graph(
            defined=defined,
            callees_raw=callees_raw,
            language="python",
            files_analysed=files_ok,
            parse_errors=files_err,
        )

    # ------------------------------------------------------------------
    # JS / TS regex analysis
    # ------------------------------------------------------------------

    # Matches: function foo(  /  const foo = (  /  const foo = async (
    _FUNC_DEF_RE = re.compile(
        r"(?:^|\s)(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("
        r"|(?:^|\s)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
        re.MULTILINE,
    )
    # Matches: foo(  — simple direct call
    _CALL_RE = re.compile(r"\b(\w+)\s*\(", re.MULTILINE)

    def _analyse_js_ts(self, language: str) -> CallGraph:
        """Regex-based analysis for JS/TS files."""
        exts = _TS_EXTS if language == "typescript" else _JS_EXTS | _TS_EXTS

        # Pass 1: collect all defined function names
        defined: dict[str, str] = {}
        file_contents: list[tuple[str, str]] = []

        files_ok = 0
        files_err = 0

        for path in self._iter_files(exts):
            rel = str(path.relative_to(self.repo_path))
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                file_contents.append((rel, source))
                for m in self._FUNC_DEF_RE.finditer(source):
                    name = m.group(1) or m.group(2)
                    if name:
                        defined[name] = rel
                files_ok += 1
            except Exception as exc:
                logger.debug("call_graph: could not read %s: %s", rel, exc)
                files_err += 1

        # Pass 2: crude per-function call extraction using line ranges
        callees_raw: dict[str, set[str]] = defaultdict(set)

        for rel, source in file_contents:
            lines = source.splitlines()
            func_ranges = _js_function_ranges(source, self._FUNC_DEF_RE)

            for func_name, start_line, end_line in func_ranges:
                body = "\n".join(lines[start_line:end_line])
                for m in self._CALL_RE.finditer(body):
                    callee = m.group(1)
                    if callee in defined and callee != func_name:
                        callees_raw[func_name].add(callee)

        return self._build_graph(
            defined=defined,
            callees_raw=callees_raw,
            language=language,
            files_analysed=files_ok,
            parse_errors=files_err,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        defined: dict[str, str],
        callees_raw: dict[str, set[str]],
        language: str,
        files_analysed: int,
        parse_errors: int,
    ) -> CallGraph:
        """Convert raw sets into the final CallGraph model."""
        callees: dict[str, list[str]] = {}
        callers: dict[str, list[str]] = defaultdict(list)

        # Ensure every defined function appears in callees (even if it calls nothing)
        for fn in defined:
            callees[fn] = sorted(callees_raw.get(fn, set()))

        # Build reverse index
        for caller, calls in callees.items():
            for callee in calls:
                callers[callee].append(caller)

        # Sort caller lists for determinism
        callers_sorted = {fn: sorted(set(lst)) for fn, lst in callers.items()}

        logger.info(
            "call_graph: %s — %d functions, %d edges, %d files (%d errors)",
            language,
            len(defined),
            sum(len(v) for v in callees.values()),
            files_analysed,
            parse_errors,
        )

        return CallGraph(
            callers=callers_sorted,
            callees=callees,
            file_map=defined,
            language=language,
            files_analysed=files_analysed,
            parse_errors=parse_errors,
        )

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    def _iter_files(self, exts: set[str], limit: int = _MAX_FILES) -> Iterator[Path]:
        """Yield source files under repo_path matching *exts*, skipping noise dirs."""
        count = 0
        for path in sorted(self.repo_path.rglob("*")):
            if count >= limit:
                logger.debug("call_graph: file limit (%d) reached", limit)
                break
            if path.is_file() and path.suffix in exts:
                if not any(part in _SKIP_DIRS for part in path.parts):
                    yield path
                    count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_call_name(node: ast.Call) -> str | None:
    """Return the bare function name from an ast.Call node, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _js_function_ranges(
    source: str,
    func_re: re.Pattern,
) -> list[tuple[str, int, int]]:
    """Return (func_name, start_line, end_line) tuples for JS/TS functions.

    Uses a simple brace-counting heuristic to find the closing ``}``.
    """
    lines = source.splitlines()
    ranges: list[tuple[str, int, int]] = []

    for m in func_re.finditer(source):
        name = m.group(1) or m.group(2)
        if not name:
            continue

        # Find the line where this match starts
        start_pos = m.start()
        start_line = source[:start_pos].count("\n")

        # Walk forward counting braces from that line
        depth = 0
        end_line = start_line
        found_open = False
        for i, line in enumerate(lines[start_line:], start=start_line):
            for ch in line:
                if ch == "{":
                    depth += 1
                    found_open = True
                elif ch == "}":
                    depth -= 1
            if found_open and depth == 0:
                end_line = i + 1
                break
        else:
            end_line = len(lines)

        ranges.append((name, start_line, end_line))

    return ranges


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_call_graph_for_prompt(graph: CallGraph, max_entries: int = 15) -> str:
    """Return a concise, prompt-ready summary of *graph*."""
    if not graph.all_functions():
        return "## Call Graph\nNo functions detected."

    total_funcs = len(graph.all_functions())
    total_edges = sum(len(v) for v in graph.callees.values())

    lines = [
        f"## Call Graph — {total_funcs} functions, {total_edges} edges"
        f" ({graph.language}, {graph.files_analysed} files)",
        "",
    ]

    top = graph.most_called(max_entries)
    if top:
        lines.append("### Most-called functions (change carefully):")
        for fn, count in top:
            file_hint = graph.file_map.get(fn, "")
            file_part = f"  [{file_hint}]" if file_hint else ""
            lines.append(f"- `{fn}` — called by {count} function(s){file_part}")

    return "\n".join(lines)
