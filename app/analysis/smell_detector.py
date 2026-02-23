"""Code smell detection — 8 AST-based rules for Python.

All analysis runs in-process using the standard ``ast`` module.
No shell tools, no subprocesses, no external dependencies.

Rules
-----
1. Long Function      — function body > 50 lines (WARNING) or > 100 lines (ERROR)
2. High Complexity    — cyclomatic complexity > 10 (ERROR)
3. God Class          — class with > 20 methods or > 500 lines (ERROR)
4. Long Parameter List — function with > 5 parameters (WARNING)
5. Duplicate Code     — two functions whose non-blank bodies are ≥ 80 % identical (WARNING)
6. Dead Code          — function defined but never called (needs call-graph data, INFO)
7. Magic Numbers      — numeric literals outside assignments to named constants (INFO)
8. Deeply Nested      — statement nesting depth > 4 levels (WARNING)

Public API
----------
``SmellDetector(repo_path, call_graph=None)``
    Accepts an optional ``CallGraph`` dict (from #12) to enable dead-code detection.
    Call ``detect()`` → ``list[CodeSmell]``.

``CodeSmell``
    Pydantic model: file, line, smell_type, severity, description, suggestion.

``format_smells_for_prompt(smells, max_smells=10)``
    Returns a compact, prompt-ready summary.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("analysis.smell_detector")

Severity = Literal["error", "warning", "info"]

_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".tox", "dist", "build", ".daedalus",
}
_MAX_FILES = 500

# Thresholds
_LONG_FUNC_WARN  = 50
_LONG_FUNC_ERROR = 100
_HIGH_COMPLEXITY = 10
_GOD_CLASS_METHODS = 20
_GOD_CLASS_LINES   = 500
_LONG_PARAM_COUNT  = 5
_DUPLICATE_THRESH  = 0.75   # 75 % similarity
_MAX_NESTING       = 4
# Magic-number whitelist (values so common they're not smells)
_MAGIC_WHITELIST: frozenset[complex] = frozenset({0, 1, -1, 2, 100})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class CodeSmell(BaseModel):
    """A single code smell finding."""

    file: str
    line: int = 0
    smell_type: str            # rule name, e.g. "LongFunction"
    severity: Severity = "warning"
    description: str
    suggestion: str = ""

    def one_line(self) -> str:
        sev = self.severity.upper()
        return (
            f"[{sev}] {self.smell_type} @ {self.file}:{self.line} — "
            f"{self.description}"
            + (f" | FIX: {self.suggestion}" if self.suggestion else "")
        )


_SEV_ORDER = {"error": 0, "warning": 1, "info": 2}


def sort_smells(smells: list[CodeSmell]) -> list[CodeSmell]:
    return sorted(smells, key=lambda s: (_SEV_ORDER.get(s.severity, 9), s.file, s.line))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SmellDetector:
    """Detect code smells in a Python repository."""

    def __init__(self, repo_path: str | Path, call_graph: dict | None = None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._call_graph = call_graph or {}

    def detect(self) -> list[CodeSmell]:
        """Run all 8 rules and return sorted findings."""
        smells: list[CodeSmell] = []
        trees: list[tuple[str, ast.Module, list[str]]] = []

        for path in self._iter_files():
            rel = str(path.relative_to(self.repo_path))
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(path))
                lines = source.splitlines()
                trees.append((rel, tree, lines))
            except SyntaxError:
                logger.debug("smell: syntax error in %s, skipping", rel)
            except Exception as exc:
                logger.debug("smell: could not read %s: %s", rel, exc)

        for rel, tree, lines in trees:
            smells.extend(_rule_long_function(rel, tree, lines))
            smells.extend(_rule_high_complexity(rel, tree))
            smells.extend(_rule_god_class(rel, tree, lines))
            smells.extend(_rule_long_param_list(rel, tree))
            smells.extend(_rule_deeply_nested(rel, tree))
            smells.extend(_rule_magic_numbers(rel, tree))
            smells.extend(_rule_dead_code(rel, tree, self._call_graph))

        # Rule 5: duplicate code across all files (needs all trees)
        smells.extend(_rule_duplicate_code(trees))

        logger.info(
            "smell: %d smells found (%d error, %d warning, %d info)",
            len(smells),
            sum(1 for s in smells if s.severity == "error"),
            sum(1 for s in smells if s.severity == "warning"),
            sum(1 for s in smells if s.severity == "info"),
        )
        return sort_smells(smells)

    def _iter_files(self, limit: int = _MAX_FILES) -> Iterator[Path]:
        count = 0
        for path in sorted(self.repo_path.rglob("*.py")):
            if count >= limit:
                break
            if not any(part in _SKIP_DIRS for part in path.parts):
                yield path
                count += 1


# ---------------------------------------------------------------------------
# Rule 1 — Long Function
# ---------------------------------------------------------------------------

def _rule_long_function(rel: str, tree: ast.Module, lines: list[str]) -> list[CodeSmell]:
    smells = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        length = end - node.lineno + 1
        if length > _LONG_FUNC_ERROR:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="LongFunction", severity="error",
                description=f"`{node.name}` is {length} lines long (limit {_LONG_FUNC_ERROR})",
                suggestion="Extract helper functions to reduce size below 50 lines.",
            ))
        elif length > _LONG_FUNC_WARN:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="LongFunction", severity="warning",
                description=f"`{node.name}` is {length} lines long (limit {_LONG_FUNC_WARN})",
                suggestion="Consider splitting into smaller focused functions.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 2 — High Complexity (cyclomatic)
# ---------------------------------------------------------------------------

def _cyclomatic_complexity(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Cyclomatic complexity = 1 + number of branches."""
    branch_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )
    count = 1
    for node in ast.walk(func):
        if isinstance(node, branch_nodes):
            count += 1
        # BoolOp: each 'and'/'or' adds a path
        elif isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
    return count


def _rule_high_complexity(rel: str, tree: ast.Module) -> list[CodeSmell]:
    smells = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        cc = _cyclomatic_complexity(node)
        if cc > _HIGH_COMPLEXITY:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="HighComplexity", severity="error",
                description=f"`{node.name}` has cyclomatic complexity {cc} (limit {_HIGH_COMPLEXITY})",
                suggestion="Break into smaller functions; reduce branching and boolean chains.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 3 — God Class
# ---------------------------------------------------------------------------

def _rule_god_class(rel: str, tree: ast.Module, lines: list[str]) -> list[CodeSmell]:
    smells = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        end = getattr(node, "end_lineno", node.lineno)
        class_lines = end - node.lineno + 1

        reasons = []
        if len(methods) > _GOD_CLASS_METHODS:
            reasons.append(f"{len(methods)} methods (limit {_GOD_CLASS_METHODS})")
        if class_lines > _GOD_CLASS_LINES:
            reasons.append(f"{class_lines} lines (limit {_GOD_CLASS_LINES})")

        if reasons:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="GodClass", severity="error",
                description=f"`{node.name}` is a God Class: {', '.join(reasons)}",
                suggestion="Split responsibilities into smaller, focused classes (SRP).",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 4 — Long Parameter List
# ---------------------------------------------------------------------------

def _rule_long_param_list(rel: str, tree: ast.Module) -> list[CodeSmell]:
    smells = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        # Count all param kinds except *args and **kwargs markers themselves
        param_count = (
            len(args.args)
            + len(args.posonlyargs)
            + len(args.kwonlyargs)
            + (1 if args.vararg else 0)
            + (1 if args.kwarg else 0)
        )
        # Exclude 'self' / 'cls'
        if args.args and args.args[0].arg in {"self", "cls"}:
            param_count -= 1

        if param_count > _LONG_PARAM_COUNT:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="LongParameterList", severity="warning",
                description=f"`{node.name}` has {param_count} parameters (limit {_LONG_PARAM_COUNT})",
                suggestion="Introduce a parameter object or use **kwargs with clear documentation.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 5 — Duplicate Code
# ---------------------------------------------------------------------------

def _body_fingerprint(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return non-blank source lines of function body for similarity comparison."""
    lines = []
    for node in ast.walk(func):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # skip docstrings
        lines.append(ast.dump(node))
    return lines


def _similarity(a: list[str], b: list[str]) -> float:
    """Jaccard-like similarity between two fingerprints."""
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _rule_duplicate_code(
    trees: list[tuple[str, ast.Module, list[str]]],
) -> list[CodeSmell]:
    # Collect all functions with their fingerprints
    funcs: list[tuple[str, int, str, list[str]]] = []  # (rel, line, name, fingerprint)
    for rel, tree, _ in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fp = _body_fingerprint(node)
                if len(fp) >= 5:  # skip trivial functions
                    funcs.append((rel, node.lineno, node.name, fp))

    smells = []
    reported: set[frozenset] = set()
    for (rel_a, line_a, name_a, fp_a), (rel_b, line_b, name_b, fp_b) in combinations(funcs, 2):
        if name_a == name_b:
            continue  # overrides are expected
        pair = frozenset({(rel_a, line_a), (rel_b, line_b)})
        if pair in reported:
            continue
        sim = _similarity(fp_a, fp_b)
        if sim >= _DUPLICATE_THRESH:
            reported.add(pair)
            smells.append(CodeSmell(
                file=rel_a, line=line_a, smell_type="DuplicateCode", severity="warning",
                description=(
                    f"`{name_a}` ({rel_a}:{line_a}) and `{name_b}` ({rel_b}:{line_b}) "
                    f"are {sim:.0%} similar"
                ),
                suggestion="Extract shared logic into a common helper function.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 6 — Dead Code (uses call graph from #12)
# ---------------------------------------------------------------------------

def _rule_dead_code(rel: str, tree: ast.Module, call_graph: dict) -> list[CodeSmell]:
    if not call_graph:
        return []

    callers: dict = call_graph.get("callers", {})
    smells = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        # Skip dunder methods, test functions, and well-known entry points
        if name.startswith("__") or name.startswith("test_") or name in {"main", "setup", "teardown"}:
            continue
        # Dead if it has no callers in the graph
        if name in call_graph.get("callees", {}) and not callers.get(name):
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="DeadCode", severity="info",
                description=f"`{name}` is defined but never called",
                suggestion="Remove or export this function if it is unused.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 7 — Magic Numbers
# ---------------------------------------------------------------------------

_MAGIC_SKIP_CONTEXTS = (ast.AnnAssign, ast.Assign)


def _rule_magic_numbers(rel: str, tree: ast.Module) -> list[CodeSmell]:
    smells = []
    # Only flag literals inside function bodies, not module-level constants
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Constant):
                continue
            val = node.value
            if not isinstance(val, (int, float)):
                continue
            if val in _MAGIC_WHITELIST:
                continue
            smells.append(CodeSmell(
                file=rel, line=node.col_offset and getattr(node, "lineno", 0),
                smell_type="MagicNumber", severity="info",
                description=f"Magic number `{val}` used directly in `{func.name}`",
                suggestion=f"Replace with a named constant, e.g. `MAX_VALUE = {val}`.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Rule 8 — Deeply Nested
# ---------------------------------------------------------------------------

_NESTING_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)


def _max_nesting_depth(node: ast.AST, current: int = 0) -> int:
    """Recursively find the maximum nesting depth of branching/looping nodes."""
    if isinstance(node, _NESTING_NODES):
        current += 1
    max_depth = current
    for child in ast.iter_child_nodes(node):
        max_depth = max(max_depth, _max_nesting_depth(child, current))
    return max_depth


def _rule_deeply_nested(rel: str, tree: ast.Module) -> list[CodeSmell]:
    smells = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        depth = _max_nesting_depth(node)
        if depth > _MAX_NESTING:
            smells.append(CodeSmell(
                file=rel, line=node.lineno, smell_type="DeeplyNested", severity="warning",
                description=f"`{node.name}` has nesting depth {depth} (limit {_MAX_NESTING})",
                suggestion="Use early returns, extract nested blocks into helper functions.",
            ))
    return smells


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_smells_for_prompt(smells: list[CodeSmell], max_smells: int = 10) -> str:
    """Return a concise, prompt-ready summary of *smells*."""
    if not smells:
        return "## Code Smells\nNo smells detected."

    errors   = [s for s in smells if s.severity == "error"]
    warnings = [s for s in smells if s.severity == "warning"]
    infos    = [s for s in smells if s.severity == "info"]

    selected = (errors + warnings + infos)[:max_smells]
    total    = len(smells)
    shown    = len(selected)

    header = (
        f"## Code Smells — {len(errors)} error(s)"
        f", {len(warnings)} warning(s)"
        f", {len(infos)} info(s)"
    )
    if total > shown:
        header += f" (showing top {shown} of {total})"

    lines = [header, ""]
    for smell in selected:
        lines.append(f"- {smell.one_line()}")

    return "\n".join(lines)
