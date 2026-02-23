"""Dependency graph analysis — module-level import topology with cycle detection.

Parses import statements purely in-process:
- Python: ``ast.Import`` / ``ast.ImportFrom``
- JavaScript / TypeScript: ``require(...)`` and ``import ... from ...`` via regex

Public API
----------
``DependencyAnalyzer(repo_path)``
    Main entry point.  Call ``analyze()`` to build the graph.

``DependencyGraph``
    Result model with:
    - ``get_imports(module)``     — direct imports of *module*
    - ``get_importers(module)``   — who imports *module*
    - ``cycles``                  — list of detected circular import paths
    - ``coupling_scores``         — {module: 0.0–1.0}
    - ``orphans``                 — modules with no importers
    - ``to_mermaid()``            — Mermaid diagram string
    - ``to_dict()``               — JSON-serialisable dict
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("analysis.dependency_graph")

_PY_EXTS  = {".py"}
_JS_EXTS  = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS  = {".ts", ".tsx"}

_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".tox", "dist", "build", ".daedalus",
}

_MAX_FILES = 500

# JS/TS import patterns
_JS_IMPORT_RE = re.compile(
    r"""(?:require\s*\(\s*['"]([^'"]+)['"]\s*\)|"""
    r"""import\s+.*?from\s+['"]([^'"]+)['"]|"""
    r"""import\s+['"]([^'"]+)['"])""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class DependencyGraph(BaseModel):
    """Module-level directed import graph.

    ``imports``:  module → list[modules it imports]   (out-edges)
    ``importers``: module → list[modules that import it]  (in-edges)
    ``cycles``:   list of cycle paths, each path is a list of module names
    ``coupling_scores``: module → float (0–1, higher = more coupled)
    ``language``: detected primary language
    ``files_analysed``: number of files processed
    ``parse_errors``: files skipped due to errors
    """

    imports:   dict[str, list[str]] = Field(default_factory=dict)
    importers: dict[str, list[str]] = Field(default_factory=dict)
    cycles:    list[list[str]]      = Field(default_factory=list)
    coupling_scores: dict[str, float] = Field(default_factory=dict)
    language:  str = "unknown"
    files_analysed: int = 0
    parse_errors:   int = 0

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_imports(self, module: str) -> list[str]:
        """Return modules directly imported by *module*."""
        return list(self.imports.get(module, []))

    def get_importers(self, module: str) -> list[str]:
        """Return modules that directly import *module*."""
        return list(self.importers.get(module, []))

    def all_modules(self) -> set[str]:
        return set(self.imports.keys()) | set(self.importers.keys())

    def orphans(self) -> list[str]:
        """Modules that nobody imports (potential dead modules)."""
        return sorted(m for m in self.all_modules() if not self.importers.get(m))

    def most_coupled(self, n: int = 5) -> list[tuple[str, float]]:
        """Top-*n* modules by coupling score."""
        return sorted(self.coupling_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    # ------------------------------------------------------------------
    # Mermaid generation
    # ------------------------------------------------------------------

    def to_mermaid(self, highlight_cycles: bool = True) -> str:
        """Return a Mermaid ``graph TD`` diagram string.

        Cycle edges are rendered in red when *highlight_cycles* is True.
        """
        lines = ["graph TD"]

        # Collect cycle edges for highlighting
        cycle_edges: set[tuple[str, str]] = set()
        if highlight_cycles:
            for cycle in self.cycles:
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((a, b))

        # Sanitise node IDs (Mermaid dislikes dots and slashes)
        def _id(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", name)

        rendered_nodes: set[str] = set()
        for mod, deps in self.imports.items():
            mid = _id(mod)
            if mid not in rendered_nodes:
                lines.append(f'    {mid}["{mod}"]')
                rendered_nodes.add(mid)
            for dep in deps:
                did = _id(dep)
                if did not in rendered_nodes:
                    lines.append(f'    {did}["{dep}"]')
                    rendered_nodes.add(did)
                if (mod, dep) in cycle_edges:
                    lines.append(f"    {mid} -->|cycle| {did}")
                else:
                    lines.append(f"    {mid} --> {did}")

        if highlight_cycles and cycle_edges:
            # Linkstyle for cycle edges (red)
            cycle_indices = [
                i for i, line in enumerate(lines)
                if "-->|cycle|" in line
            ]
            for idx in cycle_indices:
                lines.append(f"    linkStyle {idx - 1} stroke:#ff0000,stroke-width:2px")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        return cls(**data)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class DependencyAnalyzer:
    """Build a ``DependencyGraph`` for the repository at *repo_path*."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Not a directory: {self.repo_path}")

    def analyze(self) -> DependencyGraph:
        language = self._detect_language()
        logger.info("dep_graph: analysing %s repo at %s", language, self.repo_path)

        if language == "python":
            return self._analyse_python()
        if language in {"javascript", "typescript"}:
            return self._analyse_js_ts(language)

        logger.debug("dep_graph: unsupported language '%s'", language)
        return DependencyGraph(language=language)

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
        py_count = sum(1 for _ in self._iter_files(_PY_EXTS, limit=10))
        js_count = sum(1 for _ in self._iter_files(_JS_EXTS | _TS_EXTS, limit=10))
        return "python" if py_count >= js_count else "javascript"

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------

    def _analyse_python(self) -> DependencyGraph:
        """Parse Python import statements via AST."""
        # Map: module_name (dot-separated) → set of imported module names
        raw_imports: dict[str, set[str]] = defaultdict(set)
        # Track ALL discovered module names (including leaf files with no imports)
        all_modules: set[str] = set()
        files_ok = 0
        files_err = 0

        for path in self._iter_files(_PY_EXTS):
            mod_name = self._py_module_name(path)
            all_modules.add(mod_name)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            raw_imports[mod_name].add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            # Resolve relative imports to package-relative names
                            target = self._resolve_relative(mod_name, node.module, node.level or 0)
                            raw_imports[mod_name].add(target)
                files_ok += 1
            except SyntaxError as exc:
                logger.debug("dep_graph: syntax error in %s: %s", path, exc)
                files_err += 1
            except Exception as exc:
                logger.debug("dep_graph: read error %s: %s", path, exc)
                files_err += 1

        # Filter: only keep internal modules (those we discovered as files)
        # Use all_modules (includes leaf files) not raw_imports.keys()
        filtered: dict[str, set[str]] = {}
        for mod in all_modules:
            deps = raw_imports.get(mod, set())
            internal_deps = {d for d in deps if self._is_internal(d, all_modules)}
            filtered[mod] = internal_deps

        return self._build_graph(filtered, "python", files_ok, files_err)

    def _py_module_name(self, path: Path) -> str:
        """Convert a file path to a dotted module name relative to repo root."""
        try:
            rel = path.relative_to(self.repo_path)
        except ValueError:
            return path.stem
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        return ".".join(parts) if parts else path.stem

    def _resolve_relative(self, current_mod: str, target: str, level: int) -> str:
        """Resolve a relative import (``from . import x``) to an absolute name."""
        if level == 0:
            return target
        parts = current_mod.split(".")
        # Go up `level` levels
        base_parts = parts[: max(0, len(parts) - level)]
        return ".".join(base_parts + [target]) if target else ".".join(base_parts)

    def _is_internal(self, module: str, known: set[str]) -> bool:
        """Return True if *module* (or its package prefix) is in *known*."""
        if module in known:
            return True
        # Check package prefix: "app.core" matches "app"
        prefix = module.split(".")[0]
        return any(k == prefix or k.startswith(prefix + ".") for k in known)

    # ------------------------------------------------------------------
    # JS / TS
    # ------------------------------------------------------------------

    def _analyse_js_ts(self, language: str) -> DependencyGraph:
        """Parse JS/TS import/require statements via regex."""
        exts = _TS_EXTS if language == "typescript" else _JS_EXTS | _TS_EXTS
        raw_imports: dict[str, set[str]] = defaultdict(set)
        files_ok = 0
        files_err = 0

        for path in self._iter_files(exts):
            mod_name = self._js_module_name(path)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                for m in _JS_IMPORT_RE.finditer(source):
                    target = m.group(1) or m.group(2) or m.group(3)
                    if target and target.startswith("."):
                        # Relative import — resolve to a module name
                        resolved = self._resolve_js_relative(path, target)
                        raw_imports[mod_name].add(resolved)
                files_ok += 1
            except Exception as exc:
                logger.debug("dep_graph: read error %s: %s", path, exc)
                files_err += 1

        # Filter to only internal modules
        known = set(raw_imports.keys())
        filtered = {mod: {d for d in deps if d in known} for mod, deps in raw_imports.items()}

        return self._build_graph(filtered, language, files_ok, files_err)

    def _js_module_name(self, path: Path) -> str:
        """Return a slash-separated module name relative to repo root."""
        try:
            rel = path.relative_to(self.repo_path)
        except ValueError:
            return path.stem
        # Strip extension
        name = str(rel)
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        return name

    def _resolve_js_relative(self, current_file: Path, target: str) -> str:
        """Resolve a relative JS import path to our internal module name."""
        resolved = (current_file.parent / target).resolve()
        return self._js_module_name(resolved)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self,
        raw: dict[str, set[str]],
        language: str,
        files_ok: int,
        files_err: int,
    ) -> DependencyGraph:
        imports: dict[str, list[str]] = {m: sorted(deps) for m, deps in raw.items()}

        # Build reverse index (importers)
        importers: dict[str, list[str]] = defaultdict(list)
        for mod, deps in imports.items():
            for dep in deps:
                importers[dep].append(mod)
        importers_sorted = {m: sorted(set(v)) for m, v in importers.items()}

        # Ensure every module appears in both dicts
        all_mods = set(imports.keys()) | set(importers_sorted.keys())
        for m in all_mods:
            imports.setdefault(m, [])
            importers_sorted.setdefault(m, [])

        # Detect cycles
        cycles = _detect_cycles(imports)

        # Coupling scores
        n = len(all_mods) or 1
        coupling: dict[str, float] = {}
        for mod in all_mods:
            out_deg = len(imports.get(mod, []))
            in_deg  = len(importers_sorted.get(mod, []))
            coupling[mod] = min(1.0, (in_deg + out_deg) / n)

        total_edges = sum(len(v) for v in imports.values())
        logger.info(
            "dep_graph: %s — %d modules, %d edges, %d cycle(s), %d files (%d errors)",
            language, len(all_mods), total_edges, len(cycles), files_ok, files_err,
        )

        return DependencyGraph(
            imports=imports,
            importers=importers_sorted,
            cycles=cycles,
            coupling_scores=coupling,
            language=language,
            files_analysed=files_ok,
            parse_errors=files_err,
        )

    # ------------------------------------------------------------------
    # File iteration
    # ------------------------------------------------------------------

    def _iter_files(self, exts: set[str], limit: int = _MAX_FILES) -> Iterator[Path]:
        count = 0
        for path in sorted(self.repo_path.rglob("*")):
            if count >= limit:
                break
            if path.is_file() and path.suffix in exts:
                if not any(part in _SKIP_DIRS for part in path.parts):
                    yield path
                    count += 1


# ---------------------------------------------------------------------------
# Cycle detection (DFS)
# ---------------------------------------------------------------------------

def _detect_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Return all unique simple cycles in *graph* using iterative DFS.

    Each cycle is represented as the list of nodes forming the loop,
    e.g. ``["a", "b", "c"]`` means a→b→c→a.
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def _dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                _dfs(neighbour)
            elif neighbour in rec_stack:
                # Found a back-edge → extract the cycle
                cycle_start = path.index(neighbour)
                cycle = path[cycle_start:]
                # Normalise: rotate so the lexicographically smallest node is first
                min_idx = cycle.index(min(cycle))
                normalised = cycle[min_idx:] + cycle[:min_idx]
                if normalised not in cycles:
                    cycles.append(normalised)

        path.pop()
        rec_stack.discard(node)

    for node in sorted(graph.keys()):
        if node not in visited:
            _dfs(node)

    return cycles


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_dep_graph_for_prompt(graph: DependencyGraph, max_cycles: int = 5) -> str:
    """Return a concise, prompt-ready summary of *graph*."""
    if not graph.all_modules():
        return "## Dependency Graph\nNo modules detected."

    n_mods  = len(graph.all_modules())
    n_edges = sum(len(v) for v in graph.imports.values())
    n_cyc   = len(graph.cycles)

    lines = [
        f"## Dependency Graph — {n_mods} modules, {n_edges} edges"
        f", {n_cyc} cycle(s) ({graph.language})",
        "",
    ]

    if graph.cycles:
        lines.append(f"### ⚠️  Circular imports ({min(n_cyc, max_cycles)} shown):")
        for cycle in graph.cycles[:max_cycles]:
            lines.append("- " + " → ".join(cycle) + f" → {cycle[0]}")
        if n_cyc > max_cycles:
            lines.append(f"  ... and {n_cyc - max_cycles} more")
        lines.append("")

    top = graph.most_coupled(5)
    if top:
        lines.append("### Most-coupled modules (high change risk):")
        for mod, score in top:
            lines.append(f"- `{mod}` (coupling={score:.2f})")

    return "\n".join(lines)
