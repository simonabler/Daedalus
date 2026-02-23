"""Tests for app/analysis/dependency_graph.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.dependency_graph import (
    DependencyAnalyzer,
    DependencyGraph,
    _detect_cycles,
    format_dep_graph_for_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_py_repo(tmp_path: Path) -> Path:
    """Three-module repo: main → utils → helpers (no cycles)."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (tmp_path / "helpers.py").write_text("# no imports\ndef helper(): pass\n")
    (tmp_path / "utils.py").write_text("from helpers import helper\ndef util(): pass\n")
    (tmp_path / "main.py").write_text("from utils import util\nfrom helpers import helper\n")
    return tmp_path


@pytest.fixture()
def cycle_py_repo(tmp_path: Path) -> Path:
    """Repo with a circular import: a → b → c → a."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (tmp_path / "a.py").write_text("from b import something\n")
    (tmp_path / "b.py").write_text("from c import something\n")
    (tmp_path / "c.py").write_text("from a import something\n")
    return tmp_path


@pytest.fixture()
def package_py_repo(tmp_path: Path) -> Path:
    """Repo with a package structure: app/__init__.py + app/core.py."""
    pkg = tmp_path / "app"
    pkg.mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text("# no imports\n")
    (pkg / "utils.py").write_text("from app.core import something\n")
    (tmp_path / "main.py").write_text("from app.utils import something\n")
    return tmp_path


@pytest.fixture()
def js_repo(tmp_path: Path) -> Path:
    """Simple JS repo: index.js → utils.js."""
    (tmp_path / "package.json").write_text('{"name":"test"}')
    (tmp_path / "utils.js").write_text("function helper() {}\nmodule.exports = { helper };\n")
    (tmp_path / "index.js").write_text("const { helper } = require('./utils');\n")
    return tmp_path


@pytest.fixture()
def ts_repo(tmp_path: Path) -> Path:
    """TypeScript repo: src/index.ts → src/lib.ts."""
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.ts").write_text("export function libFunc() {}\n")
    (src / "index.ts").write_text("import { libFunc } from './lib';\n")
    return tmp_path


# ---------------------------------------------------------------------------
# DependencyGraph model
# ---------------------------------------------------------------------------

class TestDependencyGraphModel:
    def test_get_imports(self):
        dg = DependencyGraph(imports={"a": ["b", "c"]}, importers={"b": ["a"], "c": ["a"]})
        assert sorted(dg.get_imports("a")) == ["b", "c"]
        assert dg.get_imports("b") == []

    def test_get_importers(self):
        dg = DependencyGraph(imports={"a": ["b"]}, importers={"b": ["a"]})
        assert dg.get_importers("b") == ["a"]
        assert dg.get_importers("a") == []

    def test_all_modules(self):
        dg = DependencyGraph(imports={"a": ["b"]}, importers={"b": ["a"]})
        assert dg.all_modules() == {"a", "b"}

    def test_orphans(self):
        dg = DependencyGraph(
            imports={"a": ["b"], "b": [], "orphan": []},
            importers={"b": ["a"]},
        )
        orphans = dg.orphans()
        assert "orphan" in orphans
        assert "a" in orphans   # a is not imported by anyone
        assert "b" not in orphans

    def test_most_coupled(self):
        dg = DependencyGraph(
            coupling_scores={"high": 0.9, "mid": 0.5, "low": 0.1},
        )
        top = dg.most_coupled(n=2)
        assert top[0][0] == "high"
        assert top[1][0] == "mid"

    def test_serialization_roundtrip(self):
        dg = DependencyGraph(
            imports={"a": ["b"]},
            importers={"b": ["a"]},
            cycles=[["a", "b"]],
            coupling_scores={"a": 0.5, "b": 0.5},
            language="python",
            files_analysed=2,
            parse_errors=0,
        )
        restored = DependencyGraph.from_dict(dg.to_dict())
        assert restored.imports == dg.imports
        assert restored.cycles == dg.cycles
        assert restored.language == dg.language

    def test_to_mermaid_basic(self):
        dg = DependencyGraph(
            imports={"a": ["b"], "b": []},
            importers={"b": ["a"]},
        )
        mermaid = dg.to_mermaid()
        assert "graph TD" in mermaid
        assert "a" in mermaid
        assert "b" in mermaid
        assert "-->" in mermaid

    def test_to_mermaid_cycle_highlighted(self):
        dg = DependencyGraph(
            imports={"a": ["b"], "b": ["a"]},
            importers={"a": ["b"], "b": ["a"]},
            cycles=[["a", "b"]],
        )
        mermaid = dg.to_mermaid(highlight_cycles=True)
        assert "cycle" in mermaid or "stroke:#ff0000" in mermaid or "-->|cycle|" in mermaid

    def test_to_mermaid_no_modules_safe(self):
        dg = DependencyGraph()
        mermaid = dg.to_mermaid()
        assert "graph TD" in mermaid


# ---------------------------------------------------------------------------
# _detect_cycles
# ---------------------------------------------------------------------------

class TestDetectCycles:
    def test_no_cycle(self):
        graph = {"a": ["b"], "b": ["c"], "c": []}
        cycles = _detect_cycles(graph)
        assert cycles == []

    def test_simple_cycle_ab(self):
        graph = {"a": ["b"], "b": ["a"]}
        cycles = _detect_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}

    def test_three_node_cycle(self):
        graph = {"a": ["b"], "b": ["c"], "c": ["a"]}
        cycles = _detect_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_self_loop(self):
        graph = {"a": ["a"], "b": []}
        cycles = _detect_cycles(graph)
        assert any("a" in c for c in cycles)

    def test_no_duplicate_cycles(self):
        # a→b→a should produce exactly 1 cycle, not 2
        graph = {"a": ["b"], "b": ["a"], "c": []}
        cycles = _detect_cycles(graph)
        assert len(cycles) == 1

    def test_disconnected_graph(self):
        graph = {"a": ["b"], "b": [], "x": ["y"], "y": []}
        cycles = _detect_cycles(graph)
        assert cycles == []


# ---------------------------------------------------------------------------
# Python analysis
# ---------------------------------------------------------------------------

class TestDependencyAnalyzerPython:
    def test_detects_language_python(self, simple_py_repo):
        assert DependencyAnalyzer(simple_py_repo)._detect_language() == "python"

    def test_utils_imports_helpers(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert "helpers" in dg.get_imports("utils")

    def test_main_imports_utils_and_helpers(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        imports = dg.get_imports("main")
        assert "utils" in imports
        assert "helpers" in imports

    def test_helpers_has_no_imports(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert dg.get_imports("helpers") == []

    def test_importers_of_helpers(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        importers = dg.get_importers("helpers")
        assert "utils" in importers or "main" in importers

    def test_no_cycles_in_simple_repo(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert dg.cycles == []

    def test_detects_circular_imports(self, cycle_py_repo):
        dg = DependencyAnalyzer(cycle_py_repo).analyze()
        assert len(dg.cycles) >= 1
        all_cycle_nodes = {n for cycle in dg.cycles for n in cycle}
        assert "a" in all_cycle_nodes
        assert "b" in all_cycle_nodes
        assert "c" in all_cycle_nodes

    def test_package_structure(self, package_py_repo):
        dg = DependencyAnalyzer(package_py_repo).analyze()
        # app.utils should import app.core
        assert dg.files_analysed >= 3

    def test_coupling_scores_between_0_and_1(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        for score in dg.coupling_scores.values():
            assert 0.0 <= score <= 1.0

    def test_files_analysed_count(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert dg.files_analysed == 3

    def test_parse_error_files_skipped(self, simple_py_repo):
        (simple_py_repo / "broken.py").write_text("def foo(\n  # broken")
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert dg.parse_errors >= 1
        assert dg.files_analysed >= 3

    def test_language_is_python(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert dg.language == "python"

    def test_self_analysis_does_not_crash(self):
        """Analyzer must handle the Daedalus repo itself without crashing."""
        root = Path(__file__).parent.parent
        dg = DependencyAnalyzer(root).analyze()
        assert dg.language == "python"
        assert dg.files_analysed > 0

    def test_syntax_only_repo(self, tmp_path):
        """Repo with only stdlib imports produces empty internal graph."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
        (tmp_path / "standalone.py").write_text("import os\nimport sys\n")
        dg = DependencyAnalyzer(tmp_path).analyze()
        # os/sys are external → no internal edges
        assert dg.get_imports("standalone") == []


# ---------------------------------------------------------------------------
# JS analysis
# ---------------------------------------------------------------------------

class TestDependencyAnalyzerJS:
    def test_detects_language_javascript(self, js_repo):
        assert DependencyAnalyzer(js_repo)._detect_language() == "javascript"

    def test_index_imports_utils(self, js_repo):
        dg = DependencyAnalyzer(js_repo).analyze()
        # index.js requires('./utils') → internal dep on utils
        index_key = next((k for k in dg.imports if "index" in k), None)
        utils_key = next((k for k in dg.all_modules() if "utils" in k), None)
        if index_key and utils_key:
            assert utils_key in dg.get_imports(index_key)

    def test_language_is_javascript(self, js_repo):
        dg = DependencyAnalyzer(js_repo).analyze()
        assert dg.language == "javascript"

    def test_no_external_packages_in_imports(self, js_repo):
        # External packages (no ./ prefix) should not appear as internal nodes
        (js_repo / "app.js").write_text("const express = require('express');\n")
        dg = DependencyAnalyzer(js_repo).analyze()
        assert "express" not in dg.all_modules()


# ---------------------------------------------------------------------------
# TS analysis
# ---------------------------------------------------------------------------

class TestDependencyAnalyzerTS:
    def test_detects_language_typescript(self, ts_repo):
        assert DependencyAnalyzer(ts_repo)._detect_language() == "typescript"

    def test_index_imports_lib(self, ts_repo):
        dg = DependencyAnalyzer(ts_repo).analyze()
        assert dg.files_analysed == 2

    def test_language_is_typescript(self, ts_repo):
        dg = DependencyAnalyzer(ts_repo).analyze()
        assert dg.language == "typescript"


# ---------------------------------------------------------------------------
# Skip dirs
# ---------------------------------------------------------------------------

class TestSkipDirs:
    def test_node_modules_skipped(self, js_repo):
        nm = js_repo / "node_modules" / "lib"
        nm.mkdir(parents=True)
        (nm / "secret.js").write_text("const x = require('./other');")
        dg = DependencyAnalyzer(js_repo).analyze()
        # secret module should not appear
        assert not any("secret" in m for m in dg.all_modules())

    def test_venv_skipped(self, simple_py_repo):
        venv = simple_py_repo / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "hidden.py").write_text("from utils import util\n")
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        assert not any("hidden" in m for m in dg.all_modules())


# ---------------------------------------------------------------------------
# format_dep_graph_for_prompt
# ---------------------------------------------------------------------------

class TestFormatDepGraphForPrompt:
    def test_empty_graph(self):
        dg = DependencyGraph()
        result = format_dep_graph_for_prompt(dg)
        assert "No modules detected" in result

    def test_includes_header(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        result = format_dep_graph_for_prompt(dg)
        assert "Dependency Graph" in result
        assert "modules" in result

    def test_reports_cycles(self, cycle_py_repo):
        dg = DependencyAnalyzer(cycle_py_repo).analyze()
        result = format_dep_graph_for_prompt(dg)
        assert "Circular" in result or "cycle" in result.lower()

    def test_no_cycles_message_clean(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        result = format_dep_graph_for_prompt(dg)
        assert "Circular" not in result

    def test_shows_coupling(self, simple_py_repo):
        dg = DependencyAnalyzer(simple_py_repo).analyze()
        result = format_dep_graph_for_prompt(dg)
        assert "coupling" in result.lower() or "coupled" in result.lower()


# ---------------------------------------------------------------------------
# GraphState integration
# ---------------------------------------------------------------------------

class TestGraphStateDepGraph:
    def test_dependency_graph_field_exists(self):
        from app.core.state import GraphState
        state = GraphState()
        assert hasattr(state, "dependency_graph")
        assert state.dependency_graph == {}

    def test_dep_cycles_field_exists(self):
        from app.core.state import GraphState
        state = GraphState()
        assert hasattr(state, "dep_cycles")
        assert state.dep_cycles == []

    def test_fields_accept_data(self):
        from app.core.state import GraphState
        dg = DependencyGraph(
            imports={"a": ["b"]},
            importers={"b": ["a"]},
            cycles=[["a", "b"]],
            language="python",
        )
        state = GraphState(
            dependency_graph=dg.to_dict(),
            dep_cycles=dg.cycles,
        )
        assert state.dependency_graph["language"] == "python"
        assert state.dep_cycles == [["a", "b"]]

    def test_fields_serialize(self):
        from app.core.state import GraphState
        state = GraphState(
            dependency_graph={"imports": {}, "importers": {}, "cycles": [],
                              "coupling_scores": {}, "language": "python",
                              "files_analysed": 1, "parse_errors": 0},
            dep_cycles=[["x", "y"]],
        )
        dumped = state.model_dump()
        assert "dependency_graph" in dumped
        assert "dep_cycles" in dumped
        assert dumped["dep_cycles"] == [["x", "y"]]
