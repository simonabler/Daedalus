"""Tests for app/analysis/call_graph.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.call_graph import (
    CallGraph,
    CallGraphAnalyzer,
    _extract_call_name,
    _js_function_ranges,
    format_call_graph_for_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def py_repo(tmp_path: Path) -> Path:
    """Minimal Python repo using the sample_module fixture."""
    # Copy sample_module into tmp_path so paths are portable
    src = FIXTURES_DIR / "sample_module.py"
    dst = tmp_path / "sample_module.py"
    dst.write_text(src.read_text())
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    return tmp_path


@pytest.fixture()
def multi_py_repo(tmp_path: Path) -> Path:
    """Repo with two Python files to test cross-file detection (same function names)."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    (tmp_path / "utils.py").write_text(
        "def helper():\n    return 42\n\ndef base():\n    return helper()\n"
    )
    (tmp_path / "main.py").write_text(
        "from utils import helper\n\ndef run():\n    return helper()\n"
    )
    return tmp_path


@pytest.fixture()
def js_repo(tmp_path: Path) -> Path:
    """Minimal JS repo with known call structure."""
    (tmp_path / "package.json").write_text('{"name":"test"}')
    (tmp_path / "index.js").write_text(
        "function greet(name) {\n"
        "    return format(name);\n"
        "}\n\n"
        "function format(s) {\n"
        "    return s.trim();\n"
        "}\n\n"
        "function main() {\n"
        "    greet('world');\n"
        "}\n"
    )
    return tmp_path


@pytest.fixture()
def ts_repo(tmp_path: Path) -> Path:
    """Minimal TS repo."""
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
    (tmp_path / "index.ts").write_text(
        "function validate(x: number): boolean {\n"
        "    return check(x);\n"
        "}\n\n"
        "function check(x: number): boolean {\n"
        "    return x > 0;\n"
        "}\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# CallGraph model
# ---------------------------------------------------------------------------

class TestCallGraphModel:
    def test_get_callers(self):
        cg = CallGraph(
            callers={"greet": ["main"], "format": ["greet"]},
            callees={"main": ["greet"], "greet": ["format"], "format": []},
            file_map={},
        )
        assert cg.get_callers("greet") == ["main"]
        assert cg.get_callers("format") == ["greet"]
        assert cg.get_callers("nonexistent") == []

    def test_get_callees(self):
        cg = CallGraph(
            callers={},
            callees={"main": ["greet", "log"], "greet": [], "log": []},
            file_map={},
        )
        assert sorted(cg.get_callees("main")) == ["greet", "log"]
        assert cg.get_callees("greet") == []

    def test_get_impact_radius_depth_1(self):
        # a → b → c → d
        cg = CallGraph(
            callers={"b": ["a"], "c": ["b"], "d": ["c"]},
            callees={"a": ["b"], "b": ["c"], "c": ["d"], "d": []},
            file_map={},
        )
        radius = cg.get_impact_radius("d", depth=1)
        assert radius == {"c"}

    def test_get_impact_radius_depth_2(self):
        cg = CallGraph(
            callers={"b": ["a"], "c": ["b"], "d": ["c"]},
            callees={"a": ["b"], "b": ["c"], "c": ["d"], "d": []},
            file_map={},
        )
        radius = cg.get_impact_radius("d", depth=2)
        assert radius == {"b", "c"}

    def test_get_impact_radius_not_includes_self(self):
        cg = CallGraph(
            callers={"foo": ["bar"]},
            callees={"bar": ["foo"], "foo": []},
            file_map={},
        )
        radius = cg.get_impact_radius("foo", depth=3)
        assert "foo" not in radius

    def test_most_called(self):
        cg = CallGraph(
            callers={"shared": ["a", "b", "c"], "rare": ["x"]},
            callees={"a": ["shared"], "b": ["shared"], "c": ["shared"], "x": ["rare"]},
            file_map={},
        )
        top = cg.most_called(n=1)
        assert top[0][0] == "shared"
        assert top[0][1] == 3

    def test_all_functions(self):
        cg = CallGraph(
            callers={"b": ["a"]},
            callees={"a": ["b"], "b": []},
            file_map={},
        )
        assert cg.all_functions() == {"a", "b"}

    def test_serialization_roundtrip(self):
        cg = CallGraph(
            callers={"foo": ["bar"]},
            callees={"bar": ["foo"], "foo": []},
            file_map={"foo": "main.py", "bar": "utils.py"},
            language="python",
            files_analysed=2,
            parse_errors=0,
        )
        data = cg.to_dict()
        restored = CallGraph.from_dict(data)
        assert restored.callers == cg.callers
        assert restored.callees == cg.callees
        assert restored.file_map == cg.file_map
        assert restored.language == cg.language


# ---------------------------------------------------------------------------
# Python AST analysis — sample_module.py fixture
# ---------------------------------------------------------------------------

class TestCallGraphAnalyzerPython:
    def test_detects_language_python(self, py_repo):
        analyzer = CallGraphAnalyzer(py_repo)
        assert analyzer._detect_language() == "python"

    def test_detects_direct_callees(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        # greet calls format_name
        assert "format_name" in cg.get_callees("greet")

    def test_detects_callers(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        # format_name is called by greet
        assert "greet" in cg.get_callers("format_name")

    def test_main_calls_greet_and_compute(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        callees = cg.get_callees("main")
        assert "greet" in callees
        assert "compute" in callees

    def test_compute_calls_add_and_multiply(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        callees = cg.get_callees("compute")
        assert "add" in callees
        assert "multiply" in callees

    def test_leaf_functions_have_no_callees(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        assert cg.get_callees("add") == []
        assert cg.get_callees("multiply") == []
        assert cg.get_callees("format_name") == []

    def test_orphan_has_no_callers(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        assert cg.get_callers("orphan") == []

    def test_impact_radius_of_format_name(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        # format_name → called by greet → called by main
        radius = cg.get_impact_radius("format_name", depth=2)
        assert "greet" in radius
        assert "main" in radius

    def test_all_functions_detected(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        expected = {"main", "greet", "compute", "add", "multiply", "format_name", "orphan"}
        assert expected.issubset(cg.all_functions())

    def test_files_analysed_count(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        assert cg.files_analysed >= 1

    def test_parse_errors_zero_for_valid_files(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        assert cg.parse_errors == 0

    def test_syntax_error_file_is_skipped(self, py_repo):
        (py_repo / "broken.py").write_text("def foo(\n  # syntax error")
        cg = CallGraphAnalyzer(py_repo).analyze()
        assert cg.parse_errors >= 1
        # Analysis still completes
        assert cg.files_analysed >= 1

    def test_graph_is_serializable(self, py_repo):
        import json
        cg = CallGraphAnalyzer(py_repo).analyze()
        data = cg.to_dict()
        serialized = json.dumps(data)
        assert len(serialized) > 10

    def test_self_analysis_does_not_crash(self):
        """CallGraphAnalyzer must be able to analyse the Daedalus repo itself."""
        import sys
        daedalus_root = Path(__file__).parent.parent
        cg = CallGraphAnalyzer(daedalus_root).analyze()
        assert cg.language == "python"
        assert cg.files_analysed > 0
        assert len(cg.all_functions()) > 0

    def test_multi_file_repo(self, multi_py_repo):
        cg = CallGraphAnalyzer(multi_py_repo).analyze()
        # helper is called by base (and run)
        assert "helper" in cg.all_functions()
        callers_of_helper = set(cg.get_callers("helper"))
        assert "base" in callers_of_helper or "run" in callers_of_helper


# ---------------------------------------------------------------------------
# JS analysis
# ---------------------------------------------------------------------------

class TestCallGraphAnalyzerJS:
    def test_detects_language_javascript(self, js_repo):
        analyzer = CallGraphAnalyzer(js_repo)
        assert analyzer._detect_language() == "javascript"

    def test_detects_functions(self, js_repo):
        cg = CallGraphAnalyzer(js_repo).analyze()
        assert "greet" in cg.all_functions()
        assert "format" in cg.all_functions()
        assert "main" in cg.all_functions()

    def test_greet_calls_format(self, js_repo):
        cg = CallGraphAnalyzer(js_repo).analyze()
        assert "format" in cg.get_callees("greet")

    def test_main_calls_greet(self, js_repo):
        cg = CallGraphAnalyzer(js_repo).analyze()
        assert "greet" in cg.get_callees("main")

    def test_language_is_javascript(self, js_repo):
        cg = CallGraphAnalyzer(js_repo).analyze()
        assert cg.language == "javascript"


# ---------------------------------------------------------------------------
# TS analysis
# ---------------------------------------------------------------------------

class TestCallGraphAnalyzerTS:
    def test_detects_language_typescript(self, ts_repo):
        analyzer = CallGraphAnalyzer(ts_repo)
        assert analyzer._detect_language() == "typescript"

    def test_validate_calls_check(self, ts_repo):
        cg = CallGraphAnalyzer(ts_repo).analyze()
        assert "validate" in cg.all_functions()
        assert "check" in cg.all_functions()
        assert "check" in cg.get_callees("validate")


# ---------------------------------------------------------------------------
# Skip dirs
# ---------------------------------------------------------------------------

class TestSkipDirs:
    def test_node_modules_skipped(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"test"}')
        nm = tmp_path / "node_modules" / "lib"
        nm.mkdir(parents=True)
        (nm / "helper.js").write_text("function hidden() {}")
        (tmp_path / "index.js").write_text("function visible() {}")

        cg = CallGraphAnalyzer(tmp_path).analyze()
        assert "visible" in cg.all_functions()
        assert "hidden" not in cg.all_functions()

    def test_venv_skipped(self, py_repo):
        venv = py_repo / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "secret.py").write_text("def should_not_appear(): pass")

        cg = CallGraphAnalyzer(py_repo).analyze()
        assert "should_not_appear" not in cg.all_functions()

    def test_daedalus_dir_skipped(self, py_repo):
        d = py_repo / ".daedalus" / "cache"
        d.mkdir(parents=True)
        (d / "cached.py").write_text("def cached_func(): pass")

        cg = CallGraphAnalyzer(py_repo).analyze()
        assert "cached_func" not in cg.all_functions()


# ---------------------------------------------------------------------------
# _extract_call_name helper
# ---------------------------------------------------------------------------

class TestExtractCallName:
    def test_simple_name(self):
        import ast
        node = ast.parse("foo()").body[0].value
        assert _extract_call_name(node) == "foo"

    def test_attribute_call(self):
        import ast
        node = ast.parse("obj.method()").body[0].value
        assert _extract_call_name(node) == "method"

    def test_complex_call_returns_none(self):
        import ast
        # foo()() — nested call, not a simple name
        node = ast.parse("foo()()").body[0].value
        result = _extract_call_name(node)
        # Either None or a string — must not raise
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# _js_function_ranges
# ---------------------------------------------------------------------------

class TestJsFunctionRanges:
    def test_finds_function_declaration(self):
        import re
        from app.analysis.call_graph import CallGraphAnalyzer as CGA
        source = "function foo() {\n  return 1;\n}\n"
        ranges = _js_function_ranges(source, CGA._FUNC_DEF_RE)
        names = [r[0] for r in ranges]
        assert "foo" in names

    def test_finds_const_arrow(self):
        import re
        from app.analysis.call_graph import CallGraphAnalyzer as CGA
        source = "const bar = (x) => {\n  return x;\n};\n"
        ranges = _js_function_ranges(source, CGA._FUNC_DEF_RE)
        names = [r[0] for r in ranges]
        assert "bar" in names


# ---------------------------------------------------------------------------
# format_call_graph_for_prompt
# ---------------------------------------------------------------------------

class TestFormatCallGraphForPrompt:
    def test_empty_graph(self):
        cg = CallGraph()
        result = format_call_graph_for_prompt(cg)
        assert "No functions detected" in result

    def test_includes_header(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        result = format_call_graph_for_prompt(cg)
        assert "Call Graph" in result
        assert "functions" in result

    def test_shows_most_called(self, py_repo):
        cg = CallGraphAnalyzer(py_repo).analyze()
        result = format_call_graph_for_prompt(cg)
        # greet and compute are both called by main — should appear
        assert "greet" in result or "compute" in result or "format_name" in result

    def test_respects_max_entries(self):
        cg = CallGraph(
            callers={f"fn{i}": [f"caller{i}"] for i in range(20)},
            callees={f"caller{i}": [f"fn{i}"] for i in range(20)},
            file_map={},
        )
        result = format_call_graph_for_prompt(cg, max_entries=3)
        # Should not list all 20 functions
        assert result.count("- `") <= 3


# ---------------------------------------------------------------------------
# GraphState integration
# ---------------------------------------------------------------------------

class TestGraphStateCallGraph:
    def test_call_graph_field_exists(self):
        from app.core.state import GraphState
        state = GraphState()
        assert hasattr(state, "call_graph")
        assert state.call_graph == {}

    def test_call_graph_accepts_dict(self):
        from app.core.state import GraphState
        cg_data = CallGraph(
            callers={"foo": ["bar"]},
            callees={"bar": ["foo"], "foo": []},
            file_map={"foo": "main.py"},
            language="python",
        ).to_dict()
        state = GraphState(call_graph=cg_data)
        assert state.call_graph["language"] == "python"

    def test_call_graph_serializes(self):
        from app.core.state import GraphState
        cg_data = {"callers": {}, "callees": {}, "file_map": {},
                   "language": "python", "files_analysed": 1, "parse_errors": 0}
        state = GraphState(call_graph=cg_data)
        dumped = state.model_dump()
        assert "call_graph" in dumped
        assert dumped["call_graph"]["language"] == "python"
