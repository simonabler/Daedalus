"""Tests for app/analysis/smell_detector.py — all 8 rules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.analysis.smell_detector import (
    CodeSmell,
    SmellDetector,
    _cyclomatic_complexity,
    _max_nesting_depth,
    _rule_dead_code,
    _rule_deeply_nested,
    _rule_duplicate_code,
    _rule_god_class,
    _rule_high_complexity,
    _rule_long_function,
    _rule_long_param_list,
    _rule_magic_numbers,
    _similarity,
    format_smells_for_prompt,
    sort_smells,
)
import ast

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test"')
    for name, src in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return tmp_path


# ---------------------------------------------------------------------------
# CodeSmell model
# ---------------------------------------------------------------------------

class TestCodeSmellModel:
    def test_one_line_format(self):
        s = CodeSmell(file="a.py", line=10, smell_type="LongFunction",
                      severity="error", description="too long", suggestion="split it")
        line = s.one_line()
        assert "[ERROR]" in line
        assert "LongFunction" in line
        assert "a.py:10" in line
        assert "split it" in line

    def test_one_line_no_suggestion(self):
        s = CodeSmell(file="b.py", line=5, smell_type="MagicNumber",
                      severity="info", description="magic 42")
        assert "| FIX:" not in s.one_line()

    def test_severity_default(self):
        s = CodeSmell(file="x.py", line=1, smell_type="T", description="d")
        assert s.severity == "warning"


class TestSortSmells:
    def test_errors_first(self):
        smells = [
            CodeSmell(file="a", line=1, smell_type="T", severity="info", description="i"),
            CodeSmell(file="a", line=2, smell_type="T", severity="error", description="e"),
            CodeSmell(file="a", line=3, smell_type="T", severity="warning", description="w"),
        ]
        sorted_ = sort_smells(smells)
        assert sorted_[0].severity == "error"
        assert sorted_[1].severity == "warning"
        assert sorted_[2].severity == "info"


# ---------------------------------------------------------------------------
# Rule 1 — Long Function
# ---------------------------------------------------------------------------

class TestRuleLongFunction:
    def _make_tree(self, n_lines: int) -> tuple[ast.Module, list[str]]:
        body = "\n".join(f"    x_{i} = {i}" for i in range(n_lines))
        src = f"def big(x):\n{body}\n    return x\n"
        return ast.parse(src), src.splitlines()

    def test_no_smell_under_50(self):
        tree, lines = self._make_tree(40)
        smells = _rule_long_function("f.py", tree, lines)
        assert smells == []

    def test_warning_51_lines(self):
        tree, lines = self._make_tree(51)
        smells = _rule_long_function("f.py", tree, lines)
        assert any(s.severity == "warning" and s.smell_type == "LongFunction" for s in smells)

    def test_error_over_100(self):
        tree, lines = self._make_tree(105)
        smells = _rule_long_function("f.py", tree, lines)
        assert any(s.severity == "error" and s.smell_type == "LongFunction" for s in smells)

    def test_fixture_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"big.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "LongFunction" for s in smells)


# ---------------------------------------------------------------------------
# Rule 2 — High Complexity
# ---------------------------------------------------------------------------

class TestRuleHighComplexity:
    def test_simple_func_low_complexity(self):
        tree = parse("def f(x): return x + 1\n")
        smells = _rule_high_complexity("f.py", tree)
        assert smells == []

    def test_complexity_counting(self):
        src = """
        def f(a, b, c):
            if a:
                if b:
                    pass
                elif c:
                    pass
            for i in range(10):
                if i > 5:
                    pass
            while a and b:
                a -= 1
            return a
        """
        func = parse(src).body[0]
        cc = _cyclomatic_complexity(func)
        assert cc > 5

    def test_high_complexity_flagged(self, tmp_path):
        repo = make_repo(tmp_path, {"smelly.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "HighComplexity" for s in smells)

    def test_boolop_adds_to_complexity(self):
        src = "def f(a, b, c): return a and b and c\n"
        func = parse(src).body[0]
        cc = _cyclomatic_complexity(func)
        assert cc >= 3  # 1 + 2 boolean values


# ---------------------------------------------------------------------------
# Rule 3 — God Class
# ---------------------------------------------------------------------------

class TestRuleGodClass:
    def _make_class(self, n_methods: int) -> ast.Module:
        methods = "\n".join(f"    def m{i}(self): pass" for i in range(n_methods))
        return parse(f"class C:\n{methods}\n")

    def test_small_class_no_smell(self):
        tree = self._make_class(5)
        smells = _rule_god_class("f.py", tree, [])
        assert smells == []

    def test_21_methods_flagged(self):
        tree = self._make_class(21)
        smells = _rule_god_class("f.py", tree, [])
        assert any(s.smell_type == "GodClass" and s.severity == "error" for s in smells)

    def test_fixture_god_class_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "GodClass" for s in smells)


# ---------------------------------------------------------------------------
# Rule 4 — Long Parameter List
# ---------------------------------------------------------------------------

class TestRuleLongParamList:
    def test_5_params_no_smell(self):
        tree = parse("def f(a, b, c, d, e): pass\n")
        smells = _rule_long_param_list("f.py", tree)
        assert smells == []

    def test_6_params_flagged(self):
        tree = parse("def f(a, b, c, d, e, g): pass\n")
        smells = _rule_long_param_list("f.py", tree)
        assert any(s.smell_type == "LongParameterList" for s in smells)

    def test_self_not_counted(self):
        tree = parse("class C:\n    def f(self, a, b, c, d, e): pass\n")
        smells = _rule_long_param_list("f.py", tree)
        assert smells == []

    def test_fixture_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "LongParameterList" for s in smells)


# ---------------------------------------------------------------------------
# Rule 5 — Duplicate Code
# ---------------------------------------------------------------------------

class TestRuleDuplicateCode:
    def test_identical_functions_flagged(self, tmp_path):
        src = """
        def alpha(items):
            result = []
            for item in items:
                if item > 0:
                    value = item * 2
                    result.append(value)
                else:
                    value = item * -1
                    result.append(value)
            total = sum(result)
            return total

        def beta(elements):
            result = []
            for item in elements:
                if item > 0:
                    value = item * 2
                    result.append(value)
                else:
                    value = item * -1
                    result.append(value)
            total = sum(result)
            return total
        """
        repo = make_repo(tmp_path, {"dup.py": src})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "DuplicateCode" for s in smells)

    def test_different_functions_not_flagged(self, tmp_path):
        src = """
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b
        """
        repo = make_repo(tmp_path, {"clean.py": src})
        smells = SmellDetector(repo).detect()
        assert not any(s.smell_type == "DuplicateCode" for s in smells)

    def test_similarity_function(self):
        a = ["x", "y", "z", "w"]
        b = ["x", "y", "z", "q"]
        sim = _similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_similarity_identical(self):
        a = ["x", "y", "z"]
        assert _similarity(a, a) == 1.0

    def test_similarity_empty(self):
        assert _similarity([], ["x"]) == 0.0

    def test_fixture_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "DuplicateCode" for s in smells)


# ---------------------------------------------------------------------------
# Rule 6 — Dead Code
# ---------------------------------------------------------------------------

class TestRuleDeadCode:
    def test_no_call_graph_returns_empty(self):
        tree = parse("def orphan(): pass\n")
        smells = _rule_dead_code("f.py", tree, {})
        assert smells == []

    def test_called_function_not_flagged(self):
        tree = parse("def used(): pass\n")
        cg = {"callers": {"used": ["caller"]}, "callees": {"used": []}}
        smells = _rule_dead_code("f.py", tree, cg)
        assert smells == []

    def test_uncalled_function_flagged(self):
        tree = parse("def orphan(): pass\n")
        cg = {"callers": {}, "callees": {"orphan": []}}
        smells = _rule_dead_code("f.py", tree, cg)
        assert any(s.smell_type == "DeadCode" for s in smells)

    def test_dunder_methods_skipped(self):
        tree = parse("def __init__(self): pass\n")
        cg = {"callers": {}, "callees": {"__init__": []}}
        smells = _rule_dead_code("f.py", tree, cg)
        assert smells == []

    def test_test_functions_skipped(self):
        tree = parse("def test_something(): pass\n")
        cg = {"callers": {}, "callees": {"test_something": []}}
        smells = _rule_dead_code("f.py", tree, cg)
        assert smells == []

    def test_main_skipped(self):
        tree = parse("def main(): pass\n")
        cg = {"callers": {}, "callees": {"main": []}}
        smells = _rule_dead_code("f.py", tree, cg)
        assert smells == []


# ---------------------------------------------------------------------------
# Rule 7 — Magic Numbers
# ---------------------------------------------------------------------------

class TestRuleMagicNumbers:
    def test_whitelisted_values_skipped(self):
        tree = parse("def f(x):\n    return x + 0 + 1 + 2\n")
        smells = _rule_magic_numbers("f.py", tree)
        assert smells == []

    def test_arbitrary_literal_flagged(self):
        tree = parse("def f(x):\n    return x * 42\n")
        smells = _rule_magic_numbers("f.py", tree)
        assert any(s.smell_type == "MagicNumber" for s in smells)

    def test_module_level_constant_not_flagged(self):
        # Module-level assignments are not inside functions — should not be flagged
        tree = parse("LIMIT = 42\ndef f(x): return x\n")
        smells = _rule_magic_numbers("f.py", tree)
        assert smells == []

    def test_fixture_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "MagicNumber" for s in smells)

    def test_string_not_flagged(self):
        tree = parse("def f(): return 'hello'\n")
        smells = _rule_magic_numbers("f.py", tree)
        assert smells == []


# ---------------------------------------------------------------------------
# Rule 8 — Deeply Nested
# ---------------------------------------------------------------------------

class TestRuleDeeplyNested:
    def test_flat_func_no_smell(self):
        tree = parse("def f(x):\n    if x:\n        return x\n    return 0\n")
        smells = _rule_deeply_nested("f.py", tree)
        assert smells == []

    def test_depth_5_flagged(self):
        src = """
        def deep(data):
            for a in data:
                if a:
                    for b in a:
                        if b:
                            for c in b:
                                pass
        """
        tree = parse(src)
        smells = _rule_deeply_nested("f.py", tree)
        assert any(s.smell_type == "DeeplyNested" for s in smells)

    def test_max_nesting_depth_trivial(self):
        tree = parse("def f(): pass\n")
        func = tree.body[0]
        assert _max_nesting_depth(func) == 0

    def test_max_nesting_depth_counted(self):
        src = """
        def f(x):
            if x:
                for i in x:
                    if i:
                        pass
        """
        func = parse(src).body[0]
        depth = _max_nesting_depth(func)
        assert depth == 3

    def test_fixture_detected(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        assert any(s.smell_type == "DeeplyNested" for s in smells)


# ---------------------------------------------------------------------------
# SmellDetector — integration
# ---------------------------------------------------------------------------

class TestSmellDetectorIntegration:
    def test_clean_module_no_smells(self, tmp_path):
        src = """
        def add(a: int, b: int) -> int:
            return a + b

        def greet(name: str) -> str:
            return f"Hello, {name}!"
        """
        repo = make_repo(tmp_path, {"clean.py": src})
        smells = SmellDetector(repo).detect()
        # Clean module should produce few or no smells
        assert all(s.smell_type not in {"LongFunction", "HighComplexity", "GodClass"} for s in smells)

    def test_smelly_module_hits_all_rules(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        smell_types = {s.smell_type for s in smells}
        # All rules except DeadCode (needs call graph) should fire on the fixture
        assert "LongFunction" in smell_types
        assert "HighComplexity" in smell_types
        assert "GodClass" in smell_types
        assert "LongParameterList" in smell_types
        assert "DuplicateCode" in smell_types
        assert "MagicNumber" in smell_types
        assert "DeeplyNested" in smell_types

    def test_dead_code_with_call_graph(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        cg = {"callers": {}, "callees": {"orphan_func": []}}
        smells = SmellDetector(repo, call_graph=cg).detect()
        assert any(s.smell_type == "DeadCode" for s in smells)

    def test_skip_dirs_respected(self, tmp_path):
        repo = make_repo(tmp_path, {"clean.py": "def f(): pass\n"})
        venv = repo / ".venv" / "lib"
        venv.mkdir(parents=True)
        body = "\n".join(f"    x{i} = {i}" for i in range(110))
        (venv / "hidden.py").write_text(f"def hidden():\n{body}\n")
        smells = SmellDetector(repo).detect()
        assert not any("hidden" in s.file for s in smells)

    def test_syntax_error_file_skipped(self, tmp_path):
        repo = make_repo(tmp_path, {
            "good.py": "def f(): pass\n",
            "broken.py": "def bad(\n  # syntax error",
        })
        smells = SmellDetector(repo).detect()
        # Should not raise — broken file is silently skipped
        assert not any("broken" in s.file for s in smells)

    def test_results_sorted_errors_first(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        severities = [s.severity for s in smells]
        sev_order = {"error": 0, "warning": 1, "info": 2}
        assert severities == sorted(severities, key=lambda s: sev_order[s])

    def test_self_analysis_does_not_crash(self):
        root = Path(__file__).parent.parent
        smells = SmellDetector(root).detect()
        assert isinstance(smells, list)


# ---------------------------------------------------------------------------
# format_smells_for_prompt
# ---------------------------------------------------------------------------

class TestFormatSmellsForPrompt:
    def test_empty_list(self):
        result = format_smells_for_prompt([])
        assert "No smells detected" in result

    def test_includes_header(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        result = format_smells_for_prompt(smells)
        assert "Code Smells" in result

    def test_respects_max_smells(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        result = format_smells_for_prompt(smells, max_smells=3)
        # Should mention truncation
        lines_with_dash = [l for l in result.splitlines() if l.startswith("- ")]
        assert len(lines_with_dash) <= 3

    def test_shows_counts(self, tmp_path):
        repo = make_repo(tmp_path, {"s.py": (FIXTURES / "smelly_module.py").read_text()})
        smells = SmellDetector(repo).detect()
        result = format_smells_for_prompt(smells)
        assert "error" in result
        assert "warning" in result


# ---------------------------------------------------------------------------
# GraphState integration
# ---------------------------------------------------------------------------

class TestGraphStateCodeSmells:
    def test_code_smells_field_exists(self):
        from app.core.state import GraphState
        state = GraphState()
        assert hasattr(state, "code_smells")
        assert state.code_smells == []

    def test_code_smells_accepts_list_of_dicts(self):
        from app.core.state import GraphState
        smell = CodeSmell(
            file="x.py", line=10, smell_type="LongFunction",
            severity="error", description="too long"
        )
        state = GraphState(code_smells=[smell.model_dump()])
        assert len(state.code_smells) == 1
        assert state.code_smells[0]["smell_type"] == "LongFunction"

    def test_code_smells_serializes(self):
        from app.core.state import GraphState
        smell = CodeSmell(file="y.py", line=5, smell_type="MagicNumber",
                          severity="info", description="magic 42")
        state = GraphState(code_smells=[smell.model_dump()])
        dumped = state.model_dump()
        assert "code_smells" in dumped
        assert dumped["code_smells"][0]["smell_type"] == "MagicNumber"
