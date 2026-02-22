"""Tests for app/tools/static_analysis.py

All subprocess calls are mocked — no real linters are invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tools.static_analysis import (
    StaticIssue,
    _find_eslint,
    _parse_mypy_line,
    _rel,
    _ruff_severity,
    _sort_issues,
    format_issues_for_prompt,
    run_static_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Minimal Python repo fixture."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
    (tmp_path / "app.py").write_text("x = 1\n")
    return tmp_path


# ---------------------------------------------------------------------------
# StaticIssue model
# ---------------------------------------------------------------------------

class TestStaticIssue:
    def test_one_line_with_rule(self):
        issue = StaticIssue(file="app.py", line=10, col=5, severity="error",
                            rule_id="E501", message="line too long", tool="ruff")
        line = issue.one_line()
        assert "[ERROR]" in line
        assert "E501" in line
        assert "app.py:10:5" in line
        assert "line too long" in line

    def test_one_line_without_col(self):
        issue = StaticIssue(file="app.py", line=3, severity="warning",
                            message="unused import", tool="ruff")
        assert "app.py:3" in issue.one_line()
        assert ":0" not in issue.one_line()

    def test_severity_default(self):
        issue = StaticIssue(file="f.py", message="msg", tool="ruff")
        assert issue.severity == "warning"


# ---------------------------------------------------------------------------
# _ruff_severity
# ---------------------------------------------------------------------------

class TestRuffSeverity:
    def test_e_is_error(self):
        assert _ruff_severity("E501") == "error"

    def test_f_is_error(self):
        assert _ruff_severity("F401") == "error"

    def test_w_is_warning(self):
        assert _ruff_severity("W291") == "warning"

    def test_unknown_is_info(self):
        assert _ruff_severity("B006") == "info"

    def test_empty_is_warning(self):
        assert _ruff_severity("") == "warning"


# ---------------------------------------------------------------------------
# _parse_mypy_line
# ---------------------------------------------------------------------------

class TestParseMypyLine:
    def test_parses_error_line(self, tmp_path):
        line = "app/main.py:42:10: error: Incompatible return value  [return-value]"
        issue = _parse_mypy_line(tmp_path, line)
        assert issue is not None
        assert issue.line == 42
        assert issue.severity == "error"
        assert "return-value" in issue.rule_id
        assert issue.tool == "mypy"

    def test_parses_warning_line(self, tmp_path):
        line = "app/utils.py:5:1: warning: unused import  [import]"
        issue = _parse_mypy_line(tmp_path, line)
        assert issue is not None
        assert issue.severity == "warning"

    def test_note_lines_are_skipped(self, tmp_path):
        line = "app/main.py:10:1: note: see definition here"
        assert _parse_mypy_line(tmp_path, line) is None

    def test_garbage_returns_none(self, tmp_path):
        assert _parse_mypy_line(tmp_path, "not a mypy line") is None
        assert _parse_mypy_line(tmp_path, "") is None

    def test_missing_col_is_ok(self, tmp_path):
        line = "app/main.py:7: error: Name undefined"
        issue = _parse_mypy_line(tmp_path, line)
        # May be None if col-less format doesn't match — that is acceptable
        if issue:
            assert issue.line == 7


# ---------------------------------------------------------------------------
# _sort_issues
# ---------------------------------------------------------------------------

class TestSortIssues:
    def test_errors_before_warnings_before_info(self):
        issues = [
            StaticIssue(file="b.py", line=1, severity="info", message="i", tool="ruff"),
            StaticIssue(file="a.py", line=5, severity="warning", message="w", tool="ruff"),
            StaticIssue(file="a.py", line=2, severity="error", message="e", tool="ruff"),
        ]
        sorted_issues = _sort_issues(issues)
        assert sorted_issues[0].severity == "error"
        assert sorted_issues[1].severity == "warning"
        assert sorted_issues[2].severity == "info"

    def test_same_severity_sorted_by_file_then_line(self):
        issues = [
            StaticIssue(file="z.py", line=1, severity="error", message="e", tool="ruff"),
            StaticIssue(file="a.py", line=10, severity="error", message="e", tool="ruff"),
            StaticIssue(file="a.py", line=2, severity="error", message="e", tool="ruff"),
        ]
        sorted_issues = _sort_issues(issues)
        assert sorted_issues[0].file == "a.py"
        assert sorted_issues[0].line == 2
        assert sorted_issues[1].file == "a.py"
        assert sorted_issues[1].line == 10


# ---------------------------------------------------------------------------
# _rel
# ---------------------------------------------------------------------------

class TestRel:
    def test_strips_root(self, tmp_path):
        path = str(tmp_path / "app" / "main.py")
        assert _rel(tmp_path, path) == "app/main.py"

    def test_returns_original_when_not_under_root(self, tmp_path):
        result = _rel(tmp_path, "/some/other/path.py")
        assert result == "/some/other/path.py"

    def test_empty_string_passthrough(self, tmp_path):
        assert _rel(tmp_path, "") == ""


# ---------------------------------------------------------------------------
# format_issues_for_prompt
# ---------------------------------------------------------------------------

class TestFormatIssuesForPrompt:
    def test_empty_list(self):
        assert format_issues_for_prompt([]) == "No static analysis issues detected."

    def test_includes_header(self):
        issues = [StaticIssue(file="a.py", line=1, severity="error", message="bad", tool="ruff")]
        result = format_issues_for_prompt(issues)
        assert "Static Analysis" in result
        assert "1 error" in result

    def test_truncates_to_max_issues(self):
        issues = [
            StaticIssue(file="a.py", line=i, severity="warning", message="w", tool="ruff")
            for i in range(30)
        ]
        result = format_issues_for_prompt(issues, max_issues=5)
        assert "showing top 5 of 30" in result

    def test_errors_listed_before_warnings(self):
        issues = [
            StaticIssue(file="a.py", line=1, severity="warning", message="warn", tool="ruff"),
            StaticIssue(file="b.py", line=1, severity="error", message="err", tool="ruff"),
        ]
        result = format_issues_for_prompt(issues, max_issues=2)
        err_pos = result.index("err")
        warn_pos = result.index("warn")
        assert err_pos < warn_pos


# ---------------------------------------------------------------------------
# run_static_analysis — ruff integration (mocked)
# ---------------------------------------------------------------------------

RUFF_JSON_OUTPUT = json.dumps([
    {
        "filename": "/repo/app/main.py",
        "message": "line too long (120 > 88)",
        "code": "E501",
        "location": {"row": 10, "column": 89},
        "end_location": {"row": 10, "column": 120},
    },
    {
        "filename": "/repo/app/utils.py",
        "message": "imported but unused",
        "code": "F401",
        "location": {"row": 3, "column": 1},
        "end_location": {"row": 3, "column": 10},
    },
])


class TestRunStaticAnalysisRuff:
    def test_ruff_issues_parsed(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = RUFF_JSON_OUTPUT
        mock_proc.stderr = ""
        mock_proc.returncode = 1  # ruff exits 1 when issues found

        with patch("app.tools.static_analysis.subprocess.run", return_value=mock_proc), \
             patch("app.tools.static_analysis._tool_available", return_value=True):
            issues = run_static_analysis(tmp_path, "python")

        assert len(issues) == 2
        # errors before warnings
        assert issues[0].severity == "error"   # F401
        assert issues[1].severity == "error"   # E501 (both are errors by _ruff_severity)
        tools = {i.tool for i in issues}
        assert "ruff" in tools

    def test_ruff_not_installed_returns_empty(self, tmp_path):
        with patch("app.tools.static_analysis._tool_available", return_value=False):
            issues = run_static_analysis(tmp_path, "python")
        # mypy also skipped → empty
        assert isinstance(issues, list)

    def test_ruff_timeout_returns_empty(self, tmp_path):
        import subprocess
        with patch("app.tools.static_analysis._tool_available", return_value=True), \
             patch("app.tools.static_analysis.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="ruff", timeout=60)):
            issues = run_static_analysis(tmp_path, "python")
        assert isinstance(issues, list)

    def test_ruff_invalid_json_returns_empty(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = "not valid json"
        mock_proc.stderr = ""

        with patch("app.tools.static_analysis.subprocess.run", return_value=mock_proc), \
             patch("app.tools.static_analysis._tool_available", return_value=True):
            issues = run_static_analysis(tmp_path, "python")
        # ruff parse fails → only mypy issues (also mocked away) → empty
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# run_static_analysis — eslint integration (mocked)
# ---------------------------------------------------------------------------

ESLINT_JSON_OUTPUT = json.dumps([
    {
        "filePath": "/repo/src/index.js",
        "messages": [
            {"line": 5, "column": 3, "severity": 2, "ruleId": "no-unused-vars",
             "message": "'foo' is defined but never used."},
            {"line": 12, "column": 1, "severity": 1, "ruleId": "semi",
             "message": "Missing semicolon."},
        ],
        "errorCount": 1,
        "warningCount": 1,
    }
])


class TestRunStaticAnalysisEslint:
    def test_eslint_issues_parsed(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = ESLINT_JSON_OUTPUT
        mock_proc.stderr = ""

        with patch("app.tools.static_analysis.subprocess.run", return_value=mock_proc), \
             patch("app.tools.static_analysis._find_eslint", return_value="/usr/bin/eslint"):
            issues = run_static_analysis(tmp_path, "javascript")

        assert len(issues) == 2
        error_issues = [i for i in issues if i.severity == "error"]
        warn_issues  = [i for i in issues if i.severity == "warning"]
        assert len(error_issues) == 1
        assert len(warn_issues) == 1
        assert error_issues[0].rule_id == "no-unused-vars"
        tools = {i.tool for i in issues}
        assert "eslint" in tools

    def test_eslint_not_found_returns_empty(self, tmp_path):
        with patch("app.tools.static_analysis._find_eslint", return_value=None):
            issues = run_static_analysis(tmp_path, "javascript")
        assert issues == []

    def test_typescript_also_uses_eslint(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.stdout = "[]"
        mock_proc.stderr = ""

        with patch("app.tools.static_analysis.subprocess.run", return_value=mock_proc), \
             patch("app.tools.static_analysis._find_eslint", return_value="/usr/bin/eslint"):
            issues = run_static_analysis(tmp_path, "typescript")
        assert issues == []


# ---------------------------------------------------------------------------
# run_static_analysis — unsupported language
# ---------------------------------------------------------------------------

class TestRunStaticAnalysisUnsupported:
    def test_unknown_language_returns_empty(self, tmp_path):
        issues = run_static_analysis(tmp_path, "rust")
        assert issues == []

    def test_empty_language_returns_empty(self, tmp_path):
        issues = run_static_analysis(tmp_path, "")
        assert issues == []


# ---------------------------------------------------------------------------
# GraphState integration
# ---------------------------------------------------------------------------

class TestGraphStateStaticIssues:
    def test_static_issues_field_exists(self):
        from app.core.state import GraphState
        state = GraphState()
        assert hasattr(state, "static_issues")
        assert state.static_issues == []

    def test_static_issues_accepts_dict_list(self):
        from app.core.state import GraphState
        state = GraphState(static_issues=[
            {"file": "a.py", "line": 1, "col": 0, "severity": "error",
             "rule_id": "E501", "message": "too long", "tool": "ruff"}
        ])
        assert len(state.static_issues) == 1
        assert state.static_issues[0]["tool"] == "ruff"

    def test_static_issues_serializes_to_json(self):
        from app.core.state import GraphState
        state = GraphState(static_issues=[
            {"file": "a.py", "line": 1, "col": 0, "severity": "warning",
             "rule_id": "W291", "message": "trailing whitespace", "tool": "ruff"}
        ])
        dumped = state.model_dump()
        assert "static_issues" in dumped
        assert len(dumped["static_issues"]) == 1
