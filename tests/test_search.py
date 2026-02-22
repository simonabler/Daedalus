"""Tests for safe repository search tool."""

from unittest.mock import patch


def test_search_in_repo_basic_match(tmp_path):
    from app.tools.search import search_in_repo

    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

    with patch("app.tools.search.get_settings") as ms:
        ms.return_value.target_repo_path = str(tmp_path)
        result = search_in_repo.invoke({"pattern": "def foo", "file_pattern": "*.py"})

    assert "a.py:1:" in result
    assert "def foo" in result


def test_search_in_repo_respects_max_hits(tmp_path):
    from app.tools.search import search_in_repo

    (tmp_path / "a.py").write_text("x = 1\nx = 2\nx = 3\n", encoding="utf-8")

    with patch("app.tools.search.get_settings") as ms:
        ms.return_value.target_repo_path = str(tmp_path)
        result = search_in_repo.invoke({"pattern": "x =", "file_pattern": "*.py", "max_hits": 2})

    assert "max hits 2 reached" in result


def test_search_in_repo_no_match(tmp_path):
    from app.tools.search import search_in_repo

    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    with patch("app.tools.search.get_settings") as ms:
        ms.return_value.target_repo_path = str(tmp_path)
        result = search_in_repo.invoke({"pattern": "does_not_exist", "file_pattern": "*.py"})

    assert "No matches found" in result
