"""Tests for the safe filesystem tool."""

from unittest.mock import patch

import pytest

# We need to mock settings before importing the tools
_test_repo_dir = None


@pytest.fixture(autouse=True)
def setup_test_repo(tmp_path):
    """Create a temporary repo directory for each test."""
    global _test_repo_dir
    _test_repo_dir = tmp_path
    (tmp_path / "existing.txt").write_text("hello world")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("print('hello')")

    from app.core.active_repo import set_repo_root, clear_repo_root
    set_repo_root(str(tmp_path))
    with patch("app.tools.filesystem.get_settings") as mock_settings:
        mock_settings.return_value.target_repo_path = str(tmp_path)
        mock_settings.return_value.max_output_chars = 10000
        yield tmp_path
    clear_repo_root()


class TestResolvesSafe:
    def test_normal_path(self, setup_test_repo):
        from app.tools.filesystem import _resolve_safe
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            result = _resolve_safe("existing.txt")
            assert result == setup_test_repo / "existing.txt"

    def test_path_escape_blocked(self, setup_test_repo):
        from app.tools.filesystem import PathEscapeError, _resolve_safe
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            with pytest.raises(PathEscapeError):
                _resolve_safe("../../etc/passwd")

    def test_absolute_path_escape(self, setup_test_repo):
        from app.tools.filesystem import PathEscapeError, _resolve_safe
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            with pytest.raises(PathEscapeError):
                _resolve_safe("/etc/passwd")

    def test_prefix_bypass_escape_blocked(self, setup_test_repo):
        from app.tools.filesystem import PathEscapeError, _resolve_safe

        sibling = setup_test_repo.parent / f"{setup_test_repo.name}2"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("secret")

        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            with pytest.raises(PathEscapeError):
                _resolve_safe(f"../{sibling.name}/secret.txt")


class TestReadFile:
    def test_read_existing(self, setup_test_repo):
        from app.tools.filesystem import read_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = read_file.invoke({"path": "existing.txt"})
            assert "hello world" in result

    def test_read_nonexistent(self, setup_test_repo):
        from app.tools.filesystem import read_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = read_file.invoke({"path": "nope.txt"})
            assert "ERROR" in result


class TestWriteFile:
    def test_write_new_file(self, setup_test_repo):
        from app.tools.filesystem import write_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = write_file.invoke({"path": "new.txt", "content": "new content"})
            assert "OK" in result
            assert (setup_test_repo / "new.txt").read_text() == "new content"

    def test_write_creates_dirs(self, setup_test_repo):
        from app.tools.filesystem import write_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = write_file.invoke({"path": "a/b/c.txt", "content": "deep"})
            assert "OK" in result
            assert (setup_test_repo / "a" / "b" / "c.txt").exists()


class TestPatchFile:
    def test_patch_success(self, setup_test_repo):
        from app.tools.filesystem import patch_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = patch_file.invoke({"path": "existing.txt", "old": "world", "new": "there"})
            assert "OK" in result
            assert (setup_test_repo / "existing.txt").read_text() == "hello there"

    def test_patch_not_found(self, setup_test_repo):
        from app.tools.filesystem import patch_file
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = patch_file.invoke({"path": "existing.txt", "old": "xyz", "new": "abc"})
            assert "ERROR" in result


class TestListDirectory:
    def test_list_root(self, setup_test_repo):
        from app.tools.filesystem import list_directory
        with patch("app.tools.filesystem.get_settings") as ms:
            ms.return_value.target_repo_path = str(setup_test_repo)
            ms.return_value.max_output_chars = 10000
            result = list_directory.invoke({"path": "."})
            assert "existing.txt" in result
            assert "subdir/" in result
