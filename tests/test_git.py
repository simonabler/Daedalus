"""Tests for the safe git tool — command validation and blocklist."""

import subprocess
from unittest.mock import patch

import pytest


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo for testing."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True)

    with patch("app.tools.git.get_settings") as ms:
        ms.return_value.target_repo_path = str(tmp_path)
        ms.return_value.max_output_chars = 10000
        ms.return_value.git_author_name = "test"
        ms.return_value.git_author_email = "test@test"
        yield tmp_path


class TestCommandValidation:
    def test_merge_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git merge some-branch"})
        assert "BLOCKED" in result

    def test_rebase_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git rebase main"})
        assert "BLOCKED" in result

    def test_reset_hard_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git reset --hard HEAD~1"})
        assert "BLOCKED" in result

    def test_force_push_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git push --force origin main"})
        assert "BLOCKED" in result

    def test_clean_fd_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git clean -fd"})
        assert "BLOCKED" in result

    def test_status_allowed(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git status"})
        assert "OK" in result

    def test_diff_allowed(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git diff"})
        assert "OK" in result

    def test_log_allowed(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git log --oneline -5"})
        assert "OK" in result
        assert "init" in result

    def test_shell_operator_and_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git log --oneline && echo PWN"})
        assert "BLOCKED" in result

    def test_shell_operator_pipe_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git diff | cat"})
        assert "BLOCKED" in result

    def test_unknown_subcommand_blocked(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "git bisect start"})
        assert "BLOCKED" in result or "not allowed" in result

    def test_non_git_command_rejected(self, git_repo):
        from app.tools.git import git_command

        result = git_command.invoke({"command": "ls -la"})
        assert "ERROR" in result or "BLOCKED" in result


class TestGitCreateBranch:
    def test_create_feature_branch(self, git_repo):
        from app.tools.git import git_create_branch

        result = git_create_branch.invoke({"branch_name": "test-feature"})
        assert "OK" in result

        # Verify we're on the new branch
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(git_repo), capture_output=True, text=True,
        )
        assert "feature/test-feature" in out.stdout

    def test_auto_prefix(self, git_repo):
        from app.tools.git import git_create_branch

        result = git_create_branch.invoke({"branch_name": "my-work"})
        assert "OK" in result


class TestGitStatus:
    def test_shows_branch_and_status(self, git_repo):
        from app.tools.git import git_status

        result = git_status.invoke({})
        assert "Branch" in result
