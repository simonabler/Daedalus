"""Tests for OS-aware terminal tool selection."""

import sys
from unittest.mock import patch


def test_active_terminal_kind_matches_platform():
    from app.tools.terminal import ACTIVE_TERMINAL_KIND

    expected = "powershell" if sys.platform == "win32" else "shell"
    assert ACTIVE_TERMINAL_KIND == expected


def test_terminal_tool_enablement_flags_match_platform():
    from app.tools.powershell import ALL_POWERSHELL_TOOLS, POWERSHELL_ENABLED
    from app.tools.shell import ALL_SHELL_TOOLS, SHELL_ENABLED

    assert (len(ALL_SHELL_TOOLS) > 0) == SHELL_ENABLED
    assert (len(ALL_POWERSHELL_TOOLS) > 0) == POWERSHELL_ENABLED


def test_inactive_tool_is_disabled_message(tmp_path):
    if sys.platform == "win32":
        from app.tools.shell import run_shell

        result = run_shell.invoke({"command": "echo test", "working_dir": "."})
        assert "DISABLED" in result
    else:
        from app.tools.powershell import run_powershell

        result = run_powershell.invoke({"command": "Write-Output test", "working_dir": "."})
        assert "DISABLED" in result


def test_run_terminal_routes_to_active_tool(tmp_path):
    from app.tools.terminal import run_terminal

    if sys.platform == "win32":
        with patch("app.tools.powershell.get_settings") as ms:
            ms.return_value.target_repo_path = str(tmp_path)
            ms.return_value.max_output_chars = 10000
            ms.return_value.shell_timeout_seconds = 10
            ms.return_value.git_author_name = "test"
            ms.return_value.git_author_email = "test@test"
            result = run_terminal.invoke({"command": "Write-Output hello", "working_dir": "."})
            assert "OK" in result
            assert "hello" in result
    else:
        with patch("app.tools.shell.get_settings") as ms:
            ms.return_value.target_repo_path = str(tmp_path)
            ms.return_value.max_output_chars = 10000
            ms.return_value.shell_timeout_seconds = 10
            ms.return_value.git_author_name = "test"
            ms.return_value.git_author_email = "test@test"
            result = run_terminal.invoke({"command": "echo hello", "working_dir": "."})
            assert "OK" in result
            assert "hello" in result
