"""OS-aware terminal tool selection."""

from __future__ import annotations

import platform

from langchain_core.tools import tool

from app.tools.powershell import run_powershell
from app.tools.shell import run_shell

IS_WINDOWS = platform.system().lower().startswith("win")
ACTIVE_TERMINAL_KIND = "powershell" if IS_WINDOWS else "shell"

ACTIVE_TERMINAL_TOOL = run_powershell if IS_WINDOWS else run_shell
INACTIVE_TERMINAL_TOOL = run_shell if IS_WINDOWS else run_powershell


@tool
def run_terminal(command: str, working_dir: str = ".") -> str:
    """Execute a command in the active OS-specific terminal tool."""
    return ACTIVE_TERMINAL_TOOL.invoke({"command": command, "working_dir": working_dir})


ALL_TERMINAL_TOOLS = [run_terminal]
