"""Build & test runner tools for Python, Node/TS, and .NET projects.

These wrap the safe shell tool with project-type-aware defaults.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.tools.terminal import run_terminal

logger = get_logger("tools.build")


def _detect_project_type(repo_root: str) -> list[str]:
    """Detect which project types exist in the repo."""
    root = Path(repo_root)
    types = []
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "requirements.txt").exists():
        types.append("python")
    if (root / "package.json").exists():
        types.append("node")
    if list(root.rglob("*.csproj")) or list(root.rglob("*.sln")):
        types.append("dotnet")
    return types


@tool
def run_tests(project_type: str = "auto") -> str:
    """Run tests for the target project.

    project_type: 'python', 'node', 'dotnet', or 'auto' (detect).
    """
    settings = get_settings()
    root = settings.target_repo_path

    types = _detect_project_type(root) if project_type == "auto" else [project_type]

    if not types:
        return "WARNING: Could not detect project type. No tests run."

    results = []
    for ptype in types:
        if ptype == "python":
            r = run_terminal.invoke({"command": "python -m pytest --tb=short -q", "working_dir": "."})
        elif ptype == "node":
            r = run_terminal.invoke({"command": "npm test", "working_dir": "."})
        elif ptype == "dotnet":
            r = run_terminal.invoke({"command": "dotnet test --verbosity minimal", "working_dir": "."})
        else:
            r = f"Unknown project type: {ptype}"
        results.append(f"=== {ptype} tests ===\n{r}")

    logger.info("run_tests  | types=%s", types)
    return "\n\n".join(results)


@tool
def run_linter(project_type: str = "auto") -> str:
    """Run linters/formatters for the target project.

    project_type: 'python', 'node', 'dotnet', or 'auto'.
    """
    settings = get_settings()
    root = settings.target_repo_path

    types = _detect_project_type(root) if project_type == "auto" else [project_type]

    results = []
    for ptype in types:
        if ptype == "python":
            r = run_terminal.invoke({"command": "python -m ruff check . --fix", "working_dir": "."})
        elif ptype == "node":
            r = run_terminal.invoke({"command": "npm run lint", "working_dir": "."})
        elif ptype == "dotnet":
            r = run_terminal.invoke({"command": "dotnet build --verbosity minimal", "working_dir": "."})
        else:
            r = f"Unknown project type: {ptype}"
        results.append(f"=== {ptype} lint ===\n{r}")

    logger.info("run_linter | types=%s", types)
    return "\n\n".join(results)


@tool
def run_build(project_type: str = "auto") -> str:
    """Run build/compile for the target project.

    project_type: 'python', 'node', 'dotnet', or 'auto'.
    """
    settings = get_settings()
    root = settings.target_repo_path

    types = _detect_project_type(root) if project_type == "auto" else [project_type]

    results = []
    for ptype in types:
        if ptype == "python":
            cmd = "python -c \"import compileall,sys; ok=compileall.compile_dir('app', quiet=1); print('syntax check done'); sys.exit(0 if ok else 1)\""
            r = run_terminal.invoke({"command": cmd, "working_dir": "."})
        elif ptype == "node":
            r = run_terminal.invoke({"command": "npm run build", "working_dir": "."})
        elif ptype == "dotnet":
            r = run_terminal.invoke({"command": "dotnet build", "working_dir": "."})
        else:
            r = f"Unknown project type: {ptype}"
        results.append(f"=== {ptype} build ===\n{r}")

    logger.info("run_build  | types=%s", types)
    return "\n\n".join(results)


ALL_BUILD_TOOLS = [run_tests, run_linter, run_build]
