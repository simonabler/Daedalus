"""Repository/context analysis helper functions shared across nodes."""

from __future__ import annotations

from ._helpers import *

def _truncate_context_text(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-tail:]

def _build_context_listing(repo_path: Path, max_depth: int = 2, max_entries: int = 300) -> str:
    """Build a compact repo listing without using mutation-capable tools."""
    lines: list[str] = []
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}

    def _walk(path: Path, depth: int, prefix: str = "") -> None:
        if len(lines) >= max_entries or depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            return
        for entry in entries:
            if len(lines) >= max_entries:
                return
            if entry.name in skipped_dirs:
                continue
            if entry.name.startswith(".") and entry.name not in {".github", ".agents"}:
                continue
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                _walk(entry, depth + 1, prefix + "  ")
            else:
                lines.append(f"{prefix}{entry.name}")

    _walk(repo_path, depth=0)
    if len(lines) >= max_entries:
        lines.append("... [listing truncated]")
    return "\n".join(lines) if lines else "(empty)"

def _heuristic_analysis(repo_path: Path) -> dict:
    """Fallback file-based repository analysis."""
    facts: dict = {}

    if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
        facts["language"] = "python"
        facts["package_manager"] = "poetry" if (repo_path / "poetry.lock").exists() else "pip"
        if (repo_path / "pytest.ini").exists():
            facts["test_framework"] = "pytest"
            facts["test_command"] = "python -m pytest -q"
        else:
            facts["test_framework"] = "unittest"
            facts["test_command"] = "python -m unittest discover"
    elif (repo_path / "package.json").exists():
        facts["language"] = "javascript"
        facts["package_manager"] = "yarn" if (repo_path / "yarn.lock").exists() else "npm"
        facts["test_command"] = "npm test"
        try:
            parsed = json.loads((repo_path / "package.json").read_text(encoding="utf-8"))
            scripts = parsed.get("scripts", {})
            if isinstance(scripts, dict):
                facts["test_command"] = scripts.get("test", "npm test")
        except Exception:
            pass
    else:
        facts["language"] = "unknown"

    if (repo_path / ".github" / "workflows").exists():
        facts["ci_cd"] = "github_actions"
    elif (repo_path / ".gitlab-ci.yml").exists():
        facts["ci_cd"] = "gitlab_ci"

    facts["has_docker"] = (repo_path / "Dockerfile").exists()
    return facts

def _format_context_summary(repo_facts: dict) -> str:
    if not repo_facts:
        return "no facts detected"
    if repo_facts.get("error"):
        return f"analysis failed ({repo_facts['error']})"

    lines: list[str] = []
    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        language = tech_stack.get("language", "unknown")
        framework = tech_stack.get("framework")
        lines.append(f"language={language}")
        if framework:
            lines.append(f"framework={framework}")
    elif repo_facts.get("language"):
        lines.append(f"language={repo_facts['language']}")

    test_framework = repo_facts.get("test_framework")
    if isinstance(test_framework, dict):
        if test_framework.get("name"):
            lines.append(f"tests={test_framework['name']}")
        if test_framework.get("unit_test_command"):
            lines.append(f"test_cmd={test_framework['unit_test_command']}")
    elif isinstance(test_framework, str):
        lines.append(f"tests={test_framework}")
    elif repo_facts.get("test_command"):
        lines.append(f"test_cmd={repo_facts['test_command']}")

    conventions = repo_facts.get("conventions")
    if isinstance(conventions, dict):
        if conventions.get("linting_tool"):
            lines.append(f"lint={conventions['linting_tool']}")
        if conventions.get("formatting_tool"):
            lines.append(f"fmt={conventions['formatting_tool']}")

    ci_cd_setup = repo_facts.get("ci_cd_setup")
    if isinstance(ci_cd_setup, dict) and ci_cd_setup.get("platform"):
        lines.append(f"ci={ci_cd_setup['platform']}")
    elif repo_facts.get("ci_cd"):
        lines.append(f"ci={repo_facts['ci_cd']}")

    return ", ".join(lines) if lines else "context loaded"

def _extract_test_command(repo_facts: dict) -> str | None:
    test_framework = repo_facts.get("test_framework")
    if isinstance(test_framework, dict):
        cmd = test_framework.get("unit_test_command")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip()
    cmd = repo_facts.get("test_command")
    if isinstance(cmd, str) and cmd.strip():
        return cmd.strip()
    return None

def _extract_language(repo_facts: dict) -> str:
    """Return a normalised language string suitable for static analysis routing."""
    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        lang = tech_stack.get("language", "")
        if isinstance(lang, str):
            return lang.lower()
    lang = repo_facts.get("language", "")
    return lang.lower() if isinstance(lang, str) else ""

__all__ = [name for name in globals() if not name.startswith("__")]
