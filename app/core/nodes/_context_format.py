"""Formatting helpers for repo context, intelligence summaries, and prompt enrichment."""
from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger("core.nodes._context_format")

def _truncate_context_text(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n\n... [truncated {len(text) - limit} chars] ...\n\n" + text[-tail:]



def _format_intelligence_summary_for_prompt(state: "GraphState") -> str:
    """Return a concise, consolidated intelligence block for any agent prompt.

    Combines static analysis, call graph, dependency graph and smell data
    into a single ``## Code Intelligence Summary`` section. Designed to be
    appended to any agent's prompt without repetition.

    Budget control:
    - planner / coder: full (max smells=8, issues=8, cg=8, dg=5)
    - reviewer:        reduced (smells=5, issues=5, cg=5, dg=3)
    - tester:          minimal (smells=3, issues=5)
    """
    parts: list[str] = []

    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=8))
    if state.call_graph:
        parts.append(_format_call_graph_for_prompt(state.call_graph, max_entries=8))
    if state.dependency_graph:
        parts.append(_format_dep_graph_for_prompt(state.dependency_graph, max_entries=5))
    if state.code_smells:
        parts.append(_format_code_smells_for_prompt(state.code_smells, max_smells=8))

    if not parts:
        return ""

    cached_note = ""
    if getattr(state, "intelligence_cached", False):
        key = getattr(state, "intelligence_cache_key", "")
        cached_note = f" (cached @ {key})" if key else " (cached)"

    header = f"## Code Intelligence Summary{cached_note}\n"
    return header + "\n\n".join(parts)


def _format_intelligence_summary_reviewer(state: "GraphState") -> str:
    """Reduced intelligence block for reviewer agents."""
    parts: list[str] = []
    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=5))
    if state.call_graph:
        parts.append(_format_call_graph_for_prompt(state.call_graph, max_entries=5))
    if state.dependency_graph:
        parts.append(_format_dep_graph_for_prompt(state.dependency_graph, max_entries=3))
    if state.code_smells:
        parts.append(_format_code_smells_for_prompt(state.code_smells, max_smells=5))
    if not parts:
        return ""
    cached_note = ""
    if getattr(state, "intelligence_cached", False):
        key = getattr(state, "intelligence_cache_key", "")
        cached_note = f" (cached @ {key})" if key else " (cached)"
    return f"## Code Intelligence Summary{cached_note}\n" + "\n\n".join(parts)


def _format_intelligence_summary_tester(state: "GraphState") -> str:
    """Minimal intelligence block for tester agent — smells + static errors only."""
    parts: list[str] = []
    if state.static_issues:
        parts.append(_format_static_issues_for_prompt(state.static_issues, max_issues=5))
    if state.code_smells:
        # Tester only needs errors, not info-level smells
        errors_only = [s for s in state.code_smells if s.get("severity") == "error"]
        if errors_only:
            parts.append(_format_code_smells_for_prompt(errors_only, max_smells=3))
    if not parts:
        return ""
    return "## Code Intelligence Summary\n" + "\n\n".join(parts)


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


def _format_call_graph_for_prompt(
    call_graph: dict,
    max_entries: int = 15,
) -> str:
    """Format a serialised CallGraph dict into a concise prompt section."""
    if not call_graph:
        return ""
    try:
        from app.analysis.call_graph import CallGraph, format_call_graph_for_prompt
        cg = CallGraph.from_dict(call_graph)
        return format_call_graph_for_prompt(cg, max_entries=max_entries)
    except Exception as exc:
        logger.debug("Could not format call graph for prompt: %s", exc)
        return ""


def _format_code_smells_for_prompt(
    code_smells: list[dict],
    max_smells: int = 10,
) -> str:
    """Format a list of serialised CodeSmell dicts into a compact prompt section."""
    if not code_smells:
        return ""
    try:
        from app.analysis.smell_detector import CodeSmell, format_smells_for_prompt
        smells = [CodeSmell(**s) for s in code_smells]
        return format_smells_for_prompt(smells, max_smells=max_smells)
    except Exception as exc:
        logger.debug("Could not format code smells for prompt: %s", exc)
        return ""


def _format_dep_graph_for_prompt(
    dependency_graph: dict,
    max_entries: int = 8,
) -> str:
    """Format a serialised DependencyGraph dict into a concise prompt section."""
    if not dependency_graph:
        return ""
    try:
        from app.analysis.dependency_graph import DependencyGraph, format_dep_graph_for_prompt
        dg = DependencyGraph.from_dict(dependency_graph)
        return format_dep_graph_for_prompt(dg, max_cycles=max_entries)
    except Exception as exc:
        logger.debug("Could not format dependency graph for prompt: %s", exc)
        return ""


def _format_static_issues_for_prompt(
    issues: list[dict],
    max_issues: int = 20,
) -> str:
    """Format serialised StaticIssue dicts into a concise prompt section."""
    if not issues:
        return ""

    errors   = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    infos    = [i for i in issues if i.get("severity") == "info"]

    selected = (errors + warnings + infos)[:max_issues]
    total = len(issues)
    shown = len(selected)

    header = (
        f"## Static Analysis — {len(errors)} error(s)"
        f", {len(warnings)} warning(s)"
        f", {len(infos)} info(s)"
    )
    if total > shown:
        header += f" (showing top {shown} of {total})"

    lines = [header, ""]
    for issue in selected:
        sev = issue.get("severity", "warning").upper()
        rule = f" [{issue['rule_id']}]" if issue.get("rule_id") else ""
        loc = f"{issue.get('file', '?')}:{issue.get('line', 0)}"
        msg = issue.get("message", "")
        tool = issue.get("tool", "")
        lines.append(f"- [{sev}]{rule} {loc} — {msg}  (tool: {tool})")

    return "\n".join(lines)


def _format_repo_context_for_prompt(repo_facts: dict) -> str:
    """Format structured context so planner/coder can follow repo conventions."""
    if not repo_facts:
        return "=== REPOSITORY CONTEXT ===\nNo repository facts detected."

    lines = ["=== REPOSITORY CONTEXT ===", "You MUST follow these detected conventions.", ""]

    tech_stack = repo_facts.get("tech_stack")
    if isinstance(tech_stack, dict):
        lines.append(f"Language: {tech_stack.get('language', 'unknown')}")
        framework = tech_stack.get("framework")
        if framework:
            version = tech_stack.get("framework_version")
            lines.append(f"Framework: {framework}{f' {version}' if version else ''}")
        manager = tech_stack.get("package_manager")
        if manager:
            lines.append(f"Package Manager: {manager}")
    elif repo_facts.get("language"):
        lines.append(f"Language: {repo_facts.get('language', 'unknown')}")

    test_framework = repo_facts.get("test_framework")
    test_command = _extract_test_command(repo_facts)
    if isinstance(test_framework, dict):
        lines.append("")
        lines.append(f"Test Framework: {test_framework.get('name', 'unknown')}")
    elif isinstance(test_framework, str):
        lines.append("")
        lines.append(f"Test Framework: {test_framework}")
    if test_command:
        lines.append(f"Test Command: {test_command}")
        lines.append("CRITICAL: Prefer this test command for verification.")

    conventions = repo_facts.get("conventions")
    if isinstance(conventions, dict):
        lines.append("")
        lines.append("Code Style:")
        if conventions.get("linting_tool"):
            lines.append(f"- Linting: {conventions['linting_tool']}")
        if conventions.get("formatting_tool"):
            lines.append(f"- Formatting: {conventions['formatting_tool']}")
        if conventions.get("max_line_length"):
            lines.append(f"- Max line length: {conventions['max_line_length']}")
        if conventions.get("function_naming"):
            lines.append(f"- Function naming: {conventions['function_naming']}")
        if conventions.get("class_naming"):
            lines.append(f"- Class naming: {conventions['class_naming']}")

    architecture = repo_facts.get("architecture")
    if isinstance(architecture, dict):
        lines.append("")
        lines.append(f"Architecture: {architecture.get('type', 'unknown')}")
        layers = architecture.get("layers")
        if isinstance(layers, list) and layers:
            lines.append(f"Layers: {', '.join(str(layer) for layer in layers)}")

    ci_cd_setup = repo_facts.get("ci_cd_setup")
    if isinstance(ci_cd_setup, dict) and ci_cd_setup.get("platform"):
        lines.append("")
        lines.append(f"CI/CD: {ci_cd_setup['platform']}")
        lines.append("Be careful with CI/CD changes.")
    elif repo_facts.get("ci_cd"):
        lines.append("")
        lines.append(f"CI/CD: {repo_facts.get('ci_cd')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NODE: status / research / resume (minimal non-coding branches)
# ---------------------------------------------------------------------------

