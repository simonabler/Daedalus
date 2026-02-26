"""Tests for repository context loading before planning."""

from types import SimpleNamespace

from app.core.state import GraphState


def test_context_loader_reads_docs_and_marks_loaded(tmp_path, monkeypatch):
    from app.core import nodes

    (tmp_path / "README.md").write_text("# Test Project", encoding="utf-8")
    (tmp_path / "AGENT.md").write_text("# Agent Rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\ndependencies=['fastapi']\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    monkeypatch.setattr(
        "app.core.nodes.context_loader.get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path), max_output_chars=10000),
    )

    state = GraphState(user_request="Implement endpoint", repo_root=str(tmp_path))
    result = nodes.context_loader_node(state)

    assert result["context_loaded"] is True
    assert "=== README.md ===" in result["agent_instructions"]
    assert "=== AGENT.md ===" in result["agent_instructions"]
    language = result["repo_facts"].get("language")
    if language is None:
        language = result["repo_facts"].get("tech_stack", {}).get("language")
    assert language == "python"
    assert "README.md" in result["context_listing"]


def test_context_loader_skips_when_already_loaded(tmp_path, monkeypatch):
    from app.core import nodes

    monkeypatch.setattr(
        "app.core.nodes.context_loader.get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path), max_output_chars=10000),
    )

    state = GraphState(user_request="Implement endpoint", repo_root=str(tmp_path), context_loaded=True)
    result = nodes.context_loader_node(state)

    assert result["context_loaded"] is True
    assert result["input_intent"] == "code"


def test_context_loader_skips_agent_md_when_target_is_daedalus_root(monkeypatch):
    """AGENT.md must never be read when TARGET_REPO_PATH points to Daedalus itself.

    Daedalus' own AGENT.md is a build-spec for contributors, not a task instruction
    for the agent. When working on external repos, TARGET_REPO_PATH must be set to
    that repo — Daedalus' own root is never a valid target.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from app.core import nodes

    # Simulate: target repo IS the Daedalus root
    daedalus_root = Path(__file__).parent.parent.resolve()

    monkeypatch.setattr(
        "app.core.nodes.context_loader.get_settings",
        lambda: SimpleNamespace(
            target_repo_path=str(daedalus_root),
            max_output_chars=10000,
        ),
    )

    state = GraphState(user_request="Fix a bug", repo_root=str(daedalus_root))
    result = nodes.context_loader_node(state)

    # AGENT.md should NOT appear in agent_instructions
    instructions = result.get("agent_instructions", "")
    assert "=== AGENT.md ===" not in instructions, (
        "Daedalus' own AGENT.md must not be injected as task instructions. "
        "Set TARGET_REPO_PATH to a separate clone for self-improvement mode."
    )
    assert "=== AGENTS.md ===" not in instructions
    assert "=== docs/AGENT.md ===" not in instructions

    # README.md and other docs may still be read (they are informational, not spec)
    # context_loaded should still succeed
    assert result.get("context_loaded") is True


def test_context_loader_reads_agent_md_from_external_repo(tmp_path, monkeypatch):
    """When TARGET_REPO_PATH is an external repo (not Daedalus root), AGENT.md IS read."""
    from types import SimpleNamespace

    from app.core import nodes

    (tmp_path / "AGENT.md").write_text("# External Repo Agent Rules\nDo X, not Y.", encoding="utf-8")
    (tmp_path / "README.md").write_text("# My Project", encoding="utf-8")

    monkeypatch.setattr(
        "app.core.nodes.context_loader.get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path), max_output_chars=10000),
    )

    state = GraphState(user_request="Add feature", repo_root=str(tmp_path))
    result = nodes.context_loader_node(state)

    instructions = result.get("agent_instructions", "")
    assert "=== AGENT.md ===" in instructions, (
        "AGENT.md from an external target repo must be included in agent instructions."
    )
    assert "External Repo Agent Rules" in instructions
    assert result.get("context_loaded") is True