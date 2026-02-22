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
        nodes,
        "get_settings",
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
        nodes,
        "get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path), max_output_chars=10000),
    )

    state = GraphState(user_request="Implement endpoint", repo_root=str(tmp_path), context_loaded=True)
    result = nodes.context_loader_node(state)

    assert result["context_loaded"] is True
    assert result["input_intent"] == "code"
