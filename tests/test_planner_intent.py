"""Tests for planner intent handling (new task, resume, question)."""

from app.core.state import GraphState, TodoItem, WorkflowPhase


def test_classify_request_intent_question():
    from app.core.nodes import _classify_request_intent

    assert _classify_request_intent("Was ist der aktuelle Status?") == "question_only"


def test_classify_request_intent_resume():
    from app.core.nodes import _classify_request_intent

    assert _classify_request_intent("Bitte workflow nach neustart fortsetzen") == "resume_workflow"


def test_parse_todo_for_resume():
    from app.core.nodes import _parse_todo_for_resume

    text = """
## Plan: Example

- [x] Item 1: Setup project
  - Type: coding
  - Owner: Coder A (Claude)
  - AC: Works
  - Verify: `pytest -q`

- [ ] Item 2: Update docs
  - Type: documentation
  - Owner: Documenter
""".strip()

    items = _parse_todo_for_resume(text)
    assert len(items) == 2
    assert items[0].status.name == "DONE"
    assert items[1].task_type == "documentation"
    assert items[1].assigned_agent == "documenter"


def test_planner_handles_question_without_coder(monkeypatch):
    from app.core import nodes

    monkeypatch.setattr(nodes, "_answer_question_directly", lambda state: "Antwort")
    state = GraphState(user_request="Was ist LangGraph?")

    result = nodes.planner_plan_node(state)

    assert result["phase"] == WorkflowPhase.COMPLETE
    assert result["input_intent"] == "question_only"
    assert result["stop_reason"] == "question_answered"
    assert result["todo_items"] == []


def test_planner_resume_uses_saved_workflow(monkeypatch):
    from app.core import nodes

    resume_state = {
        "todo_items": [TodoItem(id="item-001", description="Do thing")],
        "current_item_index": 0,
        "completed_items": 0,
        "branch_name": "feature/test",
        "phase": WorkflowPhase.CODING,
        "active_coder": "coder_a",
        "active_reviewer": "reviewer_b",
    }

    monkeypatch.setattr(nodes, "_resume_from_saved_todo", lambda state: resume_state)
    state = GraphState(user_request="resume workflow please")

    result = nodes.planner_plan_node(state)

    assert result["phase"] == WorkflowPhase.CODING
    assert result["input_intent"] == "resume_workflow"
    assert result["branch_name"] == "feature/test"


def test_resume_node_prefers_checkpoint(monkeypatch):
    from app.core import nodes

    restored = GraphState(
        user_request="Implement feature",
        phase=WorkflowPhase.CODING,
        current_item_index=0,
        todo_items=[TodoItem(id="item-001", description="Do thing")],
        resumed_from_checkpoint=True,
    )

    monkeypatch.setattr(nodes.checkpoint_manager, "load_checkpoint", lambda repo_root="": restored)
    state = GraphState(user_request="resume", repo_root="repo")

    result = nodes.resume_node(state)

    assert result["input_intent"] == "resume"
    assert result["resumed_from_checkpoint"] is True
    assert result["phase"] == WorkflowPhase.CODING
