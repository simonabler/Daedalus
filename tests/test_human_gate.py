"""Tests for human approval gate behavior."""

from types import SimpleNamespace

from app.core.state import GraphState, WorkflowPhase


def test_human_gate_requires_approval(monkeypatch):
    from app.core import nodes

    def fake_git_invoke(args):
        command = args.get("command", "")
        if "status --porcelain" in command:
            return " M app/core/nodes.py\n"
        if "diff --numstat" in command:
            return "12\t3\tapp/core/nodes.py\n"
        if "git diff" in command or command == "diff":
            return "diff --git a/app/core/nodes.py b/app/core/nodes.py\n+change\n"
        return ""

    monkeypatch.setattr("app.core.nodes.gates.git_command", SimpleNamespace(invoke=fake_git_invoke))

    state = GraphState(user_request="commit changes")
    result = nodes.human_gate_node(state)

    assert result["needs_human_approval"] is True
    assert result["phase"] == WorkflowPhase.WAITING_FOR_APPROVAL
    assert result["pending_approval"]["type"] == "commit"
    assert result["pending_approval"]["approved"] is False


def test_human_gate_allows_when_already_approved():
    from app.core.nodes import human_gate_node

    state = GraphState(
        user_request="commit changes",
        needs_human_approval=True,
        pending_approval={"approved": True, "type": "commit"},
    )
    result = human_gate_node(state)

    assert result["needs_human_approval"] is False
    assert result["phase"] == WorkflowPhase.COMMITTING
