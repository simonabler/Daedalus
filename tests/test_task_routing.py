"""Tests for history-aware task routing."""

from unittest.mock import patch


def test_classify_task_type():
    from app.core.task_routing import classify_task_type

    assert classify_task_type("Update README docs") == "documentation"
    assert classify_task_type("Add unit tests for API") == "testing"
    assert classify_task_type("Configure CI pipeline") == "ops"
    assert classify_task_type("Implement login endpoint") == "coding"


def test_is_programming_request():
    from app.core.task_routing import is_programming_request

    assert is_programming_request("Implement JWT auth") is True
    assert is_programming_request("Write project summary document") is False


def test_routing_history_persists(tmp_path):
    from app.core.task_routing import load_routing_history, record_agent_outcome

    record_agent_outcome(str(tmp_path), "coding", "coder_a", success=True)
    record_agent_outcome(str(tmp_path), "coding", "coder_a", success=False)

    history = load_routing_history(str(tmp_path))
    stats = history["task_types"]["coding"]["coder_a"]
    assert stats["trials"] == 2
    assert stats["wins"] == 1
    assert stats["alpha"] > 1
    assert stats["beta"] > 1


def test_select_agent_thompson_returns_candidate(tmp_path):
    from app.core.task_routing import select_agent_thompson

    with patch("app.core.task_routing.random.betavariate", side_effect=[0.2, 0.8]):
        agent, _history = select_agent_thompson(str(tmp_path), "coding", ["coder_a", "coder_b"])
    assert agent == "coder_b"
