"""Tests for configuration and graph state."""

from unittest.mock import patch


class TestSettings:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            from app.core.config import Settings
            s = Settings(_env_file=None)
            assert s.web_port == 8420
            assert s.max_iterations_per_item == 5
            assert s.max_rework_cycles_per_item == 3
            assert s.planner_model == "gpt-4o-mini"

    def test_telegram_ids_parsing(self):
        from app.core.config import Settings
        s = Settings(telegram_allowed_user_ids="123,456,789")
        assert s.allowed_telegram_ids == [123, 456, 789]

    def test_telegram_ids_empty(self):
        from app.core.config import Settings
        s = Settings(telegram_allowed_user_ids="")
        assert s.allowed_telegram_ids == []


class TestGraphState:
    def test_initial_state(self):
        from app.core.state import GraphState, WorkflowPhase
        state = GraphState()
        assert state.phase == WorkflowPhase.IDLE
        assert state.current_item is None
        assert state.completed_items == 0
        assert state.context_loaded is False
        assert state.repo_facts == {}
        assert state.agent_instructions == ""

    def test_current_item(self):
        from app.core.state import GraphState, TodoItem
        items = [
            TodoItem(id="item-001", description="First task"),
            TodoItem(id="item-002", description="Second task"),
        ]
        state = GraphState(todo_items=items, current_item_index=1)
        assert state.current_item.id == "item-002"

    def test_current_item_out_of_bounds(self):
        from app.core.state import GraphState
        state = GraphState(current_item_index=5)
        assert state.current_item is None

    def test_progress_summary(self):
        from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase
        items = [
            TodoItem(id="item-001", description="Done task", status=ItemStatus.DONE),
            TodoItem(id="item-002", description="Current task", status=ItemStatus.IN_PROGRESS),
        ]
        state = GraphState(
            todo_items=items,
            current_item_index=1,
            phase=WorkflowPhase.CODING,
            branch_name="feature/test",
        )
        summary = state.get_progress_summary()
        assert "1/2" in summary
        assert "coding" in summary.lower()
        assert "feature/test" in summary


class TestTodoItem:
    def test_default_status(self):
        from app.core.state import ItemStatus, TodoItem
        item = TodoItem(id="test-001", description="Test item")
        assert item.status == ItemStatus.PENDING
        assert item.iteration_count == 0

    def test_iteration_tracking(self):
        from app.core.state import TodoItem
        item = TodoItem(id="test-001", description="Test")
        item.iteration_count += 1
        assert item.iteration_count == 1


class TestDualCoderSettings:
    def test_coder_a_model_default(self):
        with patch.dict("os.environ", {}, clear=True):
            from app.core.config import Settings
            s = Settings(_env_file=None)
            assert s.coder_a_model == "claude-sonnet-4-20250514"

    def test_coder_b_model_default(self):
        with patch.dict("os.environ", {}, clear=True):
            from app.core.config import Settings
            s = Settings(_env_file=None)
            assert s.coder_b_model == "gpt-5.2"

    def test_documenter_model_default(self):
        with patch.dict("os.environ", {}, clear=True):
            from app.core.config import Settings
            s = Settings(_env_file=None)
            assert s.documenter_model == "gpt-4o-mini"

    def test_coder_model_legacy_property(self):
        from app.core.config import Settings
        s = Settings()
        assert s.coder_model == s.coder_a_model

    def test_custom_models(self):
        from app.core.config import Settings
        s = Settings(coder_a_model="claude-opus-4-0-20250514", coder_b_model="gpt-4o")
        assert s.coder_a_model == "claude-opus-4-0-20250514"
        assert s.coder_b_model == "gpt-4o"


class TestDualCoderGraphState:
    def test_default_coder_assignment(self):
        from app.core.state import GraphState
        state = GraphState()
        assert state.active_coder == "coder_a"
        assert state.active_reviewer == "reviewer_b"
        assert state.peer_review_verdict == ""
        assert state.peer_review_notes == ""

    def test_peer_review_phase_exists(self):
        from app.core.state import WorkflowPhase
        assert WorkflowPhase.PEER_REVIEWING == "peer_reviewing"
