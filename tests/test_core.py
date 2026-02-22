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
        assert state.needs_human_approval is False
        assert state.pending_approval == {}
        assert state.approval_history == []
        assert state.state_checkpoint_id is None
        assert state.checkpoint_id is None
        assert state.last_checkpoint_path is None
        assert state.resumed_from_checkpoint is False

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


class TestGraphStateSerialization:
    def test_graphstate_serialization_roundtrip(self):
        from app.core.state import GraphState

        state = GraphState(
            user_request="test task",
            input_intent="code",
            repo_facts={"language": "python"},
            needs_human_approval=True,
            pending_approval={"type": "commit", "approved": False},
        )

        payload = state.model_dump()
        assert isinstance(payload, dict)
        assert payload["user_request"] == "test task"
        assert payload["repo_facts"]["language"] == "python"

        restored = GraphState.from_dict(payload)
        assert restored.user_request == state.user_request
        assert restored.repo_facts == state.repo_facts
        assert restored.needs_human_approval is True
        assert restored.pending_approval["type"] == "commit"

    def test_from_dict_accepts_checkpoint_id_alias(self):
        from app.core.state import GraphState

        restored = GraphState.from_dict({"user_request": "x", "checkpoint_id": "cp-123"})
        assert restored.state_checkpoint_id == "cp-123"
        assert restored.checkpoint_id == "cp-123"

    def test_backward_compatibility_defaults(self):
        from app.core.state import GraphState

        state = GraphState(user_request="test")
        assert state.agent_instructions == ""
        assert state.repo_facts == {}
        assert state.needs_human_approval is False
        assert state.pending_approval == {}


class TestWorkflowPhaseExtensions:
    def test_new_phase_members_exist(self):
        from app.core.state import WorkflowPhase

        assert WorkflowPhase.LOADING_CONTEXT == "loading_context"
        assert WorkflowPhase.WAITING_FOR_APPROVAL == "waiting_for_approval"
