"""Tests for workflow checkpoint management."""

from types import SimpleNamespace


def test_checkpoint_save_and_load_roundtrip(tmp_path, monkeypatch):
    from app.core.checkpoints import checkpoint_manager
    from app.core.state import GraphState, WorkflowPhase

    monkeypatch.setattr(
        "app.core.checkpoints.get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path)),
    )

    state = GraphState(
        user_request="Implement health endpoint",
        input_intent="code",
        phase=WorkflowPhase.CODING,
        repo_facts={"language": "python"},
    )
    checkpoint_id = checkpoint_manager.save_checkpoint(state, "test", repo_root=str(tmp_path))
    assert checkpoint_id.startswith("test_")

    restored = checkpoint_manager.load_checkpoint(checkpoint_id=checkpoint_id, repo_root=str(tmp_path))
    assert restored is not None
    assert restored.user_request == state.user_request
    assert restored.repo_facts == state.repo_facts
    assert restored.resumed_from_checkpoint is True


def test_mark_latest_approval_updates_checkpoint(tmp_path, monkeypatch):
    from app.core.checkpoints import checkpoint_manager
    from app.core.state import GraphState

    monkeypatch.setattr(
        "app.core.checkpoints.get_settings",
        lambda: SimpleNamespace(target_repo_path=str(tmp_path)),
    )

    state = GraphState(
        user_request="Commit task",
        needs_human_approval=True,
        pending_approval={"type": "commit", "approved": False},
    )
    checkpoint_manager.save_checkpoint(state, "await_approval", repo_root=str(tmp_path))
    assert checkpoint_manager.mark_latest_approval(True, repo_root=str(tmp_path)) is True

    restored = checkpoint_manager.load_checkpoint(repo_root=str(tmp_path))
    assert restored is not None
    assert restored.pending_approval.get("approved") is True
