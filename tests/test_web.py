"""Tests for the FastAPI web server endpoints."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with patch("app.web.server.get_settings") as ms:
        ms.return_value.target_repo_path = "/tmp/test-repo"
        ms.return_value.max_output_chars = 10000

        with patch("app.web.server.run_workflow", new_callable=AsyncMock):
            from fastapi.testclient import TestClient

            from app.web.server import app
            with TestClient(app) as c:
                yield c


class TestStatusEndpoint:
    def test_status_idle(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "idle"
        assert data["items_total"] == 0

    def test_status_fields(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert "phase" in data
        assert "progress" in data
        assert "branch" in data
        assert "items_total" in data
        assert "items_done" in data


class TestEventsEndpoint:
    def test_events_returns_list(self, client):
        resp = client.get("/api/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_events_limit(self, client):
        resp = client.get("/api/events?limit=5")
        assert resp.status_code == 200


class TestTaskEndpoint:
    def test_submit_task(self, client):
        resp = client.post("/api/task", json={"task": "Add a hello world endpoint"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "hello world" in data["task"]


class TestApprovalEndpoint:
    def test_approve_without_pending_returns_error(self, client):
        resp = client.post("/api/approve", json={"approved": True})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_approve_with_pending_state(self, client):
        """When a state with needs_human_approval is present, approving returns 'approved'."""
        from unittest.mock import MagicMock
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            needs_human_approval=True,
            pending_approval={"approved": False, "type": "commit", "files": [], "triggers": []},
            phase=WorkflowPhase.WAITING_FOR_APPROVAL,
        )

        import app.web.server as srv
        original = srv._current_state
        srv._current_state = state
        try:
            with patch("app.web.server.checkpoint_manager") as mock_cp:
                mock_cp.mark_latest_approval.return_value = None
                resp = client.post("/api/approve", json={"approved": True})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "approved"
            assert state.needs_human_approval is False
        finally:
            srv._current_state = original

    def test_reject_with_pending_state(self, client):
        """Rejecting stops the workflow."""
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            needs_human_approval=True,
            pending_approval={"approved": False, "type": "commit", "files": [], "triggers": []},
            phase=WorkflowPhase.WAITING_FOR_APPROVAL,
        )

        import app.web.server as srv
        original = srv._current_state
        srv._current_state = state
        try:
            with patch("app.web.server.checkpoint_manager") as mock_cp:
                mock_cp.mark_latest_approval.return_value = None
                resp = client.post("/api/approve", json={"approved": False})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "rejected"
            assert state.phase == WorkflowPhase.STOPPED
            assert state.stop_reason == "user_rejected"
        finally:
            srv._current_state = original


class TestPendingEndpoint:
    def test_pending_when_idle(self, client):
        import app.web.server as srv
        original = srv._current_state
        srv._current_state = None
        try:
            resp = client.get("/api/pending")
            assert resp.status_code == 200
            data = resp.json()
            assert data["needs_human_approval"] is False
            assert data["pending_approval"] == {}
        finally:
            srv._current_state = original

    def test_pending_when_waiting(self, client):
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            needs_human_approval=True,
            pending_approval={
                "approved": False,
                "type": "commit",
                "summary": "2 files changed",
                "files": ["app/main.py", "tests/test_main.py"],
                "triggers": [{"type": "commit", "reason": "Commit requires approval"}],
                "diff_preview": "diff --git a/app/main.py...",
                "git_status": "M app/main.py",
            },
            phase=WorkflowPhase.WAITING_FOR_APPROVAL,
        )

        import app.web.server as srv
        original = srv._current_state
        srv._current_state = state
        try:
            resp = client.get("/api/pending")
            assert resp.status_code == 200
            data = resp.json()
            assert data["needs_human_approval"] is True
            assert data["pending_approval"]["summary"] == "2 files changed"
            assert len(data["pending_approval"]["files"]) == 2
        finally:
            srv._current_state = original


class TestApprovalEvent:
    def test_emit_approval_needed_produces_correct_category(self):
        from app.core.events import emit_approval_needed, get_history, EventCategory
        import app.core.events as ev_module

        # Clear history
        ev_module._history.clear()

        emit_approval_needed({
            "summary": "3 files changed",
            "files": ["a.py", "b.py", "c.py"],
            "triggers": [{"type": "commit", "reason": "Commit requires approval"}],
            "diff_preview": "--- a\n+++ b\n@@ -1 +1 @@\n+new line",
            "git_status": "M a.py",
            "timestamp": "2026-01-01T00:00:00Z",
        })

        history = get_history(1)
        assert len(history) == 1
        evt = history[0]
        assert evt["category"] == EventCategory.APPROVAL_NEEDED.value
        assert evt["metadata"]["summary"] == "3 files changed"
        assert "a.py" in evt["metadata"]["files"]
        assert evt["metadata"]["triggers"][0]["type"] == "commit"

    def test_emit_approval_done_approved(self):
        from app.core.events import emit_approval_done, get_history, EventCategory
        import app.core.events as ev_module

        ev_module._history.clear()
        emit_approval_done(approved=True, pending_type="commit")

        history = get_history(1)
        assert history[0]["category"] == EventCategory.APPROVAL_DONE.value
        assert history[0]["metadata"]["approved"] is True

    def test_emit_approval_done_rejected(self):
        from app.core.events import emit_approval_done, get_history, EventCategory
        import app.core.events as ev_module

        ev_module._history.clear()
        emit_approval_done(approved=False, pending_type="commit")

        history = get_history(1)
        assert history[0]["metadata"]["approved"] is False


class TestUI:
    def test_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Daedalus" in resp.text or "DAEDALUS" in resp.text
