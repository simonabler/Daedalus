"""Tests for the FastAPI web server endpoints."""

from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with patch("app.web.server.get_settings") as ms:
        ms.return_value.target_repo_path = "/tmp/test-repo"
        ms.return_value.max_output_chars = 10000

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


class TestLogsEndpoint:
    def test_logs_returns_list(self, client):
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)

    def test_logs_limit(self, client):
        resp = client.get("/api/logs?limit=5")
        assert resp.status_code == 200


class TestTaskEndpoint:
    def test_submit_task(self, client):
        resp = client.post("/api/task", json={"task": "Add a hello world endpoint"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "hello world" in data["task"]


class TestUI:
    def test_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Daedalus" in resp.text or "DAEDALUS" in resp.text


class _OkWS:
    def __init__(self):
        self.messages: list[str] = []

    async def send_text(self, message: str):
        self.messages.append(message)


class _FailWS:
    async def send_text(self, message: str):
        raise RuntimeError("socket closed")


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_clients(self, monkeypatch):
        from app.web import server

        ok = _OkWS()
        bad = _FailWS()
        monkeypatch.setattr(server, "_ws_clients", {ok, bad})

        await server._broadcast("status", {"phase": "testing"})

        assert len(ok.messages) == 1
        assert ok in server._ws_clients
        assert bad not in server._ws_clients

    @pytest.mark.asyncio
    async def test_broadcast_with_no_clients_is_noop(self, monkeypatch):
        from app.web import server

        monkeypatch.setattr(server, "_ws_clients", set())

        await server._broadcast("status", {"phase": "idle"})
