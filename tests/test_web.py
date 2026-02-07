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
        assert "AI Dev Worker" in resp.text or "AI DEV WORKER" in resp.text
