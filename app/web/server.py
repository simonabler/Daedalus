"""FastAPI web server — REST API + WebSocket for live status/logs.

Endpoints:
  POST /api/task      — submit a new task
  GET  /api/status    — current workflow status
  GET  /api/logs      — recent log entries
  WS   /ws            — real-time status + log stream
  GET  /              — serves the web UI
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.orchestrator import run_workflow
from app.core.state import GraphState, WorkflowPhase

logger = get_logger("web.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    task = asyncio.create_task(_process_tasks())
    logger.info("Web server started — background worker running")
    yield
    task.cancel()


app = FastAPI(title="AI Dev Worker", version="0.1.0", lifespan=lifespan)

# ── In-memory state ───────────────────────────────────────────────────────
_current_state: GraphState | None = None
_log_buffer: deque[dict] = deque(maxlen=500)
_ws_clients: set[WebSocket] = set()
_task_queue: asyncio.Queue = asyncio.Queue()
_worker_running = False


# ── Models ────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    repo_path: str = ""


class StatusResponse(BaseModel):
    phase: str
    progress: str
    branch: str
    error: str
    items_total: int
    items_done: int


# ── Broadcast helper ──────────────────────────────────────────────────────

async def _broadcast(event_type: str, data: Any):
    """Send an event to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data, "ts": datetime.now(UTC).isoformat()})
    disconnected = set()
    for ws in _ws_clients:  # noqa: F823
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    _ws_clients -= disconnected


def _log_event(level: str, message: str):
    entry = {"level": level, "message": message, "ts": datetime.now(UTC).isoformat()}
    _log_buffer.append(entry)
    # Fire-and-forget broadcast
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast("log", entry))
    except RuntimeError:
        pass


# ── Background worker ─────────────────────────────────────────────────────

async def _process_tasks():
    """Background worker that processes tasks from the queue."""
    global _current_state, _worker_running
    _worker_running = True

    while True:
        task_req: TaskRequest = await _task_queue.get()
        _log_event("INFO", f"Starting task: {task_req.task[:100]}")

        try:
            settings = get_settings()
            repo = task_req.repo_path or settings.target_repo_path

            _current_state = GraphState(
                user_request=task_req.task,
                repo_root=repo,
                phase=WorkflowPhase.PLANNING,
            )
            await _broadcast("status", {"phase": "planning", "task": task_req.task})

            # Run the workflow
            final_state = await run_workflow(task_req.task, repo)
            _current_state = final_state

            await _broadcast("status", {
                "phase": final_state.phase.value,
                "completed": final_state.completed_items,
                "total": len(final_state.todo_items),
            })
            _log_event("INFO", f"Task complete: {final_state.phase.value}")

        except Exception as e:
            _log_event("ERROR", f"Task failed: {e}")
            if _current_state:
                _current_state.phase = WorkflowPhase.STOPPED
                _current_state.error_message = str(e)
            await _broadcast("error", {"message": str(e)})

        _task_queue.task_done()


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/task")
async def submit_task(req: TaskRequest):
    """Submit a new task for the AI Dev Worker."""
    await _task_queue.put(req)
    _log_event("INFO", f"Task queued: {req.task[:100]}")
    return {"status": "queued", "task": req.task, "queue_size": _task_queue.qsize()}


@app.get("/api/status")
async def get_status():
    """Get current workflow status."""
    if not _current_state:
        return StatusResponse(
            phase="idle", progress="No active task", branch="",
            error="", items_total=0, items_done=0,
        )
    return StatusResponse(
        phase=_current_state.phase.value,
        progress=_current_state.get_progress_summary(),
        branch=_current_state.branch_name,
        error=_current_state.error_message,
        items_total=len(_current_state.todo_items),
        items_done=_current_state.completed_items,
    )


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries."""
    entries = list(_log_buffer)[-limit:]
    return {"logs": entries}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time status and log streaming."""
    await ws.accept()
    _ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))

    try:
        # Send current state on connect
        if _current_state:
            await ws.send_text(json.dumps({
                "type": "status",
                "data": {"phase": _current_state.phase.value, "progress": _current_state.get_progress_summary()},
                "ts": datetime.now(UTC).isoformat(),
            }))

        # Keep connection alive, receive messages
        while True:
            data = await ws.receive_text()
            # Client can send task submissions via WebSocket too
            try:
                msg = json.loads(data)
                if msg.get("type") == "task":
                    req = TaskRequest(task=msg["task"], repo_path=msg.get("repo_path", ""))
                    await _task_queue.put(req)
                    await ws.send_text(json.dumps({"type": "ack", "data": "Task queued"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


# ── Serve static UI ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AI Dev Worker</h1><p>UI not found. Place index.html in app/web/static/</p>")
