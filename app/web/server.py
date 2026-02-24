"""FastAPI web server — wired to the event bus for real-time UI updates.

Events from nodes are broadcast to all WebSocket clients automatically.
The UI receives structured events and renders them as collapsible steps.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.checkpoints import checkpoint_manager
from app.core.config import get_settings
from app.core.events import WorkflowEvent, get_history, set_event_loop, subscribe_async
from app.core.events import emit_approval_done
from app.core.approval_registry import registry as approval_registry
from app.core.logging import get_logger
from app.core.orchestrator import request_shutdown, reset_shutdown, run_workflow
from app.core.state import GraphState, WorkflowPhase

logger = get_logger("web.server")

# ── In-memory state ───────────────────────────────────────────────────────
_current_state: GraphState | None = None
_ws_clients: set[WebSocket] = set()
_task_queue: asyncio.Queue | None = None


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
    token_budget: dict = {}


class ApprovalRequest(BaseModel):
    approved: bool = True


class AnswerRequest(BaseModel):
    answer: str


# ── WebSocket broadcast (wired to event bus) ─────────────────────────────

async def _broadcast_event(event: WorkflowEvent) -> None:
    """Forward a workflow event to all connected WebSocket clients."""
    if not _ws_clients:
        return

    message = json.dumps({"type": "event", "data": event.to_dict()})
    disconnected: set[WebSocket] = set()
    # Iterate over a snapshot so connects/disconnects during await do not mutate
    # the collection we are iterating.
    for ws in tuple(_ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    if disconnected:
        _ws_clients.difference_update(disconnected)


async def _broadcast_raw(event_type: str, data: Any) -> None:
    """Send a raw message (non-event) to all clients."""
    if not _ws_clients:
        return

    message = json.dumps({"type": event_type, "data": data, "ts": datetime.now(UTC).isoformat()})
    disconnected: set[WebSocket] = set()
    # Iterate over a snapshot so connects/disconnects during await do not mutate
    # the collection we are iterating.
    for ws in tuple(_ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    if disconnected:
        _ws_clients.difference_update(disconnected)


# ── Background worker ─────────────────────────────────────────────────────

async def _process_tasks():
    """Background worker that processes tasks from the queue."""
    global _current_state

    while True:
        task_req: TaskRequest = await _task_queue.get()

        try:
            settings = get_settings()
            repo = task_req.repo_path or settings.target_repo_path

            _current_state = GraphState(
                user_request=task_req.task,
                repo_root=repo,
                phase=WorkflowPhase.PLANNING,
            )
            await _broadcast_raw("status", {"phase": "planning", "task": task_req.task})

            final_state = await run_workflow(task_req.task, repo)
            _current_state = final_state

            # If the workflow halted waiting for human approval, register it in
            # the shared registry so the Telegram bot can also resolve it.
            if final_state.needs_human_approval:
                _register_pending_approval(final_state, repo)

            # If the workflow halted waiting for a coder answer, register the
            # question in the shared registry so Telegram can answer it too.
            if final_state.needs_coder_answer:
                _register_pending_question(final_state, repo)

            await _broadcast_raw("status", {
                "phase": final_state.phase.value,
                "completed": final_state.completed_items,
                "total": len(final_state.todo_items),
                "branch": final_state.branch_name,
            })

        except Exception as e:
            logger.error("Task failed: %s", e, exc_info=True)
            if _current_state:
                _current_state.phase = WorkflowPhase.STOPPED
                _current_state.error_message = str(e)
            await _broadcast_raw("error", {"message": str(e)})

        _task_queue.task_done()


# ── Approval registry wiring ──────────────────────────────────────────────

def _register_pending_approval(state: GraphState, repo_root: str) -> None:
    """Register the current pending approval in the shared registry.

    The resume callback queues a new 'resume' task so the workflow continues
    after the human decides, regardless of whether the decision comes from the
    web UI or the Telegram bot.
    """
    def _resume(approved: bool) -> None:
        global _current_state
        if _current_state is None:
            return

        pending_type = (_current_state.pending_approval or {}).get("type", "commit")

        if approved:
            _current_state.pending_approval["approved"] = True
            _current_state.needs_human_approval = False
            _current_state.stop_reason = ""
            checkpoint_manager.mark_latest_approval(True, repo_root=repo_root)
            emit_approval_done(approved=True, pending_type=pending_type)
            # Queue a resume task on the asyncio event loop
            import asyncio
            if _task_queue is not None:
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        _task_queue.put_nowait,
                        TaskRequest(task="resume", repo_path=repo_root),
                    )
                except Exception as exc:
                    logger.error("Failed to queue resume task: %s", exc)
        else:
            _current_state.pending_approval["approved"] = False
            _current_state.needs_human_approval = False
            _current_state.phase = WorkflowPhase.STOPPED
            _current_state.stop_reason = "user_rejected"
            checkpoint_manager.mark_latest_approval(False, repo_root=repo_root)
            emit_approval_done(approved=False, pending_type=pending_type)

    approval_registry.set_pending(state.pending_approval or {}, _resume)
    logger.info("Pending approval registered in shared registry")


def _register_pending_question(state: GraphState, repo_root: str) -> None:
    """Register the coder's question in the shared registry.

    The answer callback populates ``coder_question_answer`` on the current state
    and queues a resume task so the coder continues with the answer in context.
    """
    def _deliver_answer(answer: str) -> None:
        global _current_state
        if _current_state is None:
            return
        from app.core.events import emit_coder_answer
        _current_state.coder_question_answer = answer
        _current_state.needs_coder_answer = False
        _current_state.phase = WorkflowPhase.WAITING_FOR_ANSWER  # answer_gate will clear
        emit_coder_answer(
            asked_by=_current_state.coder_question_asked_by,
            answer=answer,
            item_id=(_current_state.current_item.id if _current_state.current_item else ""),
        )
        import asyncio
        if _task_queue is not None:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    _task_queue.put_nowait,
                    TaskRequest(task="resume", repo_path=repo_root),
                )
            except Exception as exc:
                logger.error("Failed to queue resume after answer: %s", exc)

    # Re-use the registry's set_pending with a string-typed callback wrapper.
    # We store the question payload as the "pending" dict and the deliver
    # callback as a lambda that accepts a bool — we encode the answer string
    # via a closure so the registry's bool API still works for approval, while
    # here we store the raw answer string on state before calling the callback.
    #
    # To keep the registry generic we store the answer delivery as a separate
    # attribute on the registry instance.
    approval_registry.set_answer_pending(
        question={
            "question": state.coder_question,
            "context": state.coder_question_context,
            "options": state.coder_question_options,
            "asked_by": state.coder_question_asked_by,
        },
        answer_callback=_deliver_answer,
    )
    logger.info("Coder question registered in shared registry")


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_queue
    # Create queue on the current event loop
    _task_queue = asyncio.Queue()
    # Clear any previous shutdown flag (important for test re-runs)
    reset_shutdown()
    # Register main event loop for cross-thread event delivery
    set_event_loop(asyncio.get_running_loop())
    # Subscribe to event bus
    subscribe_async(_broadcast_event)
    # Start background worker
    worker = asyncio.create_task(_process_tasks())
    logger.info("Web server started — event bus wired, worker running")
    yield
    # Graceful shutdown: signal workflow to stop, then cancel the worker
    request_shutdown()
    worker.cancel()
    with suppress(asyncio.CancelledError):
        await worker
    logger.info("Lifespan cleanup complete")


app = FastAPI(title="Daedadus", version="0.2.0", lifespan=lifespan)


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/task")
async def submit_task(req: TaskRequest):
    await _task_queue.put(req)
    return {"status": "queued", "task": req.task, "queue_size": _task_queue.qsize()}


@app.get("/api/status")
async def get_status():
    if not _current_state:
        return StatusResponse(phase="idle", progress="No active task", branch="", error="", items_total=0, items_done=0)
    return StatusResponse(
        phase=_current_state.phase.value,
        progress=_current_state.get_progress_summary(),
        branch=_current_state.branch_name,
        error=_current_state.error_message,
        items_total=len(_current_state.todo_items),
        items_done=_current_state.completed_items,
        token_budget=_current_state.token_budget or {},
    )


@app.get("/api/intelligence-summary")
async def get_intelligence_summary():
    """Return a compact summary of all code intelligence results."""
    if not _current_state:
        return {"available": False}
    s = _current_state
    smells = s.code_smells or []
    static = s.static_issues or []
    cg     = s.call_graph or {}
    dg     = s.dependency_graph or {}
    return {
        "available": bool(smells or static or cg or dg),
        "cached": s.intelligence_cached,
        "cache_key": s.intelligence_cache_key,
        "static": {
            "total": len(static),
            "errors":   sum(1 for i in static if i.get("severity") == "error"),
            "warnings": sum(1 for i in static if i.get("severity") == "warning"),
        },
        "smells": {
            "total": len(smells),
            "errors":   sum(1 for i in smells if i.get("severity") == "error"),
            "warnings": sum(1 for i in smells if i.get("severity") == "warning"),
        },
        "call_graph": {
            "functions": len(cg.get("callees", {})),
            "edges": sum(len(v) for v in cg.get("callees", {}).values()),
        },
        "dependency_graph": {
            "modules": len(dg.get("imports", {})),
            "cycles":  len(s.dep_cycles or []),
        },
    }


@app.get("/api/code-smells")
async def get_code_smells():
    """Return code smell findings for the current repo."""
    if not _current_state or not _current_state.code_smells:
        return {"smells": [], "total": 0, "errors": 0, "warnings": 0, "infos": 0}
    smells = _current_state.code_smells
    return {
        "smells": smells,
        "total": len(smells),
        "errors":   sum(1 for s in smells if s.get("severity") == "error"),
        "warnings": sum(1 for s in smells if s.get("severity") == "warning"),
        "infos":    sum(1 for s in smells if s.get("severity") == "info"),
    }


@app.get("/api/dependency-graph")
async def get_dependency_graph():
    """Return the dependency graph for the current repo as JSON + Mermaid."""
    if not _current_state or not _current_state.dependency_graph:
        return {"mermaid": "", "json": {}, "cycles": [], "modules": 0}
    try:
        from app.analysis.dependency_graph import DependencyGraph
        dg = DependencyGraph.from_dict(_current_state.dependency_graph)
        return {
            "mermaid": dg.to_mermaid(highlight_cycles=True),
            "json": _current_state.dependency_graph,
            "cycles": dg.cycles,
            "modules": len(dg.all_modules()),
            "edges": sum(len(v) for v in dg.imports.values()),
        }
    except Exception as exc:
        return {"error": str(exc), "mermaid": "", "json": {}, "cycles": []}


@app.get("/api/events")
async def get_events(limit: int = 200):
    """Return recent workflow events."""
    return {"events": get_history(limit)}


@app.get("/api/pending")
async def get_pending_approval():
    """Return the current pending approval payload (for page-reload recovery)."""
    if not _current_state:
        return {"needs_human_approval": False, "pending_approval": {}}
    return {
        "needs_human_approval": _current_state.needs_human_approval,
        "pending_approval": _current_state.pending_approval or {},
    }


@app.get("/api/question")
async def get_pending_question():
    """Return the current pending coder question (for page-reload recovery)."""
    if not _current_state:
        return {"needs_coder_answer": False, "coder_question": {}}
    if not _current_state.needs_coder_answer:
        return {"needs_coder_answer": False, "coder_question": {}}
    return {
        "needs_coder_answer": True,
        "coder_question": {
            "question": _current_state.coder_question,
            "context": _current_state.coder_question_context,
            "options": _current_state.coder_question_options,
            "asked_by": _current_state.coder_question_asked_by,
        },
    }


@app.post("/api/answer")
async def submit_coder_answer(req: AnswerRequest):
    """Submit a human answer to a coder's mid-task question."""
    global _current_state

    if not isinstance(_current_state, GraphState):
        return {"error": "No pending question"}

    if not _current_state.needs_coder_answer:
        return {"error": "No pending question"}

    answer = req.answer.strip()
    if not answer:
        return {"error": "Answer must not be empty"}

    # Deliver via registry if registered (supports both web and Telegram)
    if approval_registry.is_question_pending:
        delivered = approval_registry.deliver_answer(answer)
        if delivered:
            return {"status": "answer_submitted", "answer": answer}

    # Fallback: mutate state directly and queue resume
    from app.core.events import emit_coder_answer
    _current_state.coder_question_answer = answer
    _current_state.needs_coder_answer = False
    emit_coder_answer(
        asked_by=_current_state.coder_question_asked_by,
        answer=answer,
        item_id=(_current_state.current_item.id if _current_state.current_item else ""),
    )
    repo_root = str(_current_state.repo_root or "")
    if _task_queue is not None:
        await _task_queue.put(TaskRequest(task="resume", repo_path=repo_root))

    return {"status": "answer_submitted", "answer": answer}


@app.post("/api/approve")
async def approve_pending_action(req: ApprovalRequest):
    """Approve or reject a pending human-gate action."""
    global _current_state

    if not isinstance(_current_state, GraphState):
        return {"error": "No pending approval"}

    if not _current_state.needs_human_approval:
        return {"error": "No pending approval"}

    timestamp = datetime.now(UTC).isoformat()
    entry = {
        "timestamp": timestamp,
        "approved": req.approved,
        "pending_type": (_current_state.pending_approval or {}).get("type", "unknown"),
    }
    _current_state.approval_history.append(entry)

    # If the shared registry holds a callback, let it handle state mutation
    # and resume queueing — this keeps web and Telegram code paths identical.
    if approval_registry.is_pending:
        approval_registry.approve(req.approved)
        action = "approved" if req.approved else "rejected"
        return {"status": action, "message": f"Action {action}."}

    # Fallback: direct mutation (used when registry wasn't populated, e.g. tests)
    pending_type = (_current_state.pending_approval or {}).get("type", "commit")
    if req.approved:
        _current_state.pending_approval["approved"] = True
        _current_state.needs_human_approval = False
        _current_state.stop_reason = ""
        repo_root = str(_current_state.repo_root or "")
        checkpoint_manager.mark_latest_approval(True, repo_root=repo_root)
        emit_approval_done(approved=True, pending_type=pending_type)
        if _task_queue is not None:
            await _task_queue.put(TaskRequest(task="resume", repo_path=repo_root))
        return {"status": "approved", "message": "Action approved, resume queued."}

    _current_state.pending_approval["approved"] = False
    _current_state.needs_human_approval = False
    _current_state.phase = WorkflowPhase.STOPPED
    _current_state.stop_reason = "user_rejected"
    checkpoint_manager.mark_latest_approval(False, repo_root=str(_current_state.repo_root or ""))
    emit_approval_done(approved=False, pending_type=pending_type)
    return {"status": "rejected", "message": "Action rejected, workflow stopped."}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))

    try:
        # Send recent event history on connect
        history = get_history(50)
        for evt in history:
            await ws.send_text(json.dumps({"type": "event", "data": evt}))

        if _current_state:
            await ws.send_text(json.dumps({
                "type": "status",
                "data": {"phase": _current_state.phase.value, "progress": _current_state.get_progress_summary()},
                "ts": datetime.now(UTC).isoformat(),
            }))

        while True:
            data = await ws.receive_text()
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
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Daedadus</h1><p>UI not found.</p>")
