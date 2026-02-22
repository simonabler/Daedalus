"""Event bus for workflow activity — decouples nodes from the UI layer.

Nodes call `emit(...)` to publish structured events.  The web server
(and Telegram bot) subscribe via `subscribe()` to receive them and
forward to the user in real-time.

Event categories:
  node_start       — a graph node begins execution
  node_end         — a graph node finishes
  agent_think      — an LLM is invoked (before response)
  agent_result     — LLM returned a response
  tool_call        — a tool is being called
  tool_result      — tool returned
  plan             — planner produced / updated the plan
  status           — phase/progress change
  verdict          — review or test verdict (APPROVE/REWORK/PASS/FAIL)
  commit           — a commit was made
  error            — something went wrong
  approval_needed  — human gate fired; UI must show approve/reject panel
  approval_done    — human submitted an approve or reject decision
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from app.core.logging import get_logger

logger = get_logger("core.events")

# ── Event loop reference for cross-thread delivery ───────────────────────
_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the main asyncio event loop for cross-thread event delivery."""
    global _loop
    _loop = loop


class EventCategory(str, Enum):
    NODE_START = "node_start"
    NODE_END = "node_end"
    AGENT_THINK = "agent_think"
    AGENT_RESULT = "agent_result"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PLAN = "plan"
    STATUS = "status"
    VERDICT = "verdict"
    COMMIT = "commit"
    ERROR = "error"
    APPROVAL_NEEDED = "approval_needed"
    APPROVAL_DONE = "approval_done"


@dataclass
class WorkflowEvent:
    """A single event emitted during workflow execution."""
    category: EventCategory
    agent: str                      # e.g. "planner", "coder_a", "reviewer_b", "tester", "system"
    title: str                      # short human-readable headline
    detail: str = ""                # longer content (agent output, tool result, etc.)
    metadata: dict = field(default_factory=dict)  # structured data (verdict, item_id, etc.)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "agent": self.agent,
            "title": self.title,
            "detail": self.detail,
            "metadata": self.metadata,
            "ts": self.timestamp,
        }


# ── Singleton event bus ──────────────────────────────────────────────────

_listeners: list[Callable[[WorkflowEvent], Any]] = []
_async_listeners: list[Callable[[WorkflowEvent], Awaitable[Any]]] = []
_history: deque[WorkflowEvent] = deque(maxlen=1000)


def emit(event: WorkflowEvent) -> None:
    """Emit an event synchronously. Safe to call from any thread."""
    _history.append(event)
    logger.debug("EVENT | %s | %s | %s", event.category.value, event.agent, event.title)

    # Call sync listeners
    for listener in _listeners:
        try:
            listener(event)
        except Exception as e:
            logger.warning("Sync listener error: %s", e)

    # Schedule async listeners into the main event loop (thread-safe)
    if _loop is not None and not _loop.is_closed():
        for listener in _async_listeners:
            try:
                _loop.call_soon_threadsafe(asyncio.ensure_future, listener(event))
            except RuntimeError:
                pass  # loop already closed
    else:
        # Fallback: try current thread's loop (works when called from async context)
        for listener in _async_listeners:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(listener(event))
            except RuntimeError:
                pass


def subscribe_sync(listener: Callable[[WorkflowEvent], Any]) -> None:
    """Register a synchronous event listener."""
    _listeners.append(listener)


def subscribe_async(listener: Callable[[WorkflowEvent], Awaitable[Any]]) -> None:
    """Register an async event listener (e.g. WebSocket broadcast)."""
    _async_listeners.append(listener)


def get_history(limit: int = 200) -> list[dict]:
    """Return recent events as dicts."""
    items = list(_history)
    return [e.to_dict() for e in items[-limit:]]


def clear_listeners() -> None:
    """Remove all listeners (useful for testing)."""
    _listeners.clear()
    _async_listeners.clear()


# ── Convenience emitters (called from nodes) ─────────────────────────────

def emit_node_start(agent: str, node_name: str, item_id: str = "", item_desc: str = "") -> None:
    emit(WorkflowEvent(
        category=EventCategory.NODE_START,
        agent=agent,
        title=f"▶ {node_name} started",
        detail=f"Working on: {item_desc}" if item_desc else "",
        metadata={"node": node_name, "item_id": item_id},
    ))


def emit_node_end(agent: str, node_name: str, summary: str = "") -> None:
    emit(WorkflowEvent(
        category=EventCategory.NODE_END,
        agent=agent,
        title=f"✓ {node_name} finished",
        detail=summary,
        metadata={"node": node_name},
    ))


def emit_agent_thinking(agent: str, prompt_summary: str = "") -> None:
    emit(WorkflowEvent(
        category=EventCategory.AGENT_THINK,
        agent=agent,
        title=f"🧠 {agent} is thinking…",
        detail=prompt_summary[:500] if prompt_summary else "",
    ))


def emit_agent_result(agent: str, result: str, truncate: int = 2000) -> None:
    detail = result[:truncate] + "…" if len(result) > truncate else result
    emit(WorkflowEvent(
        category=EventCategory.AGENT_RESULT,
        agent=agent,
        title=f"💬 {agent} responded",
        detail=detail,
    ))


def emit_tool_call(agent: str, tool_name: str, args_summary: str = "") -> None:
    emit(WorkflowEvent(
        category=EventCategory.TOOL_CALL,
        agent=agent,
        title=f"🔧 {agent} → {tool_name}",
        detail=args_summary[:300] if args_summary else "",
        metadata={"tool": tool_name},
    ))


def emit_tool_result(agent: str, tool_name: str, result: str, truncate: int = 1000) -> None:
    detail = result[:truncate] + "…" if len(result) > truncate else result
    emit(WorkflowEvent(
        category=EventCategory.TOOL_RESULT,
        agent=agent,
        title=f"📎 {tool_name} returned",
        detail=detail,
        metadata={"tool": tool_name},
    ))


def emit_plan(agent: str, plan_text: str, items_count: int = 0) -> None:
    emit(WorkflowEvent(
        category=EventCategory.PLAN,
        agent=agent,
        title=f"📋 Plan created — {items_count} items",
        detail=plan_text,
        metadata={"items_count": items_count},
    ))


def emit_status(agent: str, message: str, phase: str = "", item_id: str = "", **extra) -> None:
    emit(WorkflowEvent(
        category=EventCategory.STATUS,
        agent=agent,
        title=message,
        metadata={"phase": phase, "item_id": item_id, **extra},
    ))


def emit_verdict(agent: str, verdict: str, detail: str = "", item_id: str = "") -> None:
    icon = {"APPROVE": "✅", "REWORK": "🔄", "PASS": "✅", "FAIL": "❌"}.get(verdict, "❓")
    emit(WorkflowEvent(
        category=EventCategory.VERDICT,
        agent=agent,
        title=f"{icon} {agent}: {verdict}",
        detail=detail[:1500] if detail else "",
        metadata={"verdict": verdict, "item_id": item_id},
    ))


def emit_commit(message: str, item_id: str = "") -> None:
    emit(WorkflowEvent(
        category=EventCategory.COMMIT,
        agent="system",
        title=f"📦 Committed: {message}",
        metadata={"commit_message": message, "item_id": item_id},
    ))


def emit_error(agent: str, error: str) -> None:
    emit(WorkflowEvent(
        category=EventCategory.ERROR,
        agent=agent,
        title=f"❌ Error in {agent}",
        detail=error,
    ))


def emit_approval_needed(pending: dict) -> None:
    """Emit when the human gate fires and the UI must show an approve/reject panel.

    ``pending`` is the full ``pending_approval`` dict from GraphState, which
    contains: summary, files, triggers, diff_preview, git_status, timestamp.
    """
    emit(WorkflowEvent(
        category=EventCategory.APPROVAL_NEEDED,
        agent="system",
        title="⚠️ Human approval required before commit",
        detail=pending.get("diff_preview", ""),
        metadata={
            "summary": pending.get("summary", ""),
            "files": pending.get("files", []),
            "triggers": pending.get("triggers", []),
            "git_status": pending.get("git_status", ""),
            "timestamp": pending.get("timestamp", ""),
        },
    ))


def emit_approval_done(approved: bool, pending_type: str = "commit") -> None:
    """Emit after the human has submitted an approve or reject decision."""
    icon = "✅" if approved else "❌"
    action = "approved" if approved else "rejected"
    emit(WorkflowEvent(
        category=EventCategory.APPROVAL_DONE,
        agent="system",
        title=f"{icon} Human {action} the {pending_type}",
        metadata={"approved": approved, "pending_type": pending_type},
    ))