"""Shared approval registry — decouples approval decisions from the interface that submitted the task.

Both the web server and the Telegram bot import this module.  When the human
gate fires (in any workflow, regardless of which interface started it), the
pending approval payload is registered here.  Either interface can then call
``approve()`` to resume or reject the workflow.

Usage
-----
::

    # nodes.py / human_gate_node sets the pending state:
    from app.core.approval_registry import registry
    registry.set_pending(payload, resume_callback)

    # web server or Telegram bot resolves it:
    registry.approve(approved=True)   # → calls resume_callback(True)
"""

from __future__ import annotations

import threading
from typing import Callable

from app.core.logging import get_logger

logger = get_logger("core.approval_registry")


class ApprovalRegistry:
    """Thread-safe singleton that holds the current pending approval request.

    Only one approval can be pending at a time (the workflow is halted while
    waiting).  Either the web server or the Telegram bot can call ``approve``
    to resolve it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._resume_callback: Callable[[bool], None] | None = None

    # ── Write side (set by the web server task worker) ────────────────

    def set_pending(
        self,
        pending_payload: dict,
        resume_callback: Callable[[bool], None],
    ) -> None:
        """Register a pending approval and the callback that will resume/stop the workflow.

        Args:
            pending_payload: The full ``pending_approval`` dict from GraphState
                             (contains summary, files, triggers, diff_preview, …).
            resume_callback: A callable that accepts a single bool (``approved``).
                             The registry calls this when ``approve()`` is invoked.
        """
        with self._lock:
            self._pending = dict(pending_payload)
            self._resume_callback = resume_callback
        logger.info(
            "Approval pending — %s",
            pending_payload.get("summary", "(no summary)"),
        )

    def clear(self) -> None:
        """Remove the pending approval without invoking the callback."""
        with self._lock:
            self._pending = None
            self._resume_callback = None

    # ── Read side (polled by web server and Telegram bot) ─────────────

    @property
    def is_pending(self) -> bool:
        """True if a human approval is currently waiting."""
        with self._lock:
            return self._pending is not None

    @property
    def pending(self) -> dict | None:
        """Return a copy of the current pending payload, or None."""
        with self._lock:
            return dict(self._pending) if self._pending is not None else None

    # ── Resolution ────────────────────────────────────────────────────

    def approve(self, approved: bool) -> bool:
        """Resolve the pending approval.

        Args:
            approved: ``True`` to approve and resume, ``False`` to reject and stop.

        Returns:
            ``True`` if there was a pending approval and it was resolved.
            ``False`` if nothing was pending (idempotent — safe to call twice).
        """
        with self._lock:
            if self._pending is None or self._resume_callback is None:
                logger.warning("approve() called but nothing is pending — ignoring")
                return False

            callback = self._resume_callback
            decision = "APPROVED" if approved else "REJECTED"
            self._pending = None
            self._resume_callback = None

        logger.info("Approval %s by human", decision)
        try:
            callback(approved)
        except Exception as exc:
            logger.error("Resume callback raised an exception: %s", exc, exc_info=True)
        return True


# Module-level singleton — imported by server.py and bot.py
registry = ApprovalRegistry()
