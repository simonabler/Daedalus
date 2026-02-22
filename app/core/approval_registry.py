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
    """Thread-safe singleton that holds the current pending approval request
    and/or a pending coder question that requires a human answer.

    Only one of each can be pending at a time.  Either the web server or the
    Telegram bot can resolve them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # ── Approval (approve/reject) ──────────────────────────────────
        self._pending: dict | None = None
        self._resume_callback: Callable[[bool], None] | None = None
        # ── Coder question (free-text answer) ─────────────────────────
        self._pending_question: dict | None = None
        self._answer_callback: Callable[[str], None] | None = None

    # ── Approval write side ───────────────────────────────────────────

    def set_pending(
        self,
        pending_payload: dict,
        resume_callback: Callable[[bool], None],
    ) -> None:
        """Register a pending approval and the callback that will resume/stop the workflow."""
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

    # ── Approval read side ────────────────────────────────────────────

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

    # ── Approval resolution ───────────────────────────────────────────

    def approve(self, approved: bool) -> bool:
        """Resolve the pending approval.

        Returns True if there was a pending approval and it was resolved,
        False if nothing was pending (safe to call twice).
        """
        with self._lock:
            if self._pending is None or self._resume_callback is None:
                logger.warning("approve() called but nothing is pending — ignoring")
                return False
            callback = self._resume_callback
            self._pending = None
            self._resume_callback = None

        decision = "APPROVED" if approved else "REJECTED"
        logger.info("Approval %s by human", decision)
        try:
            callback(approved)
        except Exception as exc:
            logger.error("Resume callback raised an exception: %s", exc, exc_info=True)
        return True

    # ── Coder question write side ─────────────────────────────────────

    def set_answer_pending(
        self,
        question: dict,
        answer_callback: Callable[[str], None],
    ) -> None:
        """Register a pending coder question and the callback that delivers the answer."""
        with self._lock:
            self._pending_question = dict(question)
            self._answer_callback = answer_callback
        logger.info("Coder question pending — %s", question.get("question", "")[:80])

    def clear_question(self) -> None:
        """Remove the pending question without delivering an answer."""
        with self._lock:
            self._pending_question = None
            self._answer_callback = None

    # ── Coder question read side ──────────────────────────────────────

    @property
    def is_question_pending(self) -> bool:
        """True if a coder question is waiting for a human answer."""
        with self._lock:
            return self._pending_question is not None

    @property
    def pending_question(self) -> dict | None:
        """Return a copy of the current question payload, or None."""
        with self._lock:
            return dict(self._pending_question) if self._pending_question is not None else None

    # ── Coder question resolution ─────────────────────────────────────

    def deliver_answer(self, answer: str) -> bool:
        """Deliver the human's answer to the waiting coder.

        Returns True if there was a pending question and the answer was
        delivered, False if nothing was pending.
        """
        with self._lock:
            if self._pending_question is None or self._answer_callback is None:
                logger.warning("deliver_answer() called but no question is pending — ignoring")
                return False
            callback = self._answer_callback
            self._pending_question = None
            self._answer_callback = None

        logger.info("Coder question answered by human: %s", answer[:80])
        try:
            callback(answer)
        except Exception as exc:
            logger.error("Answer callback raised an exception: %s", exc, exc_info=True)
        return True


# Module-level singleton — imported by server.py and bot.py
registry = ApprovalRegistry()
