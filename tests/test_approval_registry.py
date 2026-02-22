"""Tests for the shared ApprovalRegistry."""

from __future__ import annotations

import threading

import pytest

from app.core.approval_registry import ApprovalRegistry


# Each test gets a fresh registry instance (not the module singleton)
@pytest.fixture
def reg():
    return ApprovalRegistry()


DUMMY_PAYLOAD = {
    "type": "commit",
    "summary": "2 files changed",
    "files": ["app/main.py", "tests/test_main.py"],
    "triggers": [{"type": "commit", "reason": "Commit requires approval"}],
    "diff_preview": "--- a\n+++ b\n@@ -1 +1 @@\n+new",
    "git_status": "M app/main.py",
    "timestamp": "2026-01-01T00:00:00Z",
}


class TestRegistryDefaultState:
    def test_not_pending_initially(self, reg):
        assert reg.is_pending is False

    def test_pending_returns_none_initially(self, reg):
        assert reg.pending is None


class TestSetPending:
    def test_is_pending_after_set(self, reg):
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: None)
        assert reg.is_pending is True

    def test_pending_returns_copy(self, reg):
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: None)
        p = reg.pending
        assert p is not None
        assert p["summary"] == "2 files changed"
        # Mutating the returned copy must not affect the stored payload
        p["summary"] = "mutated"
        assert reg.pending["summary"] == "2 files changed"

    def test_second_set_overwrites_first(self, reg):
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: None)
        new_payload = dict(DUMMY_PAYLOAD, summary="5 files changed")
        reg.set_pending(new_payload, lambda approved: None)
        assert reg.pending["summary"] == "5 files changed"


class TestApprove:
    def test_approve_true_calls_callback_with_true(self, reg):
        received = []
        reg.set_pending(DUMMY_PAYLOAD, received.append)
        result = reg.approve(approved=True)
        assert result is True
        assert received == [True]

    def test_approve_false_calls_callback_with_false(self, reg):
        received = []
        reg.set_pending(DUMMY_PAYLOAD, received.append)
        result = reg.approve(approved=False)
        assert result is True
        assert received == [False]

    def test_approve_clears_pending(self, reg):
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: None)
        reg.approve(approved=True)
        assert reg.is_pending is False
        assert reg.pending is None

    def test_approve_without_pending_returns_false(self, reg):
        result = reg.approve(approved=True)
        assert result is False

    def test_second_approve_is_idempotent(self, reg):
        calls = []
        reg.set_pending(DUMMY_PAYLOAD, calls.append)
        reg.approve(approved=True)
        result = reg.approve(approved=True)  # second call — nothing pending
        assert result is False
        assert len(calls) == 1  # callback was only called once


class TestClear:
    def test_clear_removes_pending(self, reg):
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: None)
        reg.clear()
        assert reg.is_pending is False

    def test_clear_does_not_call_callback(self, reg):
        calls = []
        reg.set_pending(DUMMY_PAYLOAD, calls.append)
        reg.clear()
        assert calls == []


class TestThreadSafety:
    def test_concurrent_approvals_only_one_wins(self, reg):
        """Two threads calling approve() simultaneously — only one should win."""
        results = []
        reg.set_pending(DUMMY_PAYLOAD, lambda approved: results.append(approved))

        barrier = threading.Barrier(2)

        def _approve():
            barrier.wait()
            results.append(reg.approve(approved=True))

        t1 = threading.Thread(target=_approve)
        t2 = threading.Thread(target=_approve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one True from the callback, one False from the losing approve()
        callback_calls = [r for r in results if r is True and results.count(True) > 0]
        # The registry callback was called exactly once (the winning thread)
        # and one of the approve() calls returned False.
        approve_return_values = results[-2:]  # last two appended by _approve()
        assert sorted(approve_return_values) == [False, True]
