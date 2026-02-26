"""Tests for the Telegram bot — Phase B human-in-the-loop features.

We test the handler logic directly (without a real Telegram connection) by
constructing minimal Update/Context mock objects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_update(user_id: int = 1, text: str = "", args: list | None = None):
    """Build a minimal mock Update object."""
    update = MagicMock()
    update.effective_user.id = user_id
    update.message = AsyncMock()
    update.message.text = text
    update.callback_query = None
    return update


def _make_context(args: list | None = None):
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


# ── cmd_approve ───────────────────────────────────────────────────────────

class TestCmdApprove:
    @pytest.mark.asyncio
    async def test_approve_when_nothing_pending(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        # Use a fresh registry with nothing pending
        fresh_reg = ApprovalRegistry()
        with patch.object(bot, "approval_registry", fresh_reg):
            update = _make_update(user_id=1)
            ctx = _make_context()

            # Allow all users
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.cmd_approve(update, ctx)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Nothing is pending" in call_text or "nothing" in call_text.lower()

    @pytest.mark.asyncio
    async def test_approve_resolves_registry(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        decisions = []
        fresh_reg.set_pending({"type": "commit"}, decisions.append)

        with patch.object(bot, "approval_registry", fresh_reg):
            update = _make_update(user_id=1)
            ctx = _make_context()

            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.cmd_approve(update, ctx)

        assert decisions == [True]
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Approved" in call_text

    @pytest.mark.asyncio
    async def test_approve_blocked_for_unauthorized(self):
        from app.telegram import bot

        update = _make_update(user_id=999)
        ctx = _make_context()

        with patch("app.telegram.bot._is_allowed", return_value=False):
            await bot.cmd_approve(update, ctx)

        call_text = update.message.reply_text.call_args[0][0]
        assert "Not authorized" in call_text or "\u26d4" in call_text


# ── cmd_reject ────────────────────────────────────────────────────────────

class TestCmdReject:
    @pytest.mark.asyncio
    async def test_reject_resolves_registry_with_false(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        decisions = []
        fresh_reg.set_pending({"type": "commit"}, decisions.append)

        with patch.object(bot, "approval_registry", fresh_reg):
            update = _make_update(user_id=1)
            ctx = _make_context()

            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.cmd_reject(update, ctx)

        assert decisions == [False]
        call_text = update.message.reply_text.call_args[0][0]
        assert "Rejected" in call_text

    @pytest.mark.asyncio
    async def test_reject_when_nothing_pending(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        with patch.object(bot, "approval_registry", fresh_reg):
            update = _make_update(user_id=1)
            ctx = _make_context()
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.cmd_reject(update, ctx)

        call_text = update.message.reply_text.call_args[0][0]
        assert "pending" in call_text.lower() or "Nothing" in call_text


# ── Inline keyboard handler ───────────────────────────────────────────────

class TestInlineApproval:
    def _make_query(self, user_id: int, data: str):
        query = AsyncMock()
        query.from_user.id = user_id
        query.data = data
        return query

    @pytest.mark.asyncio
    async def test_inline_approve_resolves(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        decisions = []
        fresh_reg.set_pending({"type": "commit"}, decisions.append)

        update = MagicMock()
        update.callback_query = self._make_query(1, "approval:approve")

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.handle_inline_approval(update, _make_context())

        assert decisions == [True]

    @pytest.mark.asyncio
    async def test_inline_reject_resolves(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        decisions = []
        fresh_reg.set_pending({"type": "commit"}, decisions.append)

        update = MagicMock()
        update.callback_query = self._make_query(1, "approval:reject")

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.handle_inline_approval(update, _make_context())

        assert decisions == [False]

    @pytest.mark.asyncio
    async def test_inline_already_resolved_message(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()  # nothing pending

        update = MagicMock()
        update.callback_query = self._make_query(1, "approval:approve")

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.handle_inline_approval(update, _make_context())

        # Should call edit_message_text with a "resolved" message
        update.callback_query.edit_message_text.assert_called_once()
        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "resolved" in text.lower() or "pending" in text.lower()


# ── Notification helpers ──────────────────────────────────────────────────

class TestSendApprovalNotification:
    @pytest.mark.asyncio
    async def test_notification_contains_summary(self):
        from app.telegram import bot

        sent_messages = []

        async def _fake_notify(text, reply_markup=None):
            sent_messages.append(text)

        with patch.object(bot, "_notify_all", _fake_notify):
            await bot._send_approval_notification({
                "summary": "3 files changed, 80 lines",
                "files": ["app/core/nodes.py", "tests/test_nodes.py"],
                "triggers": [{"type": "commit", "reason": "Commit requires approval"}],
                "git_status": "M app/core/nodes.py",
            })

        assert len(sent_messages) == 1
        msg = sent_messages[0]
        assert "3 files changed" in msg
        assert "Commit requires approval" in msg
        assert "app/core/nodes.py" in msg

    @pytest.mark.asyncio
    async def test_notification_truncates_long_file_list(self):
        from app.telegram import bot

        sent_messages = []

        async def _fake_notify(text, reply_markup=None):
            sent_messages.append(text)

        files = [f"file_{i}.py" for i in range(30)]

        with patch.object(bot, "_notify_all", _fake_notify):
            await bot._send_approval_notification({
                "summary": "30 files",
                "files": files,
                "triggers": [],
                "git_status": "",
            })

        msg = sent_messages[0]
        assert "and 15 more" in msg  # 30 files, show 15, "and 15 more"


# ── create_telegram_app ───────────────────────────────────────────────────

class TestCreateTelegramApp:
    def test_returns_none_when_no_token(self):
        from app.telegram import bot

        with patch("app.telegram.bot.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = ""
            mock_settings.return_value.allowed_telegram_ids = []
            result = bot.create_telegram_app()

        assert result is None

    def test_registers_approve_reject_commands(self):
        """Verify /approve and /reject handlers are registered."""
        from app.telegram import bot
        from unittest.mock import MagicMock, patch

        mock_app = MagicMock()

        with patch("app.telegram.bot.get_settings") as mock_settings:
            mock_settings.return_value.telegram_bot_token = "fake:TOKEN"
            mock_settings.return_value.allowed_telegram_ids = []

            with patch("app.telegram.bot.Application") as mock_app_class:
                builder = MagicMock()
                mock_app_class.builder.return_value = builder
                builder.token.return_value = builder
                builder.build.return_value = mock_app

                bot.create_telegram_app()

        # Collect the command names registered via add_handler calls
        handler_args = [
            call.args[0]
            for call in mock_app.add_handler.call_args_list
        ]
        from telegram.ext import CommandHandler as CH
        command_names = []
        for h in handler_args:
            if hasattr(h, "commands"):
                command_names.extend(h.commands)

        assert "approve" in command_names
        assert "reject" in command_names


# ─────────────────────────────────────────────────────────────────────────────
# Bug regression: _on_workflow_event — COMMIT event with None metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestOnWorkflowEventNoneMetadata:
    """Regression tests for metadata None-guard in _on_workflow_event (bug #56)."""

    def _make_event(self, category_value: str, metadata):
        from app.core.events import WorkflowEvent, EventCategory
        return WorkflowEvent(
            category=EventCategory(category_value),
            agent="system",
            title="test",
            metadata=metadata,
        )

    def test_commit_event_none_metadata_does_not_raise(self):
        """COMMIT event with metadata=None must not raise AttributeError."""
        from app.telegram.bot import _on_workflow_event
        import app.telegram.bot as bot_module

        # _on_workflow_event exits early if _telegram_app is None
        original = bot_module._telegram_app
        bot_module._telegram_app = None  # ensure early exit path
        try:
            event = self._make_event("commit", None)
            # Should not raise
            _on_workflow_event(event)
        finally:
            bot_module._telegram_app = original

    def test_commit_event_without_pr_url_does_not_trigger_pr_notification(self):
        """A regular commit (no pr_url in metadata) must not call _send_pr_notification."""
        from app.telegram.bot import _on_workflow_event
        import app.telegram.bot as bot_module
        from unittest.mock import MagicMock, patch

        original = bot_module._telegram_app
        bot_module._telegram_app = None
        try:
            event = self._make_event("commit", {"commit_message": "fix: something"})
            with patch.object(bot_module, "_send_pr_notification") as mock_pr:
                _on_workflow_event(event)
                mock_pr.assert_not_called()
        finally:
            bot_module._telegram_app = original


# ─────────────────────────────────────────────────────────────────────────────
# Bug regression: issue_loaded event platform field
# ─────────────────────────────────────────────────────────────────────────────

class TestIssueLoadedEventPlatform:
    """Regression test: issue_loaded event must include platform field (bug #56)."""

    def test_issue_loaded_event_contains_platform(self):
        """_hydrate_issue must emit issue_loaded with a platform field."""
        import re
        nodes_src = open("app/core/nodes.py").read()
        idx = nodes_src.find('title="issue_loaded"')
        assert idx != -1, "issue_loaded event not found in nodes.py"
        # Find the metadata block following this title
        meta_start = nodes_src.find("metadata={", idx)
        meta_end = nodes_src.find("},", meta_start) + 2
        meta_block = nodes_src[meta_start:meta_end]
        assert "platform" in meta_block, (
            "issue_loaded event metadata must include 'platform' field for "
            "correct UI icon (🐙 GitHub vs 🦊 GitLab)"
        )
