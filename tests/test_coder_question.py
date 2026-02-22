"""Tests for Phase C — mid-task coder questions.

Covers:
  - _parse_coder_question() helper in nodes.py
  - answer_gate_node routing
  - emit_coder_question / emit_coder_answer events
  - GraphState coder question fields
  - approval_registry answer API (set_answer_pending / deliver_answer)
  - /api/answer and /api/question HTTP endpoints
  - Telegram cmd_answer and handle_inline_answer
  - Orchestrator routing (_route_after_coder, _route_after_answer_gate)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── _parse_coder_question ─────────────────────────────────────────────────

class TestParseCoderQuestion:
    def _parse(self, text: str):
        from app.core.nodes import _parse_coder_question
        return _parse_coder_question(text)

    def test_valid_json_returns_payload(self):
        payload = json.dumps({
            "action": "ask_human",
            "question": "Redis or SQLite?",
            "context": "Redis is distributed, SQLite is simpler.",
            "options": ["Redis", "SQLite", "Decide for me"],
        })
        result = self._parse(payload)
        assert result is not None
        assert result["question"] == "Redis or SQLite?"
        assert result["context"] == "Redis is distributed, SQLite is simpler."
        assert result["options"] == ["Redis", "SQLite", "Decide for me"]

    def test_json_fenced_in_markdown_is_parsed(self):
        payload = "```json\n" + json.dumps({
            "action": "ask_human",
            "question": "Which database?",
        }) + "\n```"
        result = self._parse(payload)
        assert result is not None
        assert result["question"] == "Which database?"

    def test_plain_fence_without_language_tag(self):
        payload = "```\n" + json.dumps({
            "action": "ask_human",
            "question": "Plain fence question?",
        }) + "\n```"
        result = self._parse(payload)
        assert result is not None
        assert result["question"] == "Plain fence question?"

    def test_wrong_action_returns_none(self):
        payload = json.dumps({"action": "implement", "question": "Does not matter"})
        assert self._parse(payload) is None

    def test_missing_question_returns_none(self):
        payload = json.dumps({"action": "ask_human", "question": ""})
        assert self._parse(payload) is None

    def test_normal_code_prose_returns_none(self):
        text = (
            "## Summary\nI added a new endpoint `GET /api/health`.\n\n"
            "## Files Modified\n- app/web/server.py\n\n"
            "## Suggested Commit Message\n`feat(web): add health check endpoint`"
        )
        assert self._parse(text) is None

    def test_options_defaults_to_empty_list(self):
        payload = json.dumps({"action": "ask_human", "question": "Which pattern?"})
        result = self._parse(payload)
        assert result is not None
        assert result["options"] == []

    def test_invalid_json_returns_none(self):
        assert self._parse("{not valid json}") is None

    def test_empty_string_returns_none(self):
        assert self._parse("") is None


# ── GraphState coder question fields ─────────────────────────────────────

class TestGraphStateCoderQuestionFields:
    def test_default_values(self):
        from app.core.state import GraphState
        state = GraphState(user_request="test")
        assert state.needs_coder_answer is False
        assert state.coder_question == ""
        assert state.coder_question_context == ""
        assert state.coder_question_options == []
        assert state.coder_question_asked_by == ""
        assert state.coder_question_answer == ""

    def test_fields_can_be_set(self):
        from app.core.state import GraphState
        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
            coder_question_context="Trade-off between Redis and SQLite",
            coder_question_options=["Redis", "SQLite"],
            coder_question_asked_by="coder_a",
            coder_question_answer="Redis please",
        )
        assert state.needs_coder_answer is True
        assert state.coder_question == "Which DB?"
        assert state.coder_question_options == ["Redis", "SQLite"]
        assert state.coder_question_answer == "Redis please"

    def test_waiting_for_answer_phase_exists(self):
        from app.core.state import WorkflowPhase
        assert WorkflowPhase.WAITING_FOR_ANSWER == "waiting_for_answer"


# ── answer_gate_node ──────────────────────────────────────────────────────

class TestAnswerGateNode:
    def test_halts_when_no_answer(self):
        from app.core.nodes import answer_gate_node
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
            coder_question_answer="",   # not yet answered
        )
        result = answer_gate_node(state)
        assert result["phase"] == WorkflowPhase.WAITING_FOR_ANSWER
        assert result["stop_reason"] == "waiting_for_coder_answer"

    def test_continues_when_answer_present(self):
        from app.core.nodes import answer_gate_node
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
            coder_question_answer="Redis please",
        )
        result = answer_gate_node(state)
        assert result["phase"] == WorkflowPhase.CODING
        assert result["needs_coder_answer"] is False
        assert result["stop_reason"] == ""


# ── Orchestrator routing ──────────────────────────────────────────────────

class TestOrchestratorRouting:
    def test_route_after_coder_to_answer_gate(self):
        from app.core.orchestrator import _route_after_coder
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            phase=WorkflowPhase.WAITING_FOR_ANSWER,
        )
        assert _route_after_coder(state) == "answer_gate"

    def test_route_after_coder_to_peer_review(self):
        from app.core.orchestrator import _route_after_coder
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(user_request="test", phase=WorkflowPhase.PEER_REVIEWING)
        assert _route_after_coder(state) == "peer_review"

    def test_route_after_answer_gate_stopped(self):
        from app.core.orchestrator import _route_after_answer_gate
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            phase=WorkflowPhase.WAITING_FOR_ANSWER,
        )
        assert _route_after_answer_gate(state) == "stopped"

    def test_route_after_answer_gate_coder(self):
        from app.core.orchestrator import _route_after_answer_gate
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(user_request="test", phase=WorkflowPhase.CODING)
        assert _route_after_answer_gate(state) == "coder"

    def test_route_after_resume_to_answer_gate(self):
        from app.core.orchestrator import _route_after_resume
        from app.core.state import GraphState, WorkflowPhase

        state = GraphState(
            user_request="test",
            phase=WorkflowPhase.WAITING_FOR_ANSWER,
        )
        assert _route_after_resume(state) == "answer_gate"


# ── Events ────────────────────────────────────────────────────────────────

class TestCoderQuestionEvents:
    def _clear(self):
        import app.core.events as ev
        ev._history.clear()

    def test_emit_coder_question_category(self):
        from app.core.events import emit_coder_question, get_history, EventCategory
        self._clear()
        emit_coder_question(
            asked_by="coder_a",
            question="Which DB?",
            context="Redis vs SQLite trade-off",
            options=["Redis", "SQLite"],
            item_id="item_0",
        )
        evt = get_history(1)[0]
        assert evt["category"] == EventCategory.CODER_QUESTION.value
        assert evt["agent"] == "coder_a"
        assert evt["metadata"]["question"] == "Which DB?"
        assert evt["metadata"]["options"] == ["Redis", "SQLite"]
        assert evt["metadata"]["item_id"] == "item_0"

    def test_emit_coder_answer_category(self):
        from app.core.events import emit_coder_answer, get_history, EventCategory
        self._clear()
        emit_coder_answer(asked_by="coder_b", answer="Redis please", item_id="item_1")
        evt = get_history(1)[0]
        assert evt["category"] == EventCategory.CODER_ANSWER.value
        assert evt["metadata"]["answer"] == "Redis please"
        assert evt["metadata"]["asked_by"] == "coder_b"

    def test_emit_coder_question_no_options(self):
        from app.core.events import emit_coder_question, get_history, EventCategory
        self._clear()
        emit_coder_question(asked_by="coder_a", question="Open ended question?")
        evt = get_history(1)[0]
        assert evt["metadata"]["options"] == []


# ── ApprovalRegistry answer API ───────────────────────────────────────────

class TestApprovalRegistryAnswerAPI:
    def _reg(self):
        from app.core.approval_registry import ApprovalRegistry
        return ApprovalRegistry()

    def test_not_question_pending_initially(self):
        reg = self._reg()
        assert reg.is_question_pending is False
        assert reg.pending_question is None

    def test_set_answer_pending_makes_it_pending(self):
        reg = self._reg()
        reg.set_answer_pending({"question": "Which DB?"}, lambda a: None)
        assert reg.is_question_pending is True
        assert reg.pending_question["question"] == "Which DB?"

    def test_deliver_answer_calls_callback(self):
        reg = self._reg()
        received = []
        reg.set_answer_pending({"question": "Q?"}, received.append)
        result = reg.deliver_answer("Redis")
        assert result is True
        assert received == ["Redis"]

    def test_deliver_answer_clears_pending(self):
        reg = self._reg()
        reg.set_answer_pending({"question": "Q?"}, lambda a: None)
        reg.deliver_answer("A")
        assert reg.is_question_pending is False

    def test_deliver_answer_without_pending_returns_false(self):
        reg = self._reg()
        result = reg.deliver_answer("anything")
        assert result is False

    def test_clear_question_removes_pending(self):
        reg = self._reg()
        reg.set_answer_pending({"question": "Q?"}, lambda a: None)
        reg.clear_question()
        assert reg.is_question_pending is False

    def test_deliver_answer_idempotent(self):
        reg = self._reg()
        calls = []
        reg.set_answer_pending({"question": "Q?"}, calls.append)
        reg.deliver_answer("first")
        result = reg.deliver_answer("second")  # nothing pending now
        assert result is False
        assert calls == ["first"]


# ── /api/answer and /api/question endpoints ───────────────────────────────

@pytest.fixture
def client():
    with patch("app.web.server.get_settings") as ms:
        ms.return_value.target_repo_path = "/tmp/test-repo"
        ms.return_value.max_output_chars = 10000
        with patch("app.web.server.run_workflow", new_callable=AsyncMock):
            from fastapi.testclient import TestClient
            from app.web.server import app
            with TestClient(app) as c:
                yield c


class TestAnswerEndpoint:
    def test_answer_without_pending_returns_error(self, client):
        resp = client.post("/api/answer", json={"answer": "Redis"})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_answer_empty_string_returns_error(self, client):
        from app.core.state import GraphState, WorkflowPhase
        import app.web.server as srv

        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
        )
        original = srv._current_state
        srv._current_state = state
        try:
            resp = client.post("/api/answer", json={"answer": "   "})
            data = resp.json()
            assert "error" in data
        finally:
            srv._current_state = original

    def test_answer_delivered_via_registry(self, client):
        from app.core.state import GraphState, WorkflowPhase
        from app.core.approval_registry import ApprovalRegistry
        import app.web.server as srv

        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
            coder_question_asked_by="coder_a",
        )
        original_state = srv._current_state
        original_reg = srv.approval_registry
        srv._current_state = state

        fresh_reg = ApprovalRegistry()
        delivered = []
        fresh_reg.set_answer_pending({"question": "Which DB?"}, delivered.append)
        srv.approval_registry = fresh_reg

        try:
            resp = client.post("/api/answer", json={"answer": "Redis"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "answer_submitted"
            assert delivered == ["Redis"]
        finally:
            srv._current_state = original_state
            srv.approval_registry = original_reg


class TestQuestionEndpoint:
    def test_question_when_idle(self, client):
        import app.web.server as srv
        original = srv._current_state
        srv._current_state = None
        try:
            resp = client.get("/api/question")
            assert resp.status_code == 200
            data = resp.json()
            assert data["needs_coder_answer"] is False
        finally:
            srv._current_state = original

    def test_question_when_pending(self, client):
        from app.core.state import GraphState, WorkflowPhase
        import app.web.server as srv

        state = GraphState(
            user_request="test",
            needs_coder_answer=True,
            coder_question="Which DB?",
            coder_question_context="Redis vs SQLite",
            coder_question_options=["Redis", "SQLite"],
            coder_question_asked_by="coder_a",
        )
        original = srv._current_state
        srv._current_state = state
        try:
            resp = client.get("/api/question")
            assert resp.status_code == 200
            data = resp.json()
            assert data["needs_coder_answer"] is True
            assert data["coder_question"]["question"] == "Which DB?"
            assert "Redis" in data["coder_question"]["options"]
        finally:
            srv._current_state = original


# ── Telegram cmd_answer ───────────────────────────────────────────────────

class TestCmdAnswer:
    def _make_update(self, user_id=1, args=None):
        update = MagicMock()
        update.effective_user.id = user_id
        update.message = AsyncMock()
        return update

    def _make_ctx(self, args=None):
        ctx = MagicMock()
        ctx.args = args or []
        return ctx

    @pytest.mark.asyncio
    async def test_answer_when_nothing_pending(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry
        fresh_reg = ApprovalRegistry()

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                update = self._make_update()
                await bot.cmd_answer(update, self._make_ctx(["Redis"]))

        text = update.message.reply_text.call_args[0][0]
        assert "pending" in text.lower() or "No coder" in text

    @pytest.mark.asyncio
    async def test_answer_no_args_shows_usage(self):
        from app.telegram import bot
        with patch("app.telegram.bot._is_allowed", return_value=True):
            update = self._make_update()
            await bot.cmd_answer(update, self._make_ctx([]))

        text = update.message.reply_text.call_args[0][0]
        assert "Usage" in text or "answer" in text.lower()

    @pytest.mark.asyncio
    async def test_answer_delivers_via_registry(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        delivered = []
        fresh_reg.set_answer_pending({"question": "Q?"}, delivered.append)

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                update = self._make_update()
                await bot.cmd_answer(update, self._make_ctx(["Redis", "please"]))

        assert delivered == ["Redis please"]
        text = update.message.reply_text.call_args[0][0]
        assert "delivered" in text.lower() or "Answer" in text

    @pytest.mark.asyncio
    async def test_answer_unauthorized(self):
        from app.telegram import bot
        with patch("app.telegram.bot._is_allowed", return_value=False):
            update = self._make_update(user_id=999)
            await bot.cmd_answer(update, self._make_ctx(["Redis"]))

        text = update.message.reply_text.call_args[0][0]
        assert "authorized" in text.lower() or "\u26d4" in text


# ── Telegram handle_inline_answer ─────────────────────────────────────────

class TestHandleInlineAnswer:
    @pytest.mark.asyncio
    async def test_inline_answer_delivers(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()
        delivered = []
        fresh_reg.set_answer_pending({"question": "Q?"}, delivered.append)

        update = MagicMock()
        update.callback_query = AsyncMock()
        update.callback_query.from_user.id = 1
        update.callback_query.data = "answer:0:Redis"

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.handle_inline_answer(update, MagicMock())

        assert delivered == ["Redis"]

    @pytest.mark.asyncio
    async def test_inline_answer_already_resolved(self):
        from app.telegram import bot
        from app.core.approval_registry import ApprovalRegistry

        fresh_reg = ApprovalRegistry()  # nothing pending

        update = MagicMock()
        update.callback_query = AsyncMock()
        update.callback_query.from_user.id = 1
        update.callback_query.data = "answer:0:Redis"

        with patch.object(bot, "approval_registry", fresh_reg):
            with patch("app.telegram.bot._is_allowed", return_value=True):
                await bot.handle_inline_answer(update, MagicMock())

        text = update.callback_query.edit_message_text.call_args[0][0]
        assert "answered" in text.lower() or "pending" in text.lower()


# ── Telegram create_telegram_app registers /answer ────────────────────────

class TestTelegramAppRegistersAnswer:
    def test_answer_command_registered(self):
        from app.telegram import bot

        mock_app = MagicMock()
        with patch("app.telegram.bot.get_settings") as ms:
            ms.return_value.telegram_bot_token = "fake:TOKEN"
            ms.return_value.allowed_telegram_ids = []
            with patch("app.telegram.bot.Application") as mock_cls:
                builder = MagicMock()
                mock_cls.builder.return_value = builder
                builder.token.return_value = builder
                builder.build.return_value = mock_app
                bot.create_telegram_app()

        handler_args = [call.args[0] for call in mock_app.add_handler.call_args_list]
        command_names = []
        for h in handler_args:
            if hasattr(h, "commands"):
                command_names.extend(h.commands)

        assert "answer" in command_names
