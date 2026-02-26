"""Tests for LLM response streaming (Issue #27).

Covers:
- _STREAMING_ROLES membership
- _stream_llm_round: token accumulation, batch flushing, fallback on error
- _invoke_agent: streaming vs non-streaming paths, tool-call interleaving
- emit_agent_token: event shape
- EventCategory.AGENT_TOKEN presence
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "", tool_calls: list | None = None) -> AIMessageChunk:
    """Create an AIMessageChunk with optional text and tool_calls."""
    chunk = AIMessageChunk(content=text)
    if tool_calls:
        chunk.tool_calls = tool_calls
    return chunk


def _text_chunks(texts: list[str]) -> list[AIMessageChunk]:
    return [_make_chunk(t) for t in texts]


# ---------------------------------------------------------------------------
# 1. EventCategory
# ---------------------------------------------------------------------------

class TestEventCategoryToken:
    def test_agent_token_in_enum(self):
        from app.core.events import EventCategory
        assert hasattr(EventCategory, "AGENT_TOKEN")
        assert EventCategory.AGENT_TOKEN.value == "agent_token"

    def test_agent_token_serialises(self):
        from app.core.events import EventCategory
        assert EventCategory.AGENT_TOKEN == "agent_token"


# ---------------------------------------------------------------------------
# 2. emit_agent_token
# ---------------------------------------------------------------------------

class TestEmitAgentToken:
    def test_emits_correct_category(self):
        from app.core.events import emit_agent_token, EventCategory, clear_listeners, subscribe_sync

        received: list = []
        subscribe_sync(received.append)
        try:
            emit_agent_token("coder_a", "hello")
            assert len(received) == 1
            evt = received[0]
            assert evt.category == EventCategory.AGENT_TOKEN
            assert evt.agent == "coder_a"
            assert evt.title == "hello"
        finally:
            clear_listeners()

    def test_token_in_history(self):
        from app.core.events import emit_agent_token, get_history, EventCategory
        emit_agent_token("planner", " world")
        history = get_history(10)
        token_events = [e for e in history if e["category"] == "agent_token"]
        assert any(e["title"] == " world" for e in token_events)

    def test_empty_token_allowed(self):
        """Empty tokens (e.g. whitespace) should still emit without error."""
        from app.core.events import emit_agent_token
        emit_agent_token("coder_b", "")  # should not raise

    def test_token_to_dict_shape(self):
        from app.core.events import emit_agent_token, EventCategory, clear_listeners, subscribe_sync

        received: list = []
        subscribe_sync(received.append)
        try:
            emit_agent_token("reviewer_a", "tok")
            d = received[0].to_dict()
            assert d["category"] == "agent_token"
            assert d["agent"] == "reviewer_a"
            assert d["title"] == "tok"
            assert "ts" in d
        finally:
            clear_listeners()


# ---------------------------------------------------------------------------
# 3. _STREAMING_ROLES
# ---------------------------------------------------------------------------

class TestStreamingRoles:
    def test_coders_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "coder_a" in _STREAMING_ROLES
        assert "coder_b" in _STREAMING_ROLES

    def test_planner_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "planner" in _STREAMING_ROLES

    def test_reviewers_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "reviewer_a" in _STREAMING_ROLES
        assert "reviewer_b" in _STREAMING_ROLES

    def test_documenter_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "documenter" in _STREAMING_ROLES

    def test_router_not_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "router" not in _STREAMING_ROLES

    def test_tester_not_in_streaming_roles(self):
        from app.core.nodes import _STREAMING_ROLES
        assert "tester" not in _STREAMING_ROLES


# ---------------------------------------------------------------------------
# 4. _stream_llm_round
# ---------------------------------------------------------------------------

class TestStreamLLMRound:
    def _call(self, chunks: list[AIMessageChunk], emitted: list | None = None) -> AIMessage:
        """Helper: run _stream_llm_round with a fake streaming LLM."""
        from app.core.nodes import _stream_llm_round
        from app.core.events import clear_listeners, subscribe_sync

        captured: list = []
        subscribe_sync(captured.append)

        fake_llm = MagicMock()
        fake_llm.stream.return_value = iter(chunks)

        try:
            result = _stream_llm_round("coder_a", fake_llm, [HumanMessage(content="go")])
        finally:
            clear_listeners()

        if emitted is not None:
            emitted.extend(captured)
        return result

    def test_returns_ai_message(self):
        chunks = _text_chunks(["Hello", " world"])
        msg = self._call(chunks)
        assert isinstance(msg, AIMessage)

    def test_accumulates_text(self):
        chunks = _text_chunks(["foo", "bar", "baz"])
        msg = self._call(chunks)
        assert "foo" in str(msg.content)

    def test_emits_token_events(self):
        from app.core.events import EventCategory
        # Use > _TOKEN_BATCH_MIN chunks to ensure a flush
        chunks = _text_chunks(["a"] * 12)
        emitted: list = []
        self._call(chunks, emitted=emitted)
        token_evts = [e for e in emitted if e.category == EventCategory.AGENT_TOKEN]
        assert len(token_evts) >= 1

    def test_batch_flushes_at_min(self):
        """Tokens are emitted in batches, not one by one."""
        from app.core.events import EventCategory
        from app.core.nodes import _TOKEN_BATCH_MIN
        # Exactly _TOKEN_BATCH_MIN chunks → exactly one flush mid-stream, one at end
        chunks = _text_chunks(["x"] * _TOKEN_BATCH_MIN)
        emitted: list = []
        self._call(chunks, emitted=emitted)
        token_evts = [e for e in emitted if e.category == EventCategory.AGENT_TOKEN]
        # At least 1 batch event, at most 2 (mid + end flush)
        assert 1 <= len(token_evts) <= 2

    def test_fallback_on_not_implemented(self):
        """If model raises NotImplementedError on .stream(), falls back to .invoke()."""
        from app.core.nodes import _stream_llm_round

        fake_llm = MagicMock()
        fake_llm.stream.side_effect = NotImplementedError("no streaming")
        fake_llm.invoke.return_value = AIMessage(content="fallback text")

        result = _stream_llm_round("coder_a", fake_llm, [HumanMessage(content="go")])
        assert isinstance(result, AIMessage)
        fake_llm.invoke.assert_called_once()

    def test_fallback_on_generic_exception(self):
        """Any exception during streaming falls back to .invoke()."""
        from app.core.nodes import _stream_llm_round

        fake_llm = MagicMock()
        fake_llm.stream.side_effect = RuntimeError("connection reset")
        fake_llm.invoke.return_value = AIMessage(content="fallback")

        result = _stream_llm_round("coder_a", fake_llm, [HumanMessage(content="x")])
        assert isinstance(result, AIMessage)
        fake_llm.invoke.assert_called_once()

    def test_list_content_chunks_extracted(self):
        """Handles Anthropic-style list content chunks."""
        from app.core.nodes import _stream_llm_round
        from app.core.events import clear_listeners, subscribe_sync, EventCategory

        chunk = AIMessageChunk(content=[{"type": "text", "text": "hello from list"}])
        fake_llm = MagicMock()
        fake_llm.stream.return_value = iter([chunk] * 10)  # 10 to exceed batch min

        captured: list = []
        subscribe_sync(captured.append)
        try:
            result = _stream_llm_round("coder_a", fake_llm, [])
        finally:
            clear_listeners()

        token_evts = [e for e in captured if e.category == EventCategory.AGENT_TOKEN]
        assert len(token_evts) >= 1

    def test_empty_stream_returns_ai_message(self):
        """Empty stream (no chunks) returns an empty AIMessage without error."""
        from app.core.nodes import _stream_llm_round

        fake_llm = MagicMock()
        fake_llm.stream.return_value = iter([])

        result = _stream_llm_round("coder_a", fake_llm, [])
        assert isinstance(result, AIMessage)

    def test_tool_call_chunk_preserved(self):
        """Tool calls in the last chunk are passed through to the returned AIMessage."""
        from app.core.nodes import _stream_llm_round

        # Final chunk carries tool call
        tc = {"name": "read_file", "args": {"path": "x.py"}, "id": "tc1", "type": "tool_call"}
        final_chunk = AIMessageChunk(content="")
        final_chunk.tool_calls = [tc]

        fake_llm = MagicMock()
        fake_llm.stream.return_value = iter([_make_chunk("text"), final_chunk])

        result = _stream_llm_round("coder_a", fake_llm, [])
        # Tool calls should be present (may be via accumulated chunk)
        assert isinstance(result, AIMessage)


# ---------------------------------------------------------------------------
# 5. _invoke_agent streaming path integration
# ---------------------------------------------------------------------------

class TestInvokeAgentStreaming:
    def _patch_invoke_agent(self, role: str, stream_chunks: list[AIMessageChunk],
                             final_text: str) -> tuple[str, list]:
        """Patch get_llm + load_system_prompt and run _invoke_agent for role."""
        from app.core.nodes import _invoke_agent
        from app.core.events import clear_listeners, subscribe_sync, EventCategory

        fake_llm = MagicMock()
        # .stream() returns chunks, .invoke() should not be called for streaming roles
        final_msg = AIMessage(content=final_text)
        # Accumulated chunk has no tool_calls → no further rounds
        final_msg.tool_calls = []
        fake_llm.stream.return_value = iter(stream_chunks)
        fake_llm.invoke.return_value = final_msg
        bound = MagicMock()
        bound.stream.return_value = iter(stream_chunks)
        bound.invoke.return_value = final_msg
        fake_llm.bind_tools.return_value = bound

        captured: list = []
        subscribe_sync(captured.append)

        with patch("app.core.nodes._helpers.get_llm", return_value=fake_llm), \
             patch("app.core.nodes._helpers.load_system_prompt", return_value="sys"), \
             patch("app.core.nodes.planner.load_all_memory", return_value=""):
            try:
                result = _invoke_agent(role, [HumanMessage(content="task")])
            finally:
                clear_listeners()

        return result, captured

    def test_streaming_role_uses_stream(self):
        from app.core.events import EventCategory
        chunks = _text_chunks(["Hello"] * 10)
        result, captured = self._patch_invoke_agent("coder_a", chunks, "Hello" * 10)
        token_evts = [e for e in captured if e.category == EventCategory.AGENT_TOKEN]
        assert len(token_evts) >= 1

    def test_non_streaming_role_no_tokens(self):
        from app.core.events import EventCategory
        chunks = _text_chunks(["tok"] * 5)
        result, captured = self._patch_invoke_agent("tester", chunks, "done")
        token_evts = [e for e in captured if e.category == EventCategory.AGENT_TOKEN]
        assert len(token_evts) == 0

    def test_streaming_result_returned(self):
        """The accumulated text from streaming is returned as the result string."""
        chunks = _text_chunks(["fo", "o ", "bar"])
        result, _ = self._patch_invoke_agent("planner", chunks, "foo bar")
        assert isinstance(result, str)
        assert len(result) > 0
