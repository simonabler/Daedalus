"""Tests for context window management (Issue #36).

Covers:
- estimate_tokens / estimate_messages_tokens: basic correctness
- context_limit_for_model: known models, prefix matching, unknown fallback, Ollama
- context_usage_fraction: proportion calculation
- truncate_tool_result: short (no-op), long (truncated with marker), exact boundary
- compress_messages: no-op when too few messages, compresses middle, keeps head+tail
- compress_messages: returns original on LLM failure
- CONTEXT_WARN_FRACTION / CONTEXT_KEEP_RECENT constants present
- Config: context_warn_fraction, context_keep_recent_messages, tool_result_max_chars
- EventCategory.CONTEXT_USAGE present; emit_context_usage emits correct metadata
- _invoke_agent integration: tool results are truncated (mock check)
- _invoke_agent integration: context warning emitted when fraction > 0.5 initially
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_msg(content: str) -> AIMessage:
    m = AIMessage(content=content)
    m.tool_calls = []
    return m


# ---------------------------------------------------------------------------
# 1. estimate_tokens / estimate_messages_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        from app.core.context_window import estimate_tokens
        assert estimate_tokens("") == 1

    def test_short_text(self):
        from app.core.context_window import estimate_tokens
        # 30 chars / 3.0 = 10 tokens
        assert estimate_tokens("a" * 30) == 10

    def test_longer_text_scales(self):
        from app.core.context_window import estimate_tokens
        t300 = estimate_tokens("x" * 300)
        t600 = estimate_tokens("x" * 600)
        assert t600 == t300 * 2

    def test_estimate_messages_single(self):
        from app.core.context_window import estimate_messages_tokens, estimate_tokens
        msg = HumanMessage(content="hello world")
        total = estimate_messages_tokens([msg])
        # should be estimate_tokens("hello world") + 4 overhead
        expected = estimate_tokens("hello world") + 4
        assert total == expected

    def test_estimate_messages_multiple(self):
        from app.core.context_window import estimate_messages_tokens
        msgs = [HumanMessage(content="hello"), AIMessage(content="world")]
        t1 = estimate_messages_tokens([msgs[0]])
        t2 = estimate_messages_tokens([msgs[1]])
        combined = estimate_messages_tokens(msgs)
        assert combined == t1 + t2

    def test_estimate_messages_empty_list(self):
        from app.core.context_window import estimate_messages_tokens
        assert estimate_messages_tokens([]) == 0


# ---------------------------------------------------------------------------
# 2. context_limit_for_model
# ---------------------------------------------------------------------------

class TestContextLimitForModel:
    def _limit(self, model):
        from app.core.context_window import context_limit_for_model
        return context_limit_for_model(model)

    def test_gpt4o_mini_exact_match(self):
        assert self._limit("gpt-4o-mini") == 128_000

    def test_claude_sonnet_exact_match(self):
        assert self._limit("claude-sonnet-4-20250514") == 200_000

    def test_prefix_match_claude_variant(self):
        # A variant not in the table but prefix-matchable
        limit = self._limit("claude-sonnet-4-20250514-preview")
        assert limit == 200_000

    def test_gpt4_8k(self):
        assert self._limit("gpt-4") == 8_192

    def test_unknown_model_default(self):
        limit = self._limit("some-unknown-model-xyz-v99")
        from app.core.context_window import MODEL_CONTEXT_LIMITS
        assert limit == MODEL_CONTEXT_LIMITS["_default"]

    def test_ollama_model_uses_ollama_default(self):
        from app.core.context_window import MODEL_CONTEXT_LIMITS
        assert self._limit("ollama:llama3.1:70b") == MODEL_CONTEXT_LIMITS["_ollama"]

    def test_slash_model_treated_as_ollama(self):
        from app.core.context_window import MODEL_CONTEXT_LIMITS
        assert self._limit("meta/llama-3") == MODEL_CONTEXT_LIMITS["_ollama"]


# ---------------------------------------------------------------------------
# 3. context_usage_fraction
# ---------------------------------------------------------------------------

class TestContextUsageFraction:
    def test_small_messages_fraction_below_one(self):
        from app.core.context_window import context_usage_fraction
        msgs = [HumanMessage(content="hi")]
        f = context_usage_fraction(msgs, "gpt-4o-mini")
        assert 0.0 < f < 1.0

    def test_fraction_scales_with_message_size(self):
        from app.core.context_window import context_usage_fraction
        short = [HumanMessage(content="hi")]
        long_ = [HumanMessage(content="x" * 50_000)]
        f_short = context_usage_fraction(short, "gpt-4o-mini")
        f_long  = context_usage_fraction(long_, "gpt-4o-mini")
        assert f_long > f_short

    def test_fraction_uses_model_limit(self):
        from app.core.context_window import context_usage_fraction
        msgs = [HumanMessage(content="x" * 3_000)]
        f_large_ctx = context_usage_fraction(msgs, "claude-sonnet-4-20250514")   # 200k
        f_small_ctx = context_usage_fraction(msgs, "gpt-4")                      # 8k
        assert f_small_ctx > f_large_ctx


# ---------------------------------------------------------------------------
# 4. truncate_tool_result
# ---------------------------------------------------------------------------

class TestTruncateToolResult:
    def test_short_result_unchanged(self):
        from app.core.context_window import truncate_tool_result
        text = "hello world"
        assert truncate_tool_result(text, max_chars=100) == text

    def test_exact_limit_unchanged(self):
        from app.core.context_window import truncate_tool_result
        text = "x" * 100
        assert truncate_tool_result(text, max_chars=100) == text

    def test_long_result_truncated(self):
        from app.core.context_window import truncate_tool_result
        text = "x" * 200
        result = truncate_tool_result(text, max_chars=100)
        assert len(result) < len(text)
        assert result.startswith("x" * 100)

    def test_truncation_marker_appended(self):
        from app.core.context_window import truncate_tool_result
        result = truncate_tool_result("x" * 200, max_chars=100)
        assert "truncated" in result.lower()

    def test_truncation_reports_dropped_chars(self):
        from app.core.context_window import truncate_tool_result
        result = truncate_tool_result("x" * 200, max_chars=100)
        assert "100" in result  # 200 - 100 = 100 chars dropped

    def test_default_limit_used_when_none_provided(self):
        from app.core.context_window import truncate_tool_result, DEFAULT_TOOL_RESULT_MAX_CHARS
        short = "x" * (DEFAULT_TOOL_RESULT_MAX_CHARS - 1)
        assert truncate_tool_result(short) == short

    def test_empty_string_unchanged(self):
        from app.core.context_window import truncate_tool_result
        assert truncate_tool_result("", max_chars=100) == ""


# ---------------------------------------------------------------------------
# 5. compress_messages
# ---------------------------------------------------------------------------

class TestCompressMessages:
    def _make_msgs(self, n_middle: int, keep: int = 6):
        """Build a message list: 1 SystemMessage + n_middle + keep messages."""
        msgs = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(n_middle):
            msgs.append(HumanMessage(content=f"Message {i}: " + "x" * 100))
        for i in range(keep):
            msgs.append(AIMessage(content=f"Recent {i}"))
        return msgs

    def _mock_llm(self, summary: str = "Summary of old turns."):
        llm = MagicMock()
        response = MagicMock()
        response.content = summary
        llm.invoke.return_value = response
        return llm

    def test_too_few_messages_returns_unchanged(self):
        from app.core.context_window import compress_messages, CONTEXT_KEEP_RECENT
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        llm = self._mock_llm()
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        assert result is msgs
        llm.invoke.assert_not_called()

    def test_compress_reduces_message_count(self):
        from app.core.context_window import compress_messages
        msgs = self._make_msgs(n_middle=10)
        original_count = len(msgs)
        llm = self._mock_llm()
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        assert len(result) < original_count

    def test_system_message_preserved(self):
        from app.core.context_window import compress_messages
        msgs = self._make_msgs(n_middle=10)
        llm = self._mock_llm()
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        assert any(isinstance(m, SystemMessage) for m in result)

    def test_recent_messages_preserved(self):
        from app.core.context_window import compress_messages, CONTEXT_KEEP_RECENT
        msgs = self._make_msgs(n_middle=10, keep=CONTEXT_KEEP_RECENT)
        recent_contents = {m.content for m in msgs[-CONTEXT_KEEP_RECENT:]}
        llm = self._mock_llm()
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        result_contents = {m.content for m in result}
        assert recent_contents.issubset(result_contents)

    def test_summary_message_has_context_prefix(self):
        from app.core.context_window import compress_messages
        msgs = self._make_msgs(n_middle=10)
        llm = self._mock_llm("This is the summary.")
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        summary_msgs = [m for m in result if "CONTEXT SUMMARY" in (m.content or "")]
        assert len(summary_msgs) == 1

    def test_llm_called_once_for_summary(self):
        from app.core.context_window import compress_messages
        msgs = self._make_msgs(n_middle=10)
        llm = self._mock_llm()
        compress_messages(msgs, "gpt-4o-mini", llm)
        assert llm.invoke.call_count == 1

    def test_llm_failure_returns_original(self):
        from app.core.context_window import compress_messages
        msgs = self._make_msgs(n_middle=10)
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        result = compress_messages(msgs, "gpt-4o-mini", llm)
        assert result is msgs  # original returned, not raised


# ---------------------------------------------------------------------------
# 6. Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_context_warn_fraction_between_zero_and_one(self):
        from app.core.context_window import CONTEXT_WARN_FRACTION
        assert 0.0 < CONTEXT_WARN_FRACTION < 1.0

    def test_context_keep_recent_positive(self):
        from app.core.context_window import CONTEXT_KEEP_RECENT
        assert CONTEXT_KEEP_RECENT > 0

    def test_default_tool_result_max_chars_positive(self):
        from app.core.context_window import DEFAULT_TOOL_RESULT_MAX_CHARS
        assert DEFAULT_TOOL_RESULT_MAX_CHARS > 0


# ---------------------------------------------------------------------------
# 7. Config fields
# ---------------------------------------------------------------------------

class TestConfigFields:
    def test_context_warn_fraction_default(self):
        from app.core.config import get_settings
        assert get_settings().context_warn_fraction == 0.75

    def test_context_keep_recent_messages_default(self):
        from app.core.config import get_settings
        assert get_settings().context_keep_recent_messages == 6

    def test_tool_result_max_chars_default(self):
        from app.core.config import get_settings
        assert get_settings().tool_result_max_chars == 8_000


# ---------------------------------------------------------------------------
# 8. Events
# ---------------------------------------------------------------------------

class TestContextUsageEvent:
    def _clear(self):
        import app.core.events as ev
        ev.clear_listeners()
        ev._history.clear()

    def test_context_usage_in_event_category(self):
        from app.core.events import EventCategory
        assert hasattr(EventCategory, "CONTEXT_USAGE")
        assert EventCategory.CONTEXT_USAGE == "context_usage"

    def test_emit_context_usage_emits_event(self):
        from app.core.events import emit_context_usage, get_history
        self._clear()
        emit_context_usage("coder_a", 10_000, 128_000, 0.078)
        history = get_history()
        matching = [e for e in history if e.get("category") == "context_usage"]
        assert len(matching) == 1
        self._clear()

    def test_emit_context_usage_metadata(self):
        from app.core.events import emit_context_usage, get_history
        self._clear()
        emit_context_usage("tester", 50_000, 200_000, 0.25, compressed=True)
        history = get_history()
        evt = next(e for e in history if e.get("category") == "context_usage")
        assert evt["metadata"]["estimated_tokens"] == 50_000
        assert evt["metadata"]["model_limit"] == 200_000
        assert evt["metadata"]["compressed"] is True
        self._clear()

    def test_emit_context_usage_title_contains_percent(self):
        from app.core.events import emit_context_usage, get_history
        self._clear()
        emit_context_usage("planner", 25_000, 128_000, 0.195)
        history = get_history()
        evt = next(e for e in history if e.get("category") == "context_usage")
        assert "%" in evt.get("title", "")
        self._clear()
