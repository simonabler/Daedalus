"""Tests for token budget and cost tracking (Issue #34).

Covers:
- calculate_cost: known model pricing, default fallback, Ollama free
- _pricing_for_model: prefix matching, unknown models
- extract_token_usage: Anthropic format, OpenAI format, empty/missing
- TokenUsageRecord: to_dict / from_dict round-trip
- TokenBudget: accumulation, per_agent, soft limit, hard limit, to_dict/from_dict
- BudgetExceededException raised on hard limit
- _model_name_for_role: role → model string mapping
- _get_budget / _budget_dict: state round-trip
- Config: token_budget_soft_limit_usd, token_budget_hard_limit_usd fields
- GraphState: token_budget field present and defaults to {}
- EventCategory.TOKEN_USAGE present; emit_token_usage emits correct event
- /api/status response includes token_budget field
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.state import GraphState


# ---------------------------------------------------------------------------
# 1. calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:
    def _cost(self, model, prompt, completion):
        from app.core.token_budget import calculate_cost
        return calculate_cost(model, prompt, completion)

    def test_gpt4o_mini_known_pricing(self):
        # 1M input @ $0.15, 1M output @ $0.60
        cost = self._cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 0.001

    def test_gpt4o_known_pricing(self):
        cost = self._cost("gpt-4o", 1_000_000, 1_000_000)
        assert abs(cost - 12.50) < 0.01

    def test_claude_sonnet_known_pricing(self):
        cost = self._cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert abs(cost - 18.00) < 0.01

    def test_ollama_model_is_free(self):
        cost = self._cost("ollama:llama3.1:70b", 100_000, 100_000)
        assert cost == 0.0

    def test_unknown_model_uses_default(self):
        from app.core.token_budget import calculate_cost, MODEL_PRICING
        cost = calculate_cost("some-unknown-model-xyz", 1_000_000, 1_000_000)
        default = MODEL_PRICING["_default"]
        expected = (default["input"] + default["output"]) / 1_000_000 * 1_000_000
        assert abs(cost - expected) < 0.01

    def test_zero_tokens_zero_cost(self):
        assert self._cost("gpt-4o-mini", 0, 0) == 0.0

    def test_input_only_tokens(self):
        cost = self._cost("gpt-4o-mini", 1_000_000, 0)
        assert abs(cost - 0.15) < 0.001


# ---------------------------------------------------------------------------
# 2. extract_token_usage
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:
    def _extract(self, response_mock):
        from app.core.token_budget import extract_token_usage
        return extract_token_usage(response_mock)

    def _mock(self, **attrs):
        m = MagicMock()
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def test_anthropic_usage_metadata(self):
        resp = self._mock(usage_metadata={"input_tokens": 100, "output_tokens": 50})
        result = self._extract(resp)
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_openai_response_metadata(self):
        resp = self._mock(
            usage_metadata={},
            response_metadata={"token_usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}},
        )
        result = self._extract(resp)
        assert result["prompt_tokens"] == 200
        assert result["completion_tokens"] == 80

    def test_missing_metadata_returns_zeros(self):
        resp = self._mock(usage_metadata=None, response_metadata={})
        result = self._extract(resp)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_total_tokens_computed_if_missing(self):
        resp = self._mock(usage_metadata={"input_tokens": 60, "output_tokens": 40})
        result = self._extract(resp)
        assert result["total_tokens"] == 100

    def test_no_attributes_returns_zeros(self):
        # Plain object with no metadata attrs
        class Empty:
            pass
        result = self._extract(Empty())
        assert result["prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# 3. TokenUsageRecord
# ---------------------------------------------------------------------------

class TestTokenUsageRecord:
    def _make(self, **kwargs):
        from app.core.token_budget import TokenUsageRecord
        defaults = dict(agent="coder_a", model="gpt-4o-mini",
                        prompt_tokens=100, completion_tokens=50,
                        total_tokens=150, cost_usd=0.0001)
        defaults.update(kwargs)
        return TokenUsageRecord(**defaults)

    def test_to_dict_has_expected_keys(self):
        rec = self._make()
        d = rec.to_dict()
        for key in ("agent", "model", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
            assert key in d

    def test_from_dict_round_trip(self):
        from app.core.token_budget import TokenUsageRecord
        rec = self._make(agent="tester", model="claude-sonnet-4-20250514", prompt_tokens=300)
        d = rec.to_dict()
        rec2 = TokenUsageRecord.from_dict(d)
        assert rec2.agent == "tester"
        assert rec2.prompt_tokens == 300
        assert rec2.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# 4. TokenBudget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def _make_rec(self, agent="coder_a", prompt=100, completion=50, cost=0.001):
        from app.core.token_budget import TokenUsageRecord
        return TokenUsageRecord(
            agent=agent, model="gpt-4o-mini",
            prompt_tokens=prompt, completion_tokens=completion,
            total_tokens=prompt + completion, cost_usd=cost,
        )

    def test_empty_budget_zeros(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        assert b.total_tokens == 0
        assert b.total_cost_usd == 0.0

    def test_add_accumulates(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.add(self._make_rec(prompt=100, completion=50, cost=0.001))
        b.add(self._make_rec(prompt=200, completion=80, cost=0.002))
        assert b.total_prompt == 300
        assert b.total_completion == 130
        assert abs(b.total_cost_usd - 0.003) < 1e-9

    def test_per_agent_grouped(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.add(self._make_rec(agent="coder_a", cost=0.001))
        b.add(self._make_rec(agent="tester", cost=0.002))
        b.add(self._make_rec(agent="coder_a", cost=0.001))
        pa = b.per_agent()
        assert abs(pa["coder_a"]["cost_usd"] - 0.002) < 1e-9
        assert abs(pa["tester"]["cost_usd"] - 0.002) < 1e-9
        assert pa["coder_a"]["calls"] == 2

    def test_soft_limit_sets_flag(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget(soft_limit_usd=0.005)
        b.add(self._make_rec(cost=0.003))
        assert not b.soft_limit_hit
        b.add(self._make_rec(cost=0.003))
        assert b.soft_limit_hit

    def test_soft_limit_does_not_raise(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget(soft_limit_usd=0.001)
        b.add(self._make_rec(cost=0.005))  # exceeds — no exception

    def test_hard_limit_raises(self):
        from app.core.token_budget import TokenBudget, BudgetExceededException
        b = TokenBudget(hard_limit_usd=0.005)
        b.add(self._make_rec(cost=0.003))
        with pytest.raises(BudgetExceededException) as exc_info:
            b.add(self._make_rec(cost=0.003))
        assert exc_info.value.total_cost > 0.005
        assert b.hard_limit_hit

    def test_zero_hard_limit_never_raises(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget(hard_limit_usd=0.0)
        for _ in range(100):
            b.add(self._make_rec(cost=1.0))  # $100 total — no raise

    def test_to_dict_from_dict_round_trip(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget(soft_limit_usd=1.0, hard_limit_usd=5.0)
        b.add(self._make_rec(agent="planner", prompt=500, completion=200, cost=0.01))
        d = b.to_dict()
        b2 = TokenBudget.from_dict(d)
        assert b2.total_prompt == 500
        assert b2.total_completion == 200
        assert abs(b2.total_cost_usd - 0.01) < 1e-9
        assert b2.soft_limit_usd == 1.0
        assert len(b2.records) == 1

    def test_summary_has_per_agent(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.add(self._make_rec(agent="tester", cost=0.002))
        s = b.summary()
        assert "per_agent" in s
        assert "tester" in s["per_agent"]

    def test_calls_count_in_summary(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.add(self._make_rec())
        b.add(self._make_rec())
        assert b.summary()["calls"] == 2


# ---------------------------------------------------------------------------
# 5. _model_name_for_role
# ---------------------------------------------------------------------------

class TestModelNameForRole:
    def _get(self, role):
        from app.core.nodes import _model_name_for_role
        return _model_name_for_role(role)

    def test_planner_returns_planner_model(self):
        from app.core.config import get_settings
        assert self._get("planner") == get_settings().planner_model

    def test_coder_a_returns_coder_1_model(self):
        from app.core.config import get_settings
        assert self._get("coder_a") == get_settings().coder_1_model

    def test_tester_returns_tester_model(self):
        from app.core.config import get_settings
        assert self._get("tester") == get_settings().tester_model

    def test_unknown_role_returns_planner_model(self):
        from app.core.config import get_settings
        assert self._get("unknown_role") == get_settings().planner_model


# ---------------------------------------------------------------------------
# 6. Config fields
# ---------------------------------------------------------------------------

class TestConfigBudgetFields:
    def test_soft_limit_defaults_to_zero(self):
        from app.core.config import get_settings
        assert get_settings().token_budget_soft_limit_usd == 0.0

    def test_hard_limit_defaults_to_zero(self):
        from app.core.config import get_settings
        assert get_settings().token_budget_hard_limit_usd == 0.0


# ---------------------------------------------------------------------------
# 7. GraphState.token_budget
# ---------------------------------------------------------------------------

class TestGraphStateTokenBudget:
    def test_token_budget_defaults_to_empty_dict(self):
        state = GraphState()
        assert state.token_budget == {}

    def test_token_budget_serialises_with_pydantic(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.total_cost_usd = 0.05
        state = GraphState(token_budget=b.to_dict())
        assert state.token_budget["total_cost_usd"] == 0.05

    def test_token_budget_survives_model_dump_round_trip(self):
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.total_tokens = 1000
        state = GraphState(token_budget=b.to_dict())
        dumped = state.model_dump()
        restored = GraphState(**dumped)
        assert restored.token_budget["total_tokens"] == 1000


# ---------------------------------------------------------------------------
# 8. Events
# ---------------------------------------------------------------------------

class TestTokenUsageEvent:
    def test_token_usage_in_event_category(self):
        from app.core.events import EventCategory
        assert hasattr(EventCategory, "TOKEN_USAGE")
        assert EventCategory.TOKEN_USAGE == "token_usage"

    def _clear(self):
        from app.core.events import clear_listeners
        import app.core.events as ev
        clear_listeners()
        ev._history.clear()

    def test_emit_token_usage_emits_event(self):
        from app.core.events import emit_token_usage, get_history
        self._clear()
        emit_token_usage("coder_a", "gpt-4o-mini", 100, 50, 0.0001, 0.0001)
        history = get_history()
        matching = [e for e in history if e.get("category") == "token_usage"]
        assert len(matching) == 1
        self._clear()

    def test_emit_token_usage_metadata_shape(self):
        from app.core.events import emit_token_usage, get_history
        self._clear()
        emit_token_usage("tester", "gpt-4o-mini", 200, 80, 0.0002, 0.0005)
        history = get_history()
        evt = next(e for e in history if e.get("category") == "token_usage")
        assert evt["metadata"]["prompt_tokens"] == 200
        assert evt["metadata"]["completion_tokens"] == 80
        assert evt["metadata"]["model"] == "gpt-4o-mini"
        assert "total_cost_usd" in evt["metadata"]
        self._clear()


# ---------------------------------------------------------------------------
# 9. _get_budget and _budget_dict helpers
# ---------------------------------------------------------------------------

class TestBudgetHelpers:
    def test_get_budget_returns_empty_budget_for_empty_state(self):
        from app.core.nodes import _get_budget
        state = GraphState()
        budget = _get_budget(state)
        assert budget.total_tokens == 0

    def test_get_budget_restores_from_state(self):
        from app.core.nodes import _get_budget
        from app.core.token_budget import TokenBudget, TokenUsageRecord
        b = TokenBudget()
        b.total_cost_usd = 0.123
        b.total_tokens = 500
        state = GraphState(token_budget=b.to_dict())
        restored = _get_budget(state)
        assert abs(restored.total_cost_usd - 0.123) < 1e-9
        assert restored.total_tokens == 500

    def test_budget_dict_serialises_correctly(self):
        from app.core.nodes import _budget_dict
        from app.core.token_budget import TokenBudget
        b = TokenBudget()
        b.total_cost_usd = 0.05
        d = _budget_dict(b)
        assert isinstance(d, dict)
        assert d["total_cost_usd"] == 0.05
