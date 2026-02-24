"""Token budget and cost tracking for Daedalus workflow runs.

Every LLM call records prompt/completion tokens and estimated cost.
Configurable soft (warning) and hard (stop) USD limits are enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Pricing table — USD per 1M tokens (input / output separately)
# Update when providers change pricing. Unknown models fall back to _default.
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-20250514":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4-20250514":    {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-haiku-3-5":          {"input": 0.80,  "output": 4.00},
    "claude-3-5-sonnet":         {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku":          {"input": 0.80,  "output": 4.00},
    "claude-3-opus":             {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o":                    {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":               {"input": 10.00, "output": 30.00},
    "gpt-4":                     {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":             {"input": 0.50,  "output": 1.50},
    "o1":                        {"input": 15.00, "output": 60.00},
    "o1-mini":                   {"input": 3.00,  "output": 12.00},
    # Fallback for unknown / local models
    "_default":                  {"input": 3.00,  "output": 15.00},
    "_ollama":                   {"input": 0.00,  "output": 0.00},
}


def _pricing_for_model(model: str) -> dict[str, float]:
    """Return the pricing dict for a model name.

    Matches by prefix so that model variants (e.g. 'claude-sonnet-4-20250514')
    work even if only the base name appears in the table.
    """
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Prefix matching
    for key in MODEL_PRICING:
        if key.startswith("_"):
            continue
        if model.startswith(key) or key in model:
            return MODEL_PRICING[key]
    # Ollama models are free (local)
    if model.startswith("ollama:") or "/" in model:
        return MODEL_PRICING["_ollama"]
    return MODEL_PRICING["_default"]


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated cost in USD for a single LLM call."""
    pricing = _pricing_for_model(model)
    return (
        prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class TokenUsageRecord:
    """Single LLM call token/cost record."""
    agent:             str
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    node:              str = ""
    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "agent":             self.agent,
            "model":             self.model,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "node":              self.node,
            "timestamp":         self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenUsageRecord":
        return cls(
            agent=d.get("agent", ""),
            model=d.get("model", ""),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
            cost_usd=d.get("cost_usd", 0.0),
            node=d.get("node", ""),
            timestamp=d.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class BudgetExceededException(Exception):
    """Raised when a hard budget limit is exceeded."""
    def __init__(self, total_cost: float, limit: float) -> None:
        self.total_cost = total_cost
        self.limit = limit
        super().__init__(
            f"Hard budget limit exceeded: ${total_cost:.4f} >= ${limit:.4f}"
        )


@dataclass
class TokenBudget:
    """Thread-safe accumulator for token usage across one workflow run."""

    records:           list[TokenUsageRecord] = field(default_factory=list)
    total_prompt:      int   = 0
    total_completion:  int   = 0
    total_tokens:      int   = 0
    total_cost_usd:    float = 0.0
    soft_limit_usd:    float = 0.0   # 0 = disabled
    hard_limit_usd:    float = 0.0   # 0 = disabled
    soft_limit_hit:    bool  = False
    hard_limit_hit:    bool  = False

    def add(self, record: TokenUsageRecord) -> None:
        """Accumulate a new usage record. Checks limits after accumulation."""
        self.records.append(record)
        self.total_prompt      += record.prompt_tokens
        self.total_completion  += record.completion_tokens
        self.total_tokens      += record.total_tokens
        self.total_cost_usd    += record.cost_usd

        if self.hard_limit_usd > 0 and self.total_cost_usd >= self.hard_limit_usd:
            self.hard_limit_hit = True
            raise BudgetExceededException(self.total_cost_usd, self.hard_limit_usd)

        if (
            self.soft_limit_usd > 0
            and not self.soft_limit_hit
            and self.total_cost_usd >= self.soft_limit_usd
        ):
            self.soft_limit_hit = True

    def per_agent(self) -> dict[str, dict]:
        """Return cost/token breakdown grouped by agent role."""
        result: dict[str, dict] = {}
        for rec in self.records:
            entry = result.setdefault(rec.agent, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            })
            entry["prompt_tokens"]     += rec.prompt_tokens
            entry["completion_tokens"] += rec.completion_tokens
            entry["total_tokens"]      += rec.total_tokens
            entry["cost_usd"]          += rec.cost_usd
            entry["calls"]             += 1
        # Round costs for readability
        for entry in result.values():
            entry["cost_usd"] = round(entry["cost_usd"], 6)
        return result

    def summary(self) -> dict:
        """Return a serialisable summary dict (stored in GraphState)."""
        return {
            "total_prompt":     self.total_prompt,
            "total_completion": self.total_completion,
            "total_tokens":     self.total_tokens,
            "total_cost_usd":   round(self.total_cost_usd, 6),
            "soft_limit_usd":   self.soft_limit_usd,
            "hard_limit_usd":   self.hard_limit_usd,
            "soft_limit_hit":   self.soft_limit_hit,
            "hard_limit_hit":   self.hard_limit_hit,
            "calls":            len(self.records),
            "per_agent":        self.per_agent(),
            "records":          [r.to_dict() for r in self.records],
        }

    def to_dict(self) -> dict:
        """Alias for summary() — used when serialising to GraphState."""
        return self.summary()

    @classmethod
    def from_dict(cls, d: dict) -> "TokenBudget":
        """Reconstruct from a serialised summary dict."""
        budget = cls(
            total_prompt=d.get("total_prompt", 0),
            total_completion=d.get("total_completion", 0),
            total_tokens=d.get("total_tokens", 0),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            soft_limit_usd=d.get("soft_limit_usd", 0.0),
            hard_limit_usd=d.get("hard_limit_usd", 0.0),
            soft_limit_hit=d.get("soft_limit_hit", False),
            hard_limit_hit=d.get("hard_limit_hit", False),
        )
        for rec_dict in d.get("records", []):
            budget.records.append(TokenUsageRecord.from_dict(rec_dict))
        return budget


def extract_token_usage(response: object) -> dict[str, int]:
    """Extract prompt/completion/total token counts from a LangChain response.

    Handles all three providers:
    - Anthropic: response.usage_metadata with input_tokens / output_tokens
    - OpenAI:    response.response_metadata.token_usage OR usage_metadata
    - Ollama:    same as OpenAI but may return zeros

    Returns dict with keys: prompt_tokens, completion_tokens, total_tokens.
    All values default to 0 if metadata is unavailable.
    """
    # --- Primary: usage_metadata (Anthropic + modern OpenAI LangChain) ----
    meta = getattr(response, "usage_metadata", None)
    if meta and isinstance(meta, dict):
        prompt     = meta.get("input_tokens", 0) or meta.get("prompt_tokens", 0)
        completion = meta.get("output_tokens", 0) or meta.get("completion_tokens", 0)
        total      = meta.get("total_tokens", 0) or (prompt + completion)
        if prompt or completion:
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    # --- Fallback: response_metadata (OpenAI via LangChain) ---------------
    rm = getattr(response, "response_metadata", {}) or {}
    usage = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(usage, dict):
        prompt     = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total      = usage.get("total_tokens", 0) or (prompt + completion)
        if prompt or completion:
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
