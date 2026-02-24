"""Context window management for Daedalus LLM calls.

Provides token estimation, per-model context limits, tool-result truncation,
and message-history compression to keep `all_messages` within safe bounds
during `_invoke_agent()` tool-call loops.

Design notes:
- No tiktoken dependency — uses a conservative chars/token heuristic.
- Compression pattern mirrors `_compress_memory_file` in nodes.py / memory.py:
  the planner LLM summarises old turns into a compact HumanMessage block.
- `truncate_tool_result` extends the existing `_truncate_context_text` pattern
  to dynamic tool results (same idea, different marker style).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Conservative ratio: code is ~3.5 chars/token, prose ~4.
# Using 3.0 to err on the side of over-counting (safer).
CHARS_PER_TOKEN: float = 3.0


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using a chars/token heuristic."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of LangChain messages.

    Each message gets +4 tokens of overhead (role prefix, delimiters).
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None) or str(msg)
        # Multi-modal content is a list of dicts
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        total += estimate_tokens(str(content)) + 4
    return total


# ---------------------------------------------------------------------------
# Model context limits
# ---------------------------------------------------------------------------

# Input context window sizes in tokens.
# Prefix-matched — variants like 'claude-sonnet-4-20250514-v2' still resolve.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic
    "claude-sonnet-4-20250514":  200_000,
    "claude-opus-4-20250514":    200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet":         200_000,
    "claude-3-5-haiku":          200_000,
    "claude-3-opus":             200_000,
    "claude-3-sonnet":           200_000,
    "claude-3-haiku":            200_000,
    # OpenAI
    "gpt-4o":                    128_000,
    "gpt-4o-mini":               128_000,
    "gpt-4-turbo":               128_000,
    "gpt-4":                       8_192,
    "gpt-3.5-turbo":              16_385,
    "o1":                        200_000,
    "o1-mini":                   128_000,
    # Fallbacks
    "_default":                   32_000,   # conservative for unknown models
    "_ollama":                    32_000,   # Ollama varies; safe default
}


def context_limit_for_model(model: str) -> int:
    """Return the known context-window limit for a model name.

    Uses prefix matching so that variant names resolve correctly.
    Unknown models get a conservative 32k default.
    """
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    # Prefix matching
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key.startswith("_"):
            continue
        if model.startswith(key) or key in model:
            return limit
    # Ollama models
    if model.startswith("ollama:") or "/" in model:
        return MODEL_CONTEXT_LIMITS["_ollama"]
    return MODEL_CONTEXT_LIMITS["_default"]


def context_usage_fraction(messages: list, model: str) -> float:
    """Return the fraction of the model's context window used (0.0–1.0+).

    Values > 1.0 indicate the context is already over the limit.
    """
    estimated = estimate_messages_tokens(messages)
    limit = context_limit_for_model(model)
    return estimated / limit


# ---------------------------------------------------------------------------
# Tool-result truncation
# ---------------------------------------------------------------------------

# Default max chars for a single tool result before truncation.
# Mirrors the existing max_output_chars setting but applied at the
# message-assembly level as a last line of defence.
DEFAULT_TOOL_RESULT_MAX_CHARS: int = 8_000


def truncate_tool_result(
    result: str,
    max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    """Truncate a tool result that exceeds max_chars.

    Extends the same pattern as `_truncate_context_text` in nodes.py but
    uses a single-sided cut (keep the beginning) since tool results are
    typically most relevant at the start (file headers, match context, etc.).

    Appends a clear marker so the agent knows content was dropped.
    """
    if len(result) <= max_chars:
        return result
    kept = result[:max_chars]
    dropped = len(result) - max_chars
    return (
        kept
        + f"\n\n[... {dropped:,} chars truncated"
        " — use read_file with line ranges for full content ...]"
    )


# ---------------------------------------------------------------------------
# Message-history compression
# ---------------------------------------------------------------------------

# Trigger compression when context reaches this fraction of the model limit.
CONTEXT_WARN_FRACTION: float = 0.75

# Always keep at least this many recent messages intact (never compressed).
CONTEXT_KEEP_RECENT: int = 6

# Max chars of each message included in the compression prompt.
_COMPRESS_MSG_PREVIEW: int = 600


def compress_messages(
    messages: list,
    model: str,
    llm: object,
) -> list:
    """Compress old messages to free context space.

    Strategy:
    1. Keep the SystemMessage(s) at the head — these contain instructions.
    2. Keep the last CONTEXT_KEEP_RECENT messages — these are the live context.
    3. Summarise everything in between into a single compact HumanMessage.

    The summary is produced by the same LLM that is currently running
    (mirrors the `_compress_memory_file` pattern in nodes.py).

    Returns the new, shorter message list. If there is nothing to compress
    (too few messages) the original list is returned unchanged.
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in messages if not isinstance(m, SystemMessage)]

    # Need at least CONTEXT_KEEP_RECENT + 1 non-system messages to compress
    if len(non_system) <= CONTEXT_KEEP_RECENT + 1:
        return messages

    tail   = non_system[-CONTEXT_KEEP_RECENT:]
    middle = non_system[:-CONTEXT_KEEP_RECENT]

    # Build a text representation of the middle turns
    middle_text_parts: list[str] = []
    for msg in middle:
        role_name = type(msg).__name__.replace("Message", "")
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        preview = str(content)[:_COMPRESS_MSG_PREVIEW]
        if len(str(content)) > _COMPRESS_MSG_PREVIEW:
            preview += "…"
        middle_text_parts.append(f"[{role_name}]: {preview}")

    middle_text = "\n\n".join(middle_text_parts)

    summary_prompt = (
        "Summarise the following conversation turns into a compact context block.\n"
        "Preserve: decisions made, files changed, errors encountered, current task state.\n"
        "Be concise — max 400 words. Do not add commentary.\n\n"
        f"{middle_text}"
    )

    try:
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])  # type: ignore[union-attr]
        summary_text = (
            summary_response.content
            if hasattr(summary_response, "content")
            else str(summary_response)
        )
    except Exception:
        # Compression failed — return original rather than crashing the workflow
        return messages

    compressed = HumanMessage(
        content=(
            f"[CONTEXT SUMMARY — {len(middle)} turns compressed]\n\n"
            f"{summary_text}"
        )
    )

    return system_msgs + [compressed] + tail
