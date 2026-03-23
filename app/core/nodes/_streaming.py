"""Streaming helper for LLM token-by-token delivery to the UI."""
from __future__ import annotations

from app.core.events import emit_agent_token
from app.core.logging import get_logger

logger = get_logger("core.nodes._streaming")

# Roles that emit streaming tokens to the UI (coders, planner, reviewers).
# Tester and router are short calls where streaming adds no UX value.
_STREAMING_ROLES: frozenset[str] = frozenset({
    "coder_a", "coder_b", "reviewer_a", "reviewer_b", "planner", "documenter"
})

# Minimum token batch size before emitting a streaming event.
# Batching avoids flooding the event bus on slow models.
_TOKEN_BATCH_MIN = 8


def _stream_llm_round(
    role: str,
    llm_with_tools: object,
    all_messages: list,
) -> "AIMessage":
    """Stream one LLM round, emitting token events, and return a full AIMessage.

    Falls back to .invoke() if the model does not support streaming or if
    streaming raises an exception.
    """
    from langchain_core.messages import AIMessage, AIMessageChunk

    accumulated_text = ""
    accumulated_chunk: AIMessageChunk | None = None
    batch: list[str] = []

    def _flush_batch() -> None:
        nonlocal batch
        if batch:
            emit_agent_token(role, "".join(batch))
            batch = []

    try:
        for chunk in llm_with_tools.stream(all_messages):
            if not isinstance(chunk, AIMessageChunk):
                continue

            # Accumulate for final reconstruction
            accumulated_chunk = chunk if accumulated_chunk is None else accumulated_chunk + chunk

            # Extract text content
            text = ""
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                    elif isinstance(part, str):
                        text += part

            if text:
                accumulated_text += text
                batch.append(text)
                if len(batch) >= _TOKEN_BATCH_MIN:
                    _flush_batch()

        _flush_batch()

        # Reconstruct a proper AIMessage from the accumulated chunk
        if accumulated_chunk is not None:
            return AIMessage(
                content=accumulated_chunk.content,
                tool_calls=list(accumulated_chunk.tool_calls) if accumulated_chunk.tool_calls else [],
                id=getattr(accumulated_chunk, "id", None),
            )
        return AIMessage(content=accumulated_text)

    except NotImplementedError:
        # Model doesn't support streaming — fall back silently
        logger.debug("streaming_fallback | %s does not support .stream()", role)
        return llm_with_tools.invoke(all_messages)
    except Exception as exc:
        logger.warning("streaming_error | %s — %s — falling back to .invoke()", role, exc)
        return llm_with_tools.invoke(all_messages)

