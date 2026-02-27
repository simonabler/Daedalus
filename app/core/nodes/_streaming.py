"""Streaming and LLM invocation helpers for workflow nodes."""

from __future__ import annotations

from ._helpers import *


_STREAMING_ROLES: frozenset[str] = frozenset({
    "planner", "coder_a", "coder_b", "reviewer_a", "reviewer_b",
    "documenter",
})
_TOKEN_BATCH_MIN = 8

def _model_name_for_role(role: str) -> str:
    """Return the configured model name for a given agent role."""
    settings = get_settings()
    mapping = {
        "planner":    settings.planner_model,
        "coder_a":    settings.coder_1_model,
        "coder_b":    settings.coder_2_model,
        "reviewer_a": settings.coder_1_model,
        "reviewer_b": settings.coder_2_model,
        "documenter": settings.documenter_model,
        "tester":     settings.tester_model,
    }
    return mapping.get(role, settings.planner_model)
ROUTER_INTENTS = {"code", "status", "research", "resume"}


# -- Helper: parse ask_human signal from coder response -------------------

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

def _invoke_agent(role: str, messages: list, tools: list | None = None,
                  inject_memory: bool = False,
                  budget: TokenBudget | None = None,
                  node: str = "") -> str:
    """Invoke an LLM agent, handle tool calls, emit events.

    Streaming is enabled automatically for roles in _STREAMING_ROLES.
    All other roles use .invoke() (blocking). Falls back to .invoke()
    gracefully if the model does not support streaming.

    If inject_memory=True, the shared long-term memory is prepended to the
    system prompt so the agent can use established conventions.

    If budget is provided, token usage is tracked after each LLM response
    and a BudgetExceededException is raised when the hard limit is hit.

    Context window management (always active):
    - Tool results are truncated to settings.tool_result_max_chars before
      being appended to all_messages (preventive).
    - After each LLM response, context usage is estimated. If it exceeds
      settings.context_warn_fraction of the model limit, old turns are
      compressed via an LLM summary call (reactive).
    """
    llm = get_llm(role)
    model_name = _model_name_for_role(role)
    system_prompt = load_system_prompt(role)
    settings = get_settings()

    # Inject shared memory into system prompt for coders/reviewers
    if inject_memory:
        memory_ctx = load_all_memory()
        if memory_ctx:
            system_prompt = system_prompt + "\n\n" + memory_ctx

    all_messages = [SystemMessage(content=system_prompt)] + messages

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    use_streaming = role in _STREAMING_ROLES

    prompt_summary = messages[-1].content[:300] if messages else ""
    emit_agent_thinking(role, prompt_summary)

    # -- (a) Warn if initial context is already large ----------------------
    initial_fraction = context_usage_fraction(all_messages, model_name)
    if initial_fraction > 0.5:
        emit_status(
            "system",
            f"ℹ️ Initial context at {initial_fraction:.0%} of {model_name} limit",
        )

    max_tool_rounds = 15
    for _round_num in range(max_tool_rounds):
        if use_streaming:
            response = _stream_llm_round(role, llm_with_tools, all_messages)
        else:
            response = llm_with_tools.invoke(all_messages)

        # -- Token tracking ------------------------------------------------
        if budget is not None:
            usage = extract_token_usage(response)
            cost = calculate_cost(model_name, usage["prompt_tokens"], usage["completion_tokens"])
            record = TokenUsageRecord(
                agent=role,
                model=model_name,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                cost_usd=cost,
                node=node,
            )
            try:
                budget.add(record)
                emit_token_usage(
                    agent=role,
                    model=model_name,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    cost_usd=cost,
                    total_cost_usd=budget.total_cost_usd,
                )
                if budget.soft_limit_hit and usage["prompt_tokens"] > 0:
                    # Emit soft-limit warning exactly once (flag stays True)
                    emit_status(
                        "system",
                        f"⚠️ Token budget soft limit reached: "
                        f"${budget.total_cost_usd:.4f} >= ${budget.soft_limit_usd:.2f}. "
                        "Workflow continues.",
                    )
            except BudgetExceededException as exc:
                emit_error(
                    "system",
                    f"❌ Hard budget limit exceeded: ${exc.total_cost:.4f} >= ${exc.limit:.2f}. "
                    "Stopping workflow.",
                )
                raise
        # ------------------------------------------------------------------

        all_messages.append(response)

        # -- (c) Context check + compression after each LLM turn -----------
        ctx_limit    = context_limit_for_model(model_name)
        ctx_tokens   = estimate_messages_tokens(all_messages)
        ctx_fraction = ctx_tokens / ctx_limit
        emit_context_usage(role, ctx_tokens, ctx_limit, ctx_fraction)

        warn_fraction = settings.context_warn_fraction
        if warn_fraction > 0 and ctx_fraction >= warn_fraction:
            emit_status(
                "system",
                f"⚠️ Context at {ctx_fraction:.0%} ({ctx_tokens:,} / {ctx_limit:,} tok)"
                " — compressing old turns",
            )
            all_messages = compress_messages(all_messages, model_name, llm)
            new_tokens   = estimate_messages_tokens(all_messages)
            new_fraction = new_tokens / ctx_limit
            emit_context_usage(role, new_tokens, ctx_limit, new_fraction, compressed=True)
            emit_status(
                "system",
                f"✅ Context compressed: {ctx_tokens:,} → {new_tokens:,} tok"
                f" ({new_fraction:.0%})",
            )
        # ------------------------------------------------------------------

        if not response.tool_calls:
            result = response.content if isinstance(response.content, str) else str(response.content)
            emit_agent_result(role, result)
            return result

        # After tool calls the next LLM turn may stream again — reset flag
        # so partial streaming blocks are visually separated.
        tool_map = {t.name: t for t in (tools or [])}
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in tc["args"].items())
            emit_tool_call(role, tc["name"], args_str)

            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                    emit_error(role, f"Tool {tc['name']} failed: {e}")
            else:
                result = f"Unknown tool: {tc['name']}"

            emit_tool_result(role, tc["name"], str(result))
            logger.info("tool_call  | %s(%s) -> %d chars", tc["name"], list(tc["args"].keys()), len(str(result)))

            # -- (b) Truncate tool result before entering all_messages -----
            safe_result = truncate_tool_result(
                str(result), max_chars=settings.tool_result_max_chars
            )
            all_messages.append(ToolMessage(content=safe_result, tool_call_id=tc["id"]))

    emit_error(role, "Exceeded maximum tool call rounds (15)")
    return "ERROR: Exceeded maximum tool call rounds."


# -- Tool sets -------------------------------------------------------------

PLANNER_TOOLS = [read_file, write_file, list_directory, search_in_repo, git_status, run_terminal]

CODER_TOOLS = [
    read_file, write_file, list_directory,
    search_in_repo,
    run_terminal, git_status, git_command,
    run_tests, run_linter,
]

REVIEWER_TOOLS = [
    read_file,
    list_directory,
    search_in_repo,
    run_terminal,
    git_status,
    git_command,
    run_tests,
    run_linter,
]

TESTER_TOOLS = [read_file, list_directory, run_terminal, run_tests, run_linter, git_status]
DOCUMENTER_TOOLS = [read_file, write_file, list_directory, run_terminal, git_status, git_command]


# -- Helper: coder pair assignment -----------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
