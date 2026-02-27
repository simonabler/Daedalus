"""Coder node and coder-question parsing helper."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *
from ._prompt_enrichment import *

def _parse_coder_question(response: str) -> dict | None:
    """Return the parsed ask_human payload if the coder asked a question, else None.

    The coder signals a question by returning a JSON object whose top-level key
    ``"action"`` equals ``"ask_human"``.  The response must be *only* that JSON
    (possibly wrapped in a single markdown code fence) — any preamble text
    disqualifies it to avoid false positives.

    Returns a dict with keys: question, context, options (may be empty list).
    """
    text = response.strip()

    # Strip optional ```json / ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence line and closing fence line
        inner = "\n".join(
            line for line in lines[1:]
            if not line.strip().startswith("```")
        )
        text = inner.strip()

    # Must look like a JSON object to avoid expensive parsing on normal output
    if not text.startswith("{"):
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("action") != "ask_human":
        return None

    question = str(payload.get("question", "")).strip()
    if not question:
        return None

    return {
        "question": question,
        "context":  str(payload.get("context", "")).strip(),
        "options":  [str(o) for o in payload.get("options", []) if o],
        "urgency":  payload.get("urgency", "blocking") if payload.get("urgency") in ("blocking", "advisory") else "blocking",
        "default_if_skipped": str(payload.get("default_if_skipped", "")).strip(),
    }


# -- Helper: invoke LLM with tools + event emission -----------------------

# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

# Roles that emit streaming tokens to the UI (coders, planner, reviewers).
# Tester and router are short calls where streaming adds no UX value.
_STREAMING_ROLES: frozenset[str] = frozenset({
    "coder_a", "coder_b", "reviewer_a", "reviewer_b", "planner", "documenter"
})

# Minimum token batch size before emitting a streaming event.
# Batching avoids flooding the event bus on slow models.
_TOKEN_BATCH_MIN = 8

def coder_node(state: GraphState) -> dict:
    """Dispatch to active worker (coder or documenter)."""
    item = state.current_item
    if not item:
        emit_error("system", "No current item to work on")
        return {"error_message": "No current item to work on", "phase": WorkflowPhase.STOPPED}

    active = state.active_coder
    reviewer = state.active_reviewer
    item_num = state.current_item_index + 1
    total = len(state.todo_items)

    emit_node_start(active, "Coding", item_id=item.id, item_desc=item.description)
    emit_status(
        active,
        f"[{item_num}/{total}] {_coder_label(active)} implementing: {item.description}",
        iteration=item.iteration_count + 1,
        **_progress_meta(state, "coding"),
    )

    item.status = ItemStatus.IN_PROGRESS
    item.iteration_count += 1

    settings = get_settings()
    if item.iteration_count > settings.max_iterations_per_item:
        msg = f"Item {item.id} exceeded max iterations ({settings.max_iterations_per_item})"
        emit_error(active, msg)
        return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED}

    prompt_parts = [
        f"## Task Assignment — {_coder_label(active)}",
        f"**Item ID**: {item.id}",
        f"**Task Type**: {item.task_type}",
        f"**Description**: {item.description}",
        f"**Your peer reviewer**: {_reviewer_label(reviewer)}",
        f"**Execution platform**: {state.execution_platform or platform.platform()}",
        f"**OS Note**: {_os_note(state.execution_platform or platform.platform())}",
    ]
    if item.acceptance_criteria:
        prompt_parts.append("**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria))
    if item.verification_commands:
        prompt_parts.append(
            "**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands)
        )
    if item.review_notes:
        prompt_parts.append(f"**Rework Notes (from previous review)**:\n{item.review_notes}")
    if state.peer_review_notes and state.peer_review_verdict == "REWORK":
        prompt_parts.append(f"**Peer Review Feedback (REWORK)**:\n{state.peer_review_notes}")
    if item.test_report:
        prompt_parts.append(f"**Test Report (previous)**:\n{item.test_report}")
    if state.repo_facts:
        prompt_parts.append(_format_repo_context_for_prompt(state.repo_facts))
    intelligence_summary = _format_intelligence_summary_for_prompt(state)
    if intelligence_summary:
        prompt_parts.append(intelligence_summary)
    if state.agent_instructions:
        prompt_parts.append(
            "**Repository Documentation (excerpt)**:\n"
            + _truncate_context_text(state.agent_instructions, limit=3000)
        )

    preferred_test_cmd = _extract_test_command(state.repo_facts)

    prompt_parts.append(
        "\nImplement this task. Use tools to read the codebase, make changes, "
        "add tests where needed, and update docs. Keep diffs minimal.\n"
        "Follow the coding style and architecture decisions from shared memory.\n\n"
        "CRITICAL WORKFLOW:\n"
        "1. Search for similar code first via `search_in_repo`.\n"
        "2. Read existing implementations with `read_file` before editing.\n"
        "3. Prefer minimal edits and keep architecture patterns consistent.\n"
        "4. Add or update tests with every behavior change.\n"
        f"5. Verify using preferred command: `{preferred_test_cmd or 'python -m pytest -q'}`."
    )

    # If the human answered a previous question from this coder, inject the
    # answer as the first message so the coder continues with that context.
    messages: list = []
    if state.coder_question_answer and state.coder_question_asked_by == active:
        messages.append(HumanMessage(
            content=(
                "[Human answered your earlier question]\n"
                f"Q: {state.coder_question}\n"
                f"A: {state.coder_question_answer}\n\n"
                "Continue implementing the task using this answer as your guide."
            )
        ))

    messages.append(HumanMessage(content="\n\n".join(prompt_parts)))

    tools = DOCUMENTER_TOOLS if active == "documenter" else CODER_TOOLS
    try:
        result, budget_update = _invoke_with_budget(state, active, messages, tools, inject_memory=True, node="coder")
    except BudgetExceededException:
        return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}

    # Detect ask_human signal — coder wants to pause for human input
    question_payload = _parse_coder_question(result)
    if question_payload:
        settings      = get_settings()
        urgency       = question_payload["urgency"]
        default_      = question_payload["default_if_skipped"]
        cap           = settings.coder_question_max_per_item
        mode          = settings.coder_question_advisory_mode
        already_asked = item.coder_questions_asked

        # Decide whether to skip this question
        cap_reached        = already_asked >= cap
        auto_skip_advisory = (urgency == "advisory" and mode == "auto_proceed")
        skip               = cap_reached or auto_skip_advisory

        # Always increment the per-item counter
        updated_items = list(state.todo_items)
        updated_items[state.current_item_index] = item.model_copy(
            update={"coder_questions_asked": already_asked + 1}
        )

        if skip:
            reason     = "cap reached" if cap_reached else "advisory mode = auto_proceed"
            assumption = default_ or "Proceeding with the most reasonable default."
            emit_status(
                active,
                f"⚡ Skipping question ({reason}) — assumption: {assumption}",
                **_progress_meta(state, "coding"),
            )
            # Re-invoke the coder with the assumption injected as a synthetic answer
            from langchain_core.messages import AIMessage as _AIMessage
            messages_with_assumption = messages + [
                _AIMessage(content=result),
                HumanMessage(content=f"[AUTO-SKIPPED] {assumption}"),
            ]
            try:
                result, budget_update = _invoke_with_budget(
                    state, active, messages_with_assumption, tools,
                    inject_memory=True, node="coder",
                )
            except BudgetExceededException:
                return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}
            # Fall through to normal result handling below with updated result
        else:
            emit_coder_question(
                asked_by=active,
                question=question_payload["question"],
                context=question_payload["context"],
                options=question_payload["options"],
                item_id=item.id,
                urgency=urgency,
                default_if_skipped=default_,
            )
            emit_node_end(active, "Coding", "Paused — coder is asking the human a question")
            return {
                "needs_coder_answer": True,
                "coder_question": question_payload["question"],
                "coder_question_context": question_payload["context"],
                "coder_question_options": question_payload["options"],
                "coder_question_asked_by": active,
                "coder_question_urgency": urgency,
                "coder_question_default": default_,
                "coder_question_answer": "",
                "todo_items": updated_items,
                "phase": WorkflowPhase.WAITING_FOR_ANSWER,
                "stop_reason": "waiting_for_coder_answer",
            }
    else:
        updated_items = state.todo_items

    emit_node_end(active, "Coding", f"Implementation complete — handing to {_reviewer_label(reviewer)} for peer review")
    with suppress(Exception):
        _write_todo_file(state.todo_items, state.user_request)

    updates = {
        "last_coder_result": result,
        "phase": WorkflowPhase.PEER_REVIEWING,
        "total_iterations": state.total_iterations + 1,
        # Propagate updated todo_items (coder_questions_asked counter may have changed)
        "todo_items": updated_items,
        # Clear any lingering question state from previous items
        "needs_coder_answer": False,
        "coder_question": "",
        "coder_question_context": "",
        "coder_question_options": [],
        "coder_question_asked_by": "",
        "coder_question_answer": "",
        "coder_question_urgency": "blocking",
        "coder_question_default": "",
        **budget_update,
    }
    _save_checkpoint_snapshot(state, updates, "code_complete")
    return updates


# ---------------------------------------------------------------------------
# NODE: peer_review  (reads shared memory)
# ---------------------------------------------------------------------------
