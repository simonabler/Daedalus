"""Status node."""

from __future__ import annotations

from ._helpers import *

def status_node(state: GraphState) -> dict:
    """Non-coding status/conversational response branch (no write/git tools).

    Handles questions like "who are you?", "what can you do?", "show me the
    todo list", etc.

    Key rule: if the user asks about the todo list / current tasks, we read
    the ACTUAL data from GraphState and tasks/todo.md and show it verbatim
    rather than letting the LLM fabricate a summary.
    """
    emit_node_start("planner", "Status", item_desc=state.user_request[:100])

    user_q = state.user_request.lower()

    # ── Detect todo/task list requests and answer directly ──────────────
    TODO_KEYWORDS = ("todo", "to-do", "task", "aufgabe", "plan", "items",
                     "list", "liste", "zeig", "show", "print", "display")
    is_todo_query = any(kw in user_q for kw in TODO_KEYWORDS)

    todo_content = ""
    if is_todo_query and state.todo_items:
        # Build a formatted list from live GraphState (most up-to-date)
        lines = []
        for idx, item in enumerate(state.todo_items, start=1):
            done = "\u2705" if item.status == ItemStatus.DONE else "\u2b1c"
            agent = _coder_label(item.assigned_agent or "coder_a")
            lines.append(f"{done} **{idx}. {item.description}**")
            lines.append(f"   Type: {item.task_type} \u00b7 Owner: {agent} \u00b7 Status: {item.status.value}")
            for ac in item.acceptance_criteria:
                lines.append(f"   AC: {ac}")
            lines.append("")
        todo_content = "\n".join(lines)
    elif is_todo_query:
        # Fall back to reading tasks/todo.md from disk
        raw = read_file.invoke({"path": "tasks/todo.md"})
        if not raw.startswith("ERROR"):
            todo_content = raw

    # ── Build context for the LLM ────────────────────────────────────────
    summary = state.get_progress_summary()

    if todo_content:
        context_prefix = (
            "You are Daedalus, an autonomous AI coding agent.\n\n"
            "The user asked to see the current task list. "
            "Print the task list below VERBATIM (keep markdown formatting). "
            "Then add one short sentence about overall progress. "
            "Do NOT summarise or rewrite the tasks — show them exactly as listed.\n\n"
            f"Current workflow state: {summary}\n\n"
            f"TASK LIST:\n{todo_content}\n\n"
            f"User question: {state.user_request}"
        )
    else:
        context_prefix = (
            "You are Daedalus, an autonomous AI coding agent. "
            "You help developers by autonomously cloning, analysing, coding, testing, "
            "and documenting software repositories. "
            "You use a dual-coder system (Claude + GPT) with peer review, human approval "
            "gates, checkpoint/resume, and a code intelligence pipeline.\n\n"
            "Answer the user's question conversationally. "
            "If it's about workflow status, use the context below. "
            "If the user asks to continue or resume a task and there is no active plan, "
            "tell them to type 'continue' to resume from the last checkpoint or "
            "provide a new task description. "
            "Keep your answer concise (3-6 sentences). Do not ask follow-up questions.\n\n"
            f"Current workflow state:\n{summary}\n\n"
            f"User question: {state.user_request}"
        )

    answer, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=context_prefix)],
        tools=None, inject_memory=False, node="status",
    )

    emit_agent_response("planner", answer)
    emit_node_end("planner", "Status", "Response sent")
    return {
        "planner_response": answer,
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "status_answered",
        "input_intent": "status",
        **budget_update,
    }
