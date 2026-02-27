"""Research node."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *

def research_node(state: GraphState) -> dict:
    """Research branch without repository mutation tools."""
    emit_node_start("planner", "Research", item_desc=state.user_request[:100])
    prompt = (
        "You are a research assistant for a software workflow.\n"
        "Answer the user's request in analysis mode only.\n"
        "Do not propose or perform code/file/git changes.\n\n"
        f"User request:\n{state.user_request}\n"
    )
    answer, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=None, inject_memory=False, node="research",
    )
    # Emit the answer as a visible response — shown as an expanded reply bubble
    emit_agent_response("planner", answer)
    emit_node_end("planner", "Research", "Research response prepared")
    return {
        "planner_response": answer,
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "research_answered",
        "input_intent": "research",
        **budget_update,
    }
