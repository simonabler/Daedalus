"""LangGraph orchestrator for the multi-agent workflow."""

from __future__ import annotations

import asyncio
import platform
import threading

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.core.nodes import (
    answer_gate_node,
    plan_approval_gate_node,
    code_intelligence_node,
    coder_node,
    committer_node,
    context_loader_node,
    documenter_node,
    human_gate_node,
    learn_from_review_node,
    peer_review_node,
    planner_decide_node,
    planner_env_fix_node,
    planner_plan_node,
    planner_review_node,
    research_node,
    resume_node,
    router_node,
    status_node,
    tester_node,
)
from app.core.state import GraphState, WorkflowPhase

logger = get_logger("core.orchestrator")

_shutdown_event = threading.Event()


def request_shutdown() -> None:
    """Signal the workflow to stop after the current node finishes."""
    _shutdown_event.set()
    logger.info("Shutdown requested - workflow will stop after current node")


def reset_shutdown() -> None:
    """Clear the shutdown flag (used in tests and restart cycles)."""
    _shutdown_event.clear()


def is_shutdown_requested() -> bool:
    return _shutdown_event.is_set()


def _route_after_plan(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if not state.todo_items:
        return "stopped"
    if state.needs_plan_approval:
        return "plan_approval_gate"
    return "coder"


def _route_after_plan_approval_gate(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.WAITING_FOR_PLAN_APPROVAL:
        return "stopped"    # graph halts; /api/plan-approve queues resume
    if state.phase == WorkflowPhase.PLANNING:
        return "planner"    # revision requested — one more planning round
    return "coder"          # approved, proceed to coding


def route_after_router(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"

    intent = (state.input_intent or "").strip().lower()
    logger.info("Routing router intent '%s'", intent or "(empty)")

    if intent == "status":
        return "status"
    if intent == "research":
        return "research"
    if intent == "resume":
        return "resume"
    if intent == "code":
        return "context"

    logger.warning("Unknown intent '%s' - defaulting to context", intent)
    return "context"


def _route_after_resume(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.WAITING_FOR_APPROVAL:
        return "stopped"
    if state.phase == WorkflowPhase.WAITING_FOR_PLAN_APPROVAL:
        return "plan_approval_gate"
    if state.phase == WorkflowPhase.WAITING_FOR_ANSWER:
        return "answer_gate"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    if state.phase == WorkflowPhase.COMMITTING:
        return "commit"
    if state.phase == WorkflowPhase.DOCUMENTING:
        return "documenter"
    if state.phase == WorkflowPhase.ENV_FIXING:
        return "env_fix"
    return "complete"


def _route_after_coder(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.WAITING_FOR_ANSWER:
        return "answer_gate"
    # Env-fix items (ops tasks prepended by planner_env_fix) skip peer review
    # and go straight back to tester so the fix can be verified immediately.
    current = state.current_item
    if current is not None and current.task_type == "ops" and current.id.startswith("env_fix_"):
        return "tester"
    return "peer_review"


def _route_after_answer_gate(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.WAITING_FOR_ANSWER:
        return "stopped"   # halt until /api/answer is called
    return "coder"         # answer received — coder continues


def _route_after_peer_review(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "learn"


def _route_after_learn(state: GraphState) -> str:
    """Route by phase after extracting learnings."""
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.TESTING:
        return "tester"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "planner_review"


def _route_after_planner_review(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "tester"


def _route_after_tester(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.ENV_FIXING:
        return "env_fix"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "decide"


def _route_after_env_fix(state: GraphState) -> str:
    """After planner_env_fix creates the fix item, always go to coder.

    Env-fix items skip peer review — they are mechanical installs, not
    creative coding tasks. The coder goes directly back to tester after.
    """
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "coder"


def _route_after_decide(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "human_gate"


def _route_after_human_gate(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.needs_human_approval or state.phase == WorkflowPhase.WAITING_FOR_APPROVAL:
        return "stopped"
    return "commit"


def _route_after_commit(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.COMPLETE:
        return "complete"
    # Always pass through the documenter node after a successful commit.
    return "documenter"


def _route_after_documenter(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "complete"


def build_graph() -> StateGraph:
    """Construct the workflow graph."""
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("context", context_loader_node)
    graph.add_node("intelligence", code_intelligence_node)
    graph.add_node("status", status_node)
    graph.add_node("research", research_node)
    graph.add_node("resume", resume_node)
    graph.add_node("planner", planner_plan_node)
    graph.add_node("plan_approval_gate", plan_approval_gate_node)
    graph.add_node("coder", coder_node)
    graph.add_node("answer_gate", answer_gate_node)
    graph.add_node("peer_review", peer_review_node)
    graph.add_node("learn", learn_from_review_node)
    graph.add_node("planner_review", planner_review_node)
    graph.add_node("tester", tester_node)
    graph.add_node("decide", planner_decide_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("commit", committer_node)
    graph.add_node("documenter", documenter_node)
    graph.add_node("env_fix", planner_env_fix_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"status": "status", "research": "research", "resume": "resume", "context": "context", "stopped": END},
    )
    graph.add_edge("context", "intelligence")
    graph.add_edge("intelligence", "planner")
    graph.add_edge("status", END)
    graph.add_edge("research", END)
    graph.add_conditional_edges(
        "resume",
        _route_after_resume,
        {"coder": "coder", "answer_gate": "answer_gate", "plan_approval_gate": "plan_approval_gate",
         "commit": "commit", "documenter": "documenter", "env_fix": "env_fix", "complete": END, "stopped": END},
    )
    graph.add_conditional_edges(
        "planner",
        _route_after_plan,
        {"plan_approval_gate": "plan_approval_gate", "coder": "coder", "stopped": END},
    )
    graph.add_conditional_edges(
        "plan_approval_gate",
        _route_after_plan_approval_gate,
        {"planner": "planner", "coder": "coder", "stopped": END},
    )
    graph.add_conditional_edges(
        "coder",
        _route_after_coder,
        {"answer_gate": "answer_gate", "peer_review": "peer_review", "tester": "tester", "stopped": END},
    )
    graph.add_conditional_edges(
        "answer_gate",
        _route_after_answer_gate,
        {"coder": "coder", "stopped": END},
    )
    graph.add_conditional_edges("peer_review", _route_after_peer_review, {"learn": "learn", "stopped": END})
    graph.add_conditional_edges(
        "learn",
        _route_after_learn,
        {"planner_review": "planner_review", "tester": "tester", "coder": "coder", "stopped": END},
    )
    graph.add_conditional_edges(
        "planner_review",
        _route_after_planner_review,
        {"tester": "tester", "coder": "coder", "stopped": END},
    )
    graph.add_conditional_edges(
        "tester",
        _route_after_tester,
        {"decide": "decide", "coder": "coder", "env_fix": "env_fix", "stopped": END},
    )
    graph.add_conditional_edges(
        "env_fix", _route_after_env_fix, {"coder": "coder", "stopped": END}
    )
    graph.add_conditional_edges("decide", _route_after_decide, {"human_gate": "human_gate", "stopped": END})
    graph.add_conditional_edges("human_gate", _route_after_human_gate, {"commit": "commit", "stopped": END})
    graph.add_conditional_edges("commit", _route_after_commit, {"documenter": "documenter", "complete": END})
    graph.add_conditional_edges(
        "documenter", _route_after_documenter, {"coder": "coder", "complete": END, "stopped": END}
    )

    return graph


def compile_graph():
    graph = build_graph()
    return graph.compile()


async def run_workflow(user_request: str, repo_path: str) -> GraphState:
    """Execute the full workflow for a user request."""
    from app.core.config import get_settings

    settings = get_settings()
    compiled = compile_graph()

    initial_state = GraphState(
        user_request=user_request,
        repo_root=repo_path or settings.target_repo_path,
        execution_platform=platform.platform(),
        phase=WorkflowPhase.PLANNING,
        active_coder="coder_a",
        active_reviewer="reviewer_b",
    )

    logger.info(
        "Starting workflow | request: %s | repo: %s",
        user_request[:100],
        initial_state.repo_root,
    )

    final_state_dict = await asyncio.to_thread(compiled.invoke, initial_state.model_dump())
    final_state = GraphState(**final_state_dict)

    logger.info(
        "Workflow complete | phase: %s | completed: %d/%d items | stop_reason: %s",
        final_state.phase.value,
        final_state.completed_items,
        len(final_state.todo_items),
        final_state.stop_reason or "none",
    )

    return final_state
