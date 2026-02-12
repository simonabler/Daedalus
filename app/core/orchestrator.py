"""LangGraph orchestrator for the multi-agent workflow."""

from __future__ import annotations

import asyncio
import platform
import threading

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.core.nodes import (
    coder_node,
    committer_node,
    learn_from_review_node,
    peer_review_node,
    planner_decide_node,
    planner_plan_node,
    planner_review_node,
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
    return "coder"


def _route_after_coder(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "peer_review"


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
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "decide"


def _route_after_decide(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "commit"


def _route_after_commit(state: GraphState) -> str:
    if _shutdown_event.is_set() or state.phase == WorkflowPhase.COMPLETE:
        return "complete"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "complete"


def build_graph() -> StateGraph:
    """Construct the workflow graph."""
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_plan_node)
    graph.add_node("coder", coder_node)
    graph.add_node("peer_review", peer_review_node)
    graph.add_node("learn", learn_from_review_node)
    graph.add_node("planner_review", planner_review_node)
    graph.add_node("tester", tester_node)
    graph.add_node("decide", planner_decide_node)
    graph.add_node("commit", committer_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", _route_after_plan, {"coder": "coder", "stopped": END})
    graph.add_conditional_edges("coder", _route_after_coder, {"peer_review": "peer_review", "stopped": END})
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
    graph.add_conditional_edges("tester", _route_after_tester, {"decide": "decide", "coder": "coder", "stopped": END})
    graph.add_conditional_edges("decide", _route_after_decide, {"commit": "commit", "stopped": END})
    graph.add_conditional_edges("commit", _route_after_commit, {"coder": "coder", "complete": END})

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
