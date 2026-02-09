"""LangGraph orchestrator — dual-coder workflow with peer review.

Workflow:
  Planner → Coder (A|B) → Peer Review (B|A) → Planner Review → Tester → Decide → Commit
                 ↑              |  REWORK           |  REWORK         |  FAIL
                 └──────────────┘───────────────────┘────────────────┘

Coder assignment alternates per TODO item:
  - Even items (0, 2, 4…): Coder A (Claude) codes, Reviewer B (GPT-5.2) reviews
  - Odd  items (1, 3, 5…): Coder B (GPT-5.2) codes, Reviewer A (Claude) reviews
"""

from __future__ import annotations

import asyncio

from langgraph.graph import StateGraph, END

from app.core.logging import get_logger
from app.core.state import GraphState, WorkflowPhase
from app.core.nodes import (
    planner_plan_node,
    coder_node,
    peer_review_node,
    learn_from_review_node,
    planner_review_node,
    tester_node,
    planner_decide_node,
    committer_node,
)

logger = get_logger("core.orchestrator")


# ── Routing functions ─────────────────────────────────────────────────────

def _route_after_plan(state: GraphState) -> str:
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if not state.todo_items:
        return "stopped"
    return "coder"


def _route_after_coder(state: GraphState) -> str:
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "peer_review"


def _route_after_peer_review(state: GraphState) -> str:
    """After peer review: always go to learn node (extracts insights), then route."""
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    # Always extract learnings first, even on REWORK
    return "learn"


def _route_after_learn(state: GraphState) -> str:
    """After learning extraction: APPROVE → planner review, REWORK → back to coder."""
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.CODING:
        # Peer reviewer said REWORK — go back to coder
        return "coder"
    return "planner_review"


def _route_after_planner_review(state: GraphState) -> str:
    """After planner review: APPROVE → tester, REWORK → back to coder."""
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.CODING:
        return "coder"
    return "tester"


def _route_after_tester(state: GraphState) -> str:
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    if state.phase == WorkflowPhase.CODING:
        # Tests failed — back to original coder
        return "coder"
    return "decide"


def _route_after_decide(state: GraphState) -> str:
    if state.phase == WorkflowPhase.STOPPED:
        return "stopped"
    return "commit"


def _route_after_commit(state: GraphState) -> str:
    if state.phase == WorkflowPhase.COMPLETE:
        return "complete"
    if state.phase == WorkflowPhase.CODING:
        # More items — next coder (alternated in committer_node)
        return "coder"
    return "complete"


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct the LangGraph workflow with dual-coder peer review."""
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("planner", planner_plan_node)
    graph.add_node("coder", coder_node)
    graph.add_node("peer_review", peer_review_node)
    graph.add_node("learn", learn_from_review_node)
    graph.add_node("planner_review", planner_review_node)
    graph.add_node("tester", tester_node)
    graph.add_node("decide", planner_decide_node)
    graph.add_node("commit", committer_node)

    # Entry
    graph.set_entry_point("planner")

    # Edges
    graph.add_conditional_edges("planner", _route_after_plan, {
        "coder": "coder",
        "stopped": END,
    })

    graph.add_conditional_edges("coder", _route_after_coder, {
        "peer_review": "peer_review",
        "stopped": END,
    })

    graph.add_conditional_edges("peer_review", _route_after_peer_review, {
        "learn": "learn",
        "stopped": END,
    })

    graph.add_conditional_edges("learn", _route_after_learn, {
        "planner_review": "planner_review",
        "coder": "coder",
        "stopped": END,
    })

    graph.add_conditional_edges("planner_review", _route_after_planner_review, {
        "tester": "tester",
        "coder": "coder",
        "stopped": END,
    })

    graph.add_conditional_edges("tester", _route_after_tester, {
        "decide": "decide",
        "coder": "coder",
        "stopped": END,
    })

    graph.add_conditional_edges("decide", _route_after_decide, {
        "commit": "commit",
        "stopped": END,
    })

    graph.add_conditional_edges("commit", _route_after_commit, {
        "coder": "coder",
        "complete": END,
    })

    return graph


def compile_graph():
    """Build and compile the graph for execution."""
    graph = build_graph()
    return graph.compile()


async def run_workflow(user_request: str, repo_path: str) -> GraphState:
    """Execute the full dual-coder workflow for a user request."""
    from app.core.config import get_settings
    settings = get_settings()

    compiled = compile_graph()

    initial_state = GraphState(
        user_request=user_request,
        repo_root=repo_path or settings.target_repo_path,
        phase=WorkflowPhase.PLANNING,
        active_coder="coder_a",
        active_reviewer="reviewer_b",
    )

    logger.info(
        "Starting dual-coder workflow | request: %s | repo: %s",
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
