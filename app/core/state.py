"""LangGraph shared state definition for the orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ItemStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class TodoItem(BaseModel):
    """A single work item from the plan."""
    id: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    status: ItemStatus = ItemStatus.PENDING
    review_notes: str = ""
    test_report: str = ""
    commit_message: str = ""
    iteration_count: int = 0


class WorkflowPhase(StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    CODING = "coding"
    PEER_REVIEWING = "peer_reviewing"    # cross-coder review
    REVIEWING = "reviewing"               # planner final gate review
    TESTING = "testing"
    DECIDING = "deciding"
    COMMITTING = "committing"
    COMPLETE = "complete"
    STOPPED = "stopped"


class GraphState(BaseModel):
    """The complete state passed between LangGraph nodes."""

    # ── Repository context ────────────────────────────────────────────
    repo_root: str = ""
    branch_name: str = ""

    # ── User request ──────────────────────────────────────────────────
    user_request: str = ""

    # ── Plan & progress ───────────────────────────────────────────────
    todo_items: list[TodoItem] = Field(default_factory=list)
    current_item_index: int = -1
    phase: WorkflowPhase = WorkflowPhase.IDLE

    # ── Agent outputs ─────────────────────────────────────────────────
    last_coder_result: str = ""
    last_test_result: str = ""
    last_review_verdict: str = ""  # "APPROVE" | "REWORK" | ...
    review_notes: str = ""

    # ── Dual-coder peer review ────────────────────────────────────────
    # Tracks which coder implemented the current item and who reviews it.
    # Alternates: item 0 → coder_a codes / reviewer_b reviews,
    #             item 1 → coder_b codes / reviewer_a reviews, etc.
    active_coder: str = "coder_a"    # "coder_a" | "coder_b"
    active_reviewer: str = "reviewer_b"  # "reviewer_a" | "reviewer_b"
    peer_review_notes: str = ""       # review output from the peer reviewer
    peer_review_verdict: str = ""     # "APPROVE" | "REWORK"

    # ── Control flow ──────────────────────────────────────────────────
    stop_reason: str = ""
    needs_replan: bool = False
    error_message: str = ""

    # ── Messages (for LangGraph message passing) ──────────────────────
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────
    total_iterations: int = 0
    completed_items: int = 0

    @property
    def current_item(self) -> TodoItem | None:
        if 0 <= self.current_item_index < len(self.todo_items):
            return self.todo_items[self.current_item_index]
        return None

    def get_progress_summary(self) -> str:
        done = sum(1 for i in self.todo_items if i.status == ItemStatus.DONE)
        total = len(self.todo_items)
        current = self.current_item
        return (
            f"Phase: {self.phase.value} | "
            f"Progress: {done}/{total} items done | "
            f"Current: {current.description if current else 'none'} | "
            f"Branch: {self.branch_name or 'not set'}"
        )
