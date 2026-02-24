"""LangGraph shared state definition for the orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

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
    task_type: str = "coding"  # coding | documentation | testing | ops
    assigned_agent: str = ""  # coder_a | coder_b | documenter
    assigned_reviewer: str = ""  # reviewer_a | reviewer_b
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    status: ItemStatus = ItemStatus.PENDING
    review_notes: str = ""
    test_report: str = ""
    commit_message: str = ""
    iteration_count: int = 0
    rework_count: int = 0
    test_fail_count: int = 0


class WorkflowPhase(StrEnum):
    IDLE = "idle"
    ROUTING = "routing"
    LOADING_CONTEXT = "loading_context"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    CODING = "coding"
    PEER_REVIEWING = "peer_reviewing"    # cross-coder review
    REVIEWING = "reviewing"               # planner final gate review
    TESTING = "testing"
    DECIDING = "deciding"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_ANSWER = "waiting_for_answer"
    COMMITTING = "committing"
    DOCUMENTING = "documenting"
    ENV_FIXING = "env_fixing"   # planner is creating an env-setup fix item
    COMPLETE = "complete"
    STOPPED = "stopped"


class GraphState(BaseModel):
    """The complete state passed between LangGraph nodes."""

    # ── Repository context ────────────────────────────────────────────
    repo_root: str = ""
    branch_name: str = ""

    # ── User request ──────────────────────────────────────────────────
    user_request: str = ""
    execution_platform: str = ""
    input_intent: str = ""
    planner_response: str = ""
    agent_instructions: str = ""
    repo_facts: dict[str, Any] = Field(default_factory=dict)
    context_listing: str = ""
    context_loaded: bool = False
    needs_human_approval: bool = False
    pending_approval: dict[str, Any] = Field(default_factory=dict)
    approval_history: list[dict[str, Any]] = Field(default_factory=list)

    # ── Mid-task coder question ───────────────────────────────────────
    # A coder can pause mid-item to ask the human a critical question.
    # The workflow halts at answer_gate_node until coder_question_answer
    # is populated (via /api/answer or Telegram /answer).
    needs_coder_answer: bool = False
    coder_question: str = ""           # the question text
    coder_question_context: str = ""   # why the coder is asking
    coder_question_options: list[str] = Field(default_factory=list)  # suggested choices
    coder_question_asked_by: str = ""  # "coder_a" | "coder_b"
    coder_question_answer: str = ""    # human's answer (filled by UI/Telegram)
    state_checkpoint_id: str | None = None
    last_checkpoint_path: str | None = None
    resumed_from_checkpoint: bool = False

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

    # ── Static analysis ──────────────────────────────────────────────────
    # Populated by context_loader_node after repo facts are detected.
    # Each entry is a dict representation of a StaticIssue.
    static_issues: list[dict[str, Any]] = Field(default_factory=list)

    # ── Call graph ───────────────────────────────────────────────────────
    # Populated by context_loader_node. Serialised CallGraph.to_dict().
    # Keys: callers, callees, file_map, language, files_analysed, parse_errors
    call_graph: dict[str, Any] = Field(default_factory=dict)

    # ── Dependency graph ──────────────────────────────────────────────
    # Populated by context_loader_node. Serialised DependencyGraph.to_dict().
    # Keys: imports, importers, cycles, coupling_scores, language, ...
    dependency_graph: dict[str, Any] = Field(default_factory=dict)
    # Convenience: detected circular import paths (each path = list[str])
    dep_cycles: list[list[str]] = Field(default_factory=list)

    # ── Code smells ───────────────────────────────────────────────────
    # Populated by context_loader_node. Each entry is a CodeSmell.model_dump().
    # Sorted: errors first, then warnings, then info.
    code_smells: list[dict[str, Any]] = Field(default_factory=list)

    # ── Code intelligence cache ───────────────────────────────────────
    # Set by code_intelligence_node. Tracks whether analysis was loaded from cache.
    intelligence_cache_key: str = ""      # git commit hash used as cache key
    intelligence_cached: bool = False     # True = results loaded from cache

    # ── Metadata ──────────────────────────────────────────────────────
    total_iterations: int = 0
    completed_items: int = 0
    env_fix_attempts: int = 0   # number of env-fix rounds attempted for current item

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

    @property
    def checkpoint_id(self) -> str | None:
        return self.state_checkpoint_id

    @checkpoint_id.setter
    def checkpoint_id(self, value: str | None) -> None:
        self.state_checkpoint_id = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphState:
        """Deserialize state payloads from checkpoints or persisted snapshots."""
        if "checkpoint_id" in data and "state_checkpoint_id" not in data:
            data = {**data, "state_checkpoint_id": data["checkpoint_id"]}
        return cls(**data)
