"""Tests for the Plan Approval Gate — Issue #38.

Covers:
- New WorkflowPhase value
- New GraphState fields
- plan_approval_gate_node routing (all three exit paths)
- planner_plan_node behaviour change (sets needs_plan_approval)
- Revision guard (plan_revision_count >= 1 skips gate)
- planner_env_fix_node does NOT set needs_plan_approval
- EventCategory.PLAN_APPROVAL and emit_plan_approval_needed
- POST /api/plan-approve (approve / reject / with feedback)
- GET /api/status includes needs_plan_approval and pending_plan_items
- Resume routing handles WAITING_FOR_PLAN_APPROVAL phase
- _format_plan_for_human helper output
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.events import EventCategory, WorkflowEvent
from app.core.nodes import _format_plan_for_human, plan_approval_gate_node
from app.core.orchestrator import (
    _route_after_plan,
    _route_after_plan_approval_gate,
    _route_after_resume,
)
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_item(desc: str = "Do something", task_type: str = "coding", agent: str = "coder_a") -> TodoItem:
    return TodoItem(
        id="item-001",
        description=desc,
        task_type=task_type,
        acceptance_criteria=["It works."],
        verification_commands=["pytest"],
        assigned_agent=agent,
        status=ItemStatus.PENDING,
    )


def _state(**kwargs) -> GraphState:
    defaults = dict(
        user_request="add health endpoint",
        todo_items=[_make_item()],
        phase=WorkflowPhase.PLANNING,
    )
    defaults.update(kwargs)
    return GraphState(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# 1. WorkflowPhase new value
# ═══════════════════════════════════════════════════════════════════════════

def test_workflow_phase_has_waiting_for_plan_approval():
    assert WorkflowPhase.WAITING_FOR_PLAN_APPROVAL == "waiting_for_plan_approval"


def test_workflow_phase_waiting_for_plan_approval_is_str():
    assert isinstance(WorkflowPhase.WAITING_FOR_PLAN_APPROVAL, str)


# ═══════════════════════════════════════════════════════════════════════════
# 2. GraphState new fields
# ═══════════════════════════════════════════════════════════════════════════

def test_graphstate_has_needs_plan_approval():
    s = GraphState()
    assert s.needs_plan_approval is False


def test_graphstate_has_plan_approved():
    s = GraphState()
    assert s.plan_approved is False


def test_graphstate_has_plan_approval_feedback():
    s = GraphState()
    assert s.plan_approval_feedback == ""


def test_graphstate_has_plan_revision_count():
    s = GraphState()
    assert s.plan_revision_count == 0


def test_graphstate_plan_fields_serialise():
    s = GraphState(
        needs_plan_approval=True,
        plan_approved=False,
        plan_approval_feedback="add tests",
        plan_revision_count=1,
    )
    d = s.model_dump()
    assert d["needs_plan_approval"] is True
    assert d["plan_approval_feedback"] == "add tests"
    assert d["plan_revision_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. plan_approval_gate_node — still waiting
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_halts_when_not_yet_approved():
    state = _state(needs_plan_approval=True, plan_approved=False)
    with patch("app.core.nodes.emit_plan_approval_needed"), \
         patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        result = plan_approval_gate_node(state)
    assert result["phase"] == WorkflowPhase.WAITING_FOR_PLAN_APPROVAL
    assert result["needs_plan_approval"] is True
    assert result["stop_reason"] == "waiting_for_plan_approval"


def test_gate_emits_plan_approval_event_when_waiting():
    state = _state(needs_plan_approval=True, plan_approved=False)
    with patch("app.core.nodes.emit_plan_approval_needed") as mock_emit, \
         patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        plan_approval_gate_node(state)
    mock_emit.assert_called_once()
    args = mock_emit.call_args
    assert args[0][2] == state.todo_items   # items argument


def test_gate_status_waiting_includes_pending_plan_items_metadata():
    state = _state(needs_plan_approval=True, plan_approved=False, todo_items=[_make_item("add endpoint", task_type="coding", agent="coder_a")])
    with patch("app.core.nodes.emit_plan_approval_needed"), \
         patch("app.core.nodes.emit_status") as mock_status, \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        plan_approval_gate_node(state)

    mock_status.assert_called_once()
    kwargs = mock_status.call_args.kwargs
    assert kwargs["phase"] == WorkflowPhase.WAITING_FOR_PLAN_APPROVAL
    assert kwargs["needs_plan_approval"] is True
    assert isinstance(kwargs["pending_plan_items"], list)
    assert kwargs["pending_plan_items"][0]["description"] == "add endpoint"
    assert kwargs["pending_plan_items"][0]["task_type"] == "coding"
    assert kwargs["pending_plan_items"][0]["assigned_agent"] == "coder_a"


# ═══════════════════════════════════════════════════════════════════════════
# 4. plan_approval_gate_node — approved without feedback → CODING
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_routes_to_coding_on_pure_approve():
    state = _state(plan_approved=True, plan_approval_feedback="")
    with patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        result = plan_approval_gate_node(state)
    assert result["phase"] == WorkflowPhase.CODING
    assert result["needs_plan_approval"] is False
    assert result["plan_approved"] is False
    assert result["plan_approval_feedback"] == ""
    assert result["stop_reason"] == ""


# ═══════════════════════════════════════════════════════════════════════════
# 5. plan_approval_gate_node — approved with feedback → PLANNING (revise)
# ═══════════════════════════════════════════════════════════════════════════

def test_gate_routes_to_planning_on_approve_with_feedback():
    state = _state(plan_approved=True, plan_approval_feedback="add error handling", plan_revision_count=0)
    with patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        result = plan_approval_gate_node(state)
    assert result["phase"] == WorkflowPhase.PLANNING
    assert result["needs_plan_approval"] is False
    assert result["plan_approved"] is False
    assert result["plan_revision_count"] == 1
    assert result["stop_reason"] == ""


def test_gate_increments_revision_count():
    state = _state(plan_approved=True, plan_approval_feedback="revise it", plan_revision_count=0)
    with patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        result = plan_approval_gate_node(state)
    assert result["plan_revision_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. Revision guard in planner_plan_node
# ═══════════════════════════════════════════════════════════════════════════

def test_planner_skips_gate_after_one_revision():
    """When plan_revision_count >= 1 planner must set needs_plan_approval=False."""
    # We test this via the state that planner_plan_node produces.
    # Rather than running the full LLM planner, we verify the guard logic
    # directly by inspecting what planner_plan_node would set.
    # We do this by checking that the condition `state.plan_revision_count < 1`
    # correctly gates the flag — tested via the node's actual return on a mock.
    from app.core.nodes import planner_plan_node
    from app.core.state import GraphState, WorkflowPhase

    base_state = GraphState(
        user_request="add endpoint",
        input_intent="code",
        plan_revision_count=1,   # already revised once
        repo_root="/tmp",
        branch_name="feature/test",
    )

    mock_items = [_make_item("implement endpoint")]

    with patch("app.core.nodes._invoke_agent", return_value='{"plan": [{"description": "implement endpoint", "task_type": "coding", "acceptance_criteria": ["works"], "verification_commands": ["pytest"]}]}'), \
         patch("app.core.nodes._parse_plan_from_result", return_value=mock_items), \
         patch("app.core.nodes.ensure_memory_files"), \
         patch("app.core.nodes.get_memory_stats", return_value={}), \
         patch("app.core.nodes.load_all_memory", return_value=""), \
         patch("app.core.nodes.read_file"), \
         patch("app.core.nodes.git_create_branch"), \
         patch("app.core.nodes._save_checkpoint_snapshot"), \
         patch("app.core.nodes.emit_plan"), \
         patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"), \
         patch("app.core.nodes.history_summary", return_value=""), \
         patch("app.core.nodes.select_agent_thompson", return_value=("coder_a", {})), \
         patch("app.core.nodes._write_todo_file"):
        result = planner_plan_node(base_state)

    # With plan_revision_count=1, needs_plan_approval must be False
    assert result.get("needs_plan_approval") is False
    # And phase must be CODING (bypassing the gate)
    assert result.get("phase") == WorkflowPhase.CODING


# ═══════════════════════════════════════════════════════════════════════════
# 7. planner_env_fix_node does NOT set needs_plan_approval
# ═══════════════════════════════════════════════════════════════════════════

def test_env_fix_node_does_not_set_needs_plan_approval():
    from app.core.nodes import planner_env_fix_node

    state = _state(
        last_test_result="ModuleNotFoundError: No module named 'httpx'",
        env_fix_attempts=0,
        active_coder="coder_a",
        repo_facts={"tech_stack": {"language": "python", "package_manager": "pip"}},
    )

    with patch("app.core.nodes._invoke_agent", return_value='{"description": "Install httpx", "command": "pip install httpx", "reason": "missing"}'), \
         patch("app.core.nodes.emit_status"), \
         patch("app.core.nodes.emit_node_start"), \
         patch("app.core.nodes.emit_node_end"):
        result = planner_env_fix_node(state)

    assert "needs_plan_approval" not in result or result.get("needs_plan_approval") is False


# ═══════════════════════════════════════════════════════════════════════════
# 8. EventCategory and emit_plan_approval_needed
# ═══════════════════════════════════════════════════════════════════════════

def test_event_category_has_plan_approval():
    assert EventCategory.PLAN_APPROVAL == "plan_approval"


def test_emit_plan_approval_needed_emits_event():
    from app.core.events import emit_plan_approval_needed

    items = [_make_item("implement /health"), _make_item("write tests", task_type="testing")]
    emitted: list[WorkflowEvent] = []

    with patch("app.core.events.emit", side_effect=lambda e: emitted.append(e)):
        emit_plan_approval_needed("planner", "1. coding: implement /health\n2. testing: write tests", items)

    assert len(emitted) == 1
    ev = emitted[0]
    assert ev.category == EventCategory.PLAN_APPROVAL
    assert ev.agent == "planner"
    assert "2" in ev.title   # items_count in title


def test_emit_plan_approval_needed_metadata_shape():
    from app.core.events import emit_plan_approval_needed

    items = [_make_item("implement /health", task_type="coding", agent="coder_a")]
    emitted: list[WorkflowEvent] = []

    with patch("app.core.events.emit", side_effect=lambda e: emitted.append(e)):
        emit_plan_approval_needed("planner", "plan summary", items)

    meta = emitted[0].metadata
    assert "items_count" in meta
    assert "items" in meta
    assert "plan_summary" in meta
    assert meta["items_count"] == 1
    assert meta["items"][0]["task_type"] == "coding"
    assert meta["items"][0]["assigned_agent"] == "coder_a"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Orchestrator routing functions
# ═══════════════════════════════════════════════════════════════════════════

def test_route_after_plan_goes_to_gate_when_needs_approval():
    state = _state(needs_plan_approval=True, todo_items=[_make_item()])
    assert _route_after_plan(state) == "plan_approval_gate"


def test_route_after_plan_goes_to_coder_when_no_approval_needed():
    state = _state(needs_plan_approval=False, todo_items=[_make_item()])
    assert _route_after_plan(state) == "coder"


def test_route_after_plan_goes_to_stopped_when_no_items():
    state = _state(needs_plan_approval=False, todo_items=[])
    assert _route_after_plan(state) == "stopped"


def test_route_after_plan_approval_gate_stopped_when_waiting():
    state = _state(phase=WorkflowPhase.WAITING_FOR_PLAN_APPROVAL)
    assert _route_after_plan_approval_gate(state) == "stopped"


def test_route_after_plan_approval_gate_planner_when_revision():
    state = _state(phase=WorkflowPhase.PLANNING)
    assert _route_after_plan_approval_gate(state) == "planner"


def test_route_after_plan_approval_gate_coder_when_approved():
    state = _state(phase=WorkflowPhase.CODING)
    assert _route_after_plan_approval_gate(state) == "coder"


def test_route_after_resume_goes_to_gate_when_waiting_for_plan():
    state = _state(phase=WorkflowPhase.WAITING_FOR_PLAN_APPROVAL)
    assert _route_after_resume(state) == "plan_approval_gate"


# ═══════════════════════════════════════════════════════════════════════════
# 10. _format_plan_for_human
# ═══════════════════════════════════════════════════════════════════════════

def test_format_plan_for_human_numbers_items():
    items = [_make_item("write endpoint"), _make_item("write tests", task_type="testing")]
    result = _format_plan_for_human(items)
    assert "1." in result
    assert "2." in result


def test_format_plan_for_human_includes_task_type():
    items = [_make_item("write endpoint", task_type="coding")]
    result = _format_plan_for_human(items)
    assert "coding" in result


def test_format_plan_for_human_includes_description():
    items = [_make_item("implement the /health route")]
    result = _format_plan_for_human(items)
    assert "implement the /health route" in result


def test_format_plan_for_human_empty_list():
    assert _format_plan_for_human([]) == "(empty plan)"


# ═══════════════════════════════════════════════════════════════════════════
# 11. /api/plan-approve endpoint
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def client_with_pending_plan():
    """Return a TestClient with _current_state set to a plan-awaiting state."""
    import app.web.server as srv
    from app.web.server import app as fastapi_app

    state = _state(
        needs_plan_approval=True,
        plan_approved=False,
        plan_approval_feedback="",
        phase=WorkflowPhase.WAITING_FOR_PLAN_APPROVAL,
    )
    original_state = srv._current_state
    original_queue = srv._task_queue

    srv._current_state = state
    srv._task_queue = MagicMock()
    srv._task_queue.put = AsyncMock()

    yield TestClient(fastapi_app), state

    srv._current_state = original_state
    srv._task_queue = original_queue


def test_plan_approve_approve_sets_flags(client_with_pending_plan):
    client, state = client_with_pending_plan
    resp = client.post("/api/plan-approve", json={"approved": True, "feedback": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["action"] == "coding_started"
    assert state.plan_approved is True
    assert state.needs_plan_approval is False


def test_plan_approve_with_feedback_sets_feedback(client_with_pending_plan):
    client, state = client_with_pending_plan
    resp = client.post("/api/plan-approve", json={"approved": True, "feedback": "add retry logic"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "revision_requested"
    assert state.plan_approval_feedback == "add retry logic"


def test_plan_approve_reject_stops_workflow(client_with_pending_plan):
    client, state = client_with_pending_plan
    resp = client.post("/api/plan-approve", json={"approved": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stopped"
    assert state.phase == WorkflowPhase.STOPPED
    assert state.stop_reason == "plan_rejected_by_human"


def test_plan_approve_returns_400_when_no_active_workflow():
    import app.web.server as srv
    from app.web.server import app as fastapi_app

    original = srv._current_state
    srv._current_state = None
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/plan-approve", json={"approved": True})
        assert resp.status_code == 400
    finally:
        srv._current_state = original


def test_plan_approve_returns_400_when_no_pending_plan():
    import app.web.server as srv
    from app.web.server import app as fastapi_app

    original = srv._current_state
    srv._current_state = _state(needs_plan_approval=False)
    try:
        client = TestClient(fastapi_app)
        resp = client.post("/api/plan-approve", json={"approved": True})
        assert resp.status_code == 400
    finally:
        srv._current_state = original


# ═══════════════════════════════════════════════════════════════════════════
# 12. GET /api/status includes plan approval fields
# ═══════════════════════════════════════════════════════════════════════════

def test_status_includes_needs_plan_approval_true():
    import app.web.server as srv
    from app.web.server import app as fastapi_app

    original = srv._current_state
    srv._current_state = _state(
        needs_plan_approval=True,
        todo_items=[_make_item("add endpoint")],
        phase=WorkflowPhase.WAITING_FOR_PLAN_APPROVAL,
    )
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/status")
        data = resp.json()
        assert data["needs_plan_approval"] is True
        assert len(data["pending_plan_items"]) == 1
        assert data["pending_plan_items"][0]["description"] == "add endpoint"
        assert "assigned_agent" in data["pending_plan_items"][0]
    finally:
        srv._current_state = original


def test_status_pending_plan_items_empty_when_not_waiting():
    import app.web.server as srv
    from app.web.server import app as fastapi_app

    original = srv._current_state
    srv._current_state = _state(
        needs_plan_approval=False,
        todo_items=[_make_item("add endpoint")],
        phase=WorkflowPhase.CODING,
    )
    try:
        client = TestClient(fastapi_app)
        resp = client.get("/api/status")
        data = resp.json()
        assert data["needs_plan_approval"] is False
        assert data["pending_plan_items"] == []
    finally:
        srv._current_state = original
