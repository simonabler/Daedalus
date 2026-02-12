"""Tests for the dual-coder LangGraph orchestrator — graph structure, routing, and coder assignment."""

from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase


class TestCoderAssignment:
    """Verify coder/reviewer alternation logic."""

    def test_even_item_assigns_coder_a(self):
        from app.core.nodes import _assign_coder_pair
        coder, reviewer = _assign_coder_pair(0)
        assert coder == "coder_a"
        assert reviewer == "reviewer_b"

    def test_odd_item_assigns_coder_b(self):
        from app.core.nodes import _assign_coder_pair
        coder, reviewer = _assign_coder_pair(1)
        assert coder == "coder_b"
        assert reviewer == "reviewer_a"

    def test_alternation_pattern(self):
        from app.core.nodes import _assign_coder_pair
        results = [_assign_coder_pair(i) for i in range(6)]
        assert results == [
            ("coder_a", "reviewer_b"),
            ("coder_b", "reviewer_a"),
            ("coder_a", "reviewer_b"),
            ("coder_b", "reviewer_a"),
            ("coder_a", "reviewer_b"),
            ("coder_b", "reviewer_a"),
        ]


class TestRouting:
    def test_route_after_plan_to_coder(self):
        from app.core.orchestrator import _route_after_plan
        state = GraphState(
            phase=WorkflowPhase.CODING,
            todo_items=[TodoItem(id="item-001", description="test")],
        )
        assert _route_after_plan(state) == "coder"

    def test_route_after_plan_stopped(self):
        from app.core.orchestrator import _route_after_plan
        state = GraphState(phase=WorkflowPhase.STOPPED)
        assert _route_after_plan(state) == "stopped"

    def test_route_after_plan_no_items(self):
        from app.core.orchestrator import _route_after_plan
        state = GraphState(phase=WorkflowPhase.CODING, todo_items=[])
        assert _route_after_plan(state) == "stopped"

    def test_route_after_coder_to_peer_review(self):
        from app.core.orchestrator import _route_after_coder
        state = GraphState(phase=WorkflowPhase.PEER_REVIEWING)
        assert _route_after_coder(state) == "peer_review"

    def test_route_after_peer_review_always_goes_to_learn(self):
        from app.core.orchestrator import _route_after_peer_review
        # APPROVE → learn
        state = GraphState(phase=WorkflowPhase.REVIEWING)
        assert _route_after_peer_review(state) == "learn"
        # REWORK → also learn (extracts insights even from rework)
        state2 = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_peer_review(state2) == "learn"

    def test_route_after_learn_approve(self):
        from app.core.orchestrator import _route_after_learn
        state = GraphState(phase=WorkflowPhase.REVIEWING)
        assert _route_after_learn(state) == "planner_review"

    def test_route_after_learn_rework(self):
        from app.core.orchestrator import _route_after_learn
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_learn(state) == "coder"

    def test_route_after_learn_escalated_testing(self):
        from app.core.orchestrator import _route_after_learn
        state = GraphState(phase=WorkflowPhase.TESTING)
        assert _route_after_learn(state) == "tester"

    def test_route_after_planner_review_approve(self):
        from app.core.orchestrator import _route_after_planner_review
        state = GraphState(phase=WorkflowPhase.TESTING)
        assert _route_after_planner_review(state) == "tester"

    def test_route_after_planner_review_rework(self):
        from app.core.orchestrator import _route_after_planner_review
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_planner_review(state) == "coder"

    def test_route_after_tester_pass(self):
        from app.core.orchestrator import _route_after_tester
        state = GraphState(phase=WorkflowPhase.DECIDING)
        assert _route_after_tester(state) == "decide"

    def test_route_after_tester_fail(self):
        from app.core.orchestrator import _route_after_tester
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_tester(state) == "coder"

    def test_route_after_commit_more_items(self):
        from app.core.orchestrator import _route_after_commit
        state = GraphState(phase=WorkflowPhase.CODING)
        assert _route_after_commit(state) == "coder"

    def test_route_after_commit_complete(self):
        from app.core.orchestrator import _route_after_commit
        state = GraphState(phase=WorkflowPhase.COMPLETE)
        assert _route_after_commit(state) == "complete"


class TestGraphBuild:
    def test_graph_compiles(self):
        from app.core.orchestrator import compile_graph
        compiled = compile_graph()
        assert compiled is not None

    def test_graph_has_peer_review_node(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        # Verify peer_review node exists in the graph
        assert "peer_review" in graph.nodes

    def test_graph_has_learn_node(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        assert "learn" in graph.nodes


class TestParsePlan:
    def test_parse_checkboxes(self):
        from app.core.nodes import _parse_plan_from_result
        text = """
Here's the plan:
- [ ] Set up project structure
- [ ] Add authentication module
- [ ] Write tests
"""
        items = _parse_plan_from_result(text)
        assert len(items) == 3
        assert items[0].description == "Set up project structure"
        assert items[0].status == ItemStatus.PENDING

    def test_parse_numbered(self):
        from app.core.nodes import _parse_plan_from_result
        text = """
1. Create the API endpoint
2. Add validation logic
3. Write integration tests
"""
        items = _parse_plan_from_result(text)
        assert len(items) == 3

    def test_parse_empty(self):
        from app.core.nodes import _parse_plan_from_result
        items = _parse_plan_from_result("No items here, just text.")
        assert len(items) == 0

    def test_parse_json_plan(self):
        from app.core.nodes import _parse_plan_from_result
        text = """
{
  "plan": [
    {
      "description": "Update README",
      "task_type": "documentation",
      "acceptance_criteria": ["README reflects new flow"],
      "verification_commands": []
    }
  ]
}
"""
        items = _parse_plan_from_result(text)
        assert len(items) == 1
        assert items[0].task_type == "documentation"


class TestExtractCommitMessage:
    def test_extract_from_peer_review(self):
        from app.core.nodes import _extract_commit_message
        peer_notes = """
Review looks good.
Suggested commit: `feat(auth): add JWT token validation`
All criteria met.
"""
        msg = _extract_commit_message(peer_notes, "Planner says ok", "fallback desc")
        assert msg == "feat(auth): add JWT token validation"

    def test_extract_from_planner_when_peer_has_none(self):
        from app.core.nodes import _extract_commit_message
        planner_notes = """
Approved. Commit: `fix(api): handle null response`
"""
        msg = _extract_commit_message("Looks fine, APPROVE", planner_notes, "fallback")
        assert msg == "fix(api): handle null response"

    def test_fallback(self):
        from app.core.nodes import _extract_commit_message
        msg = _extract_commit_message("Looks good!", "Also good!", "add user login")
        assert msg == "feat: add user login"


class TestDualCoderState:
    """Verify GraphState correctly tracks dual-coder fields."""

    def test_default_coder_assignment(self):
        state = GraphState()
        assert state.active_coder == "coder_a"
        assert state.active_reviewer == "reviewer_b"

    def test_peer_review_fields(self):
        state = GraphState(
            active_coder="coder_b",
            active_reviewer="reviewer_a",
            peer_review_verdict="APPROVE",
            peer_review_notes="Clean implementation.",
        )
        assert state.active_coder == "coder_b"
        assert state.active_reviewer == "reviewer_a"
        assert state.peer_review_verdict == "APPROVE"

    def test_peer_review_phase_exists(self):
        assert WorkflowPhase.PEER_REVIEWING == "peer_reviewing"

    def test_progress_summary(self):
        state = GraphState(
            phase=WorkflowPhase.PEER_REVIEWING,
            branch_name="feature/test",
            todo_items=[
                TodoItem(id="1", description="task one", status=ItemStatus.DONE),
                TodoItem(id="2", description="task two", status=ItemStatus.IN_REVIEW),
            ],
            current_item_index=1,
        )
        summary = state.get_progress_summary()
        assert "1/2" in summary
        assert "peer_reviewing" in summary
