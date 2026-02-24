"""Tests for tester env-failure detection and planner_env_fix routing (Issue #32).

Covers:
- _is_env_failure: all supported patterns
- _is_test_pass: pass detection
- _classify_test_output: full classification logic
- tester_node: env_failure path → ENV_FIXING phase
- tester_node: cap at _MAX_ENV_FIX_ATTEMPTS → STOPPED
- tester_node: normal PASS path unchanged
- tester_node: normal FAIL path unchanged
- planner_env_fix_node: creates fix TodoItem, increments env_fix_attempts
- planner_env_fix_node: prepends item to plan at correct index
- Orchestrator routing: tester → env_fix, env_fix → coder
- Orchestrator routing: coder → tester for env_fix items (skip peer review)
- WorkflowPhase.ENV_FIXING present in enum
- GraphState.env_fix_attempts default = 0
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.state import GraphState, TodoItem, WorkflowPhase, ItemStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(**kwargs) -> TodoItem:
    defaults = dict(id="item_0", description="Add feature X")
    defaults.update(kwargs)
    return TodoItem(**defaults)


def _make_state(**kwargs) -> GraphState:
    defaults = dict(phase=WorkflowPhase.TESTING)
    defaults.update(kwargs)
    state = GraphState(**defaults)
    return state


# ---------------------------------------------------------------------------
# 1. WorkflowPhase.ENV_FIXING and GraphState.env_fix_attempts
# ---------------------------------------------------------------------------

class TestStateAdditions:
    def test_env_fixing_phase_exists(self):
        assert hasattr(WorkflowPhase, "ENV_FIXING")
        assert WorkflowPhase.ENV_FIXING == "env_fixing"

    def test_env_fix_attempts_default_zero(self):
        state = GraphState()
        assert state.env_fix_attempts == 0

    def test_env_fix_attempts_is_int(self):
        state = GraphState(env_fix_attempts=2)
        assert state.env_fix_attempts == 2


# ---------------------------------------------------------------------------
# 2. _is_env_failure pattern matching
# ---------------------------------------------------------------------------

class TestIsEnvFailure:
    def _check(self, text: str) -> bool:
        from app.core.nodes import _is_env_failure
        return _is_env_failure(text)

    def test_command_not_found(self):
        assert self._check("bash: pytest: command not found") is True

    def test_pytest_not_recognized_windows(self):
        assert self._check("'pytest' is not recognized as an internal or external command") is True

    def test_pytest_not_recognized_powershell(self):
        assert self._check("The term 'pytest' is not recognized as the name of a cmdlet") is True

    def test_no_module_named(self):
        assert self._check("ModuleNotFoundError: No module named 'fastapi'") is True

    def test_module_not_found_error(self):
        assert self._check("ModuleNotFoundError: No module named 'uvicorn'") is True

    def test_import_error(self):
        assert self._check("ImportError: cannot import name 'Depends' from 'fastapi'") is True

    def test_cannot_find_node_module(self):
        assert self._check("Error: Cannot find module 'express'") is True

    def test_externally_managed_environment(self):
        assert self._check("error: externally-managed-environment") is True

    def test_python_not_found(self):
        assert self._check("python3: command not found") is True

    def test_node_not_found(self):
        assert self._check("node: not found") is True

    def test_assertion_error_is_not_env_failure(self):
        assert self._check("AssertionError: assert 1 == 2") is False

    def test_test_failure_is_not_env_failure(self):
        assert self._check("FAILED tests/test_api.py::test_health - assert response.status_code == 200") is False

    def test_empty_string_is_not_env_failure(self):
        assert self._check("") is False

    def test_pass_output_is_not_env_failure(self):
        assert self._check("5 passed in 1.23s") is False


# ---------------------------------------------------------------------------
# 3. _is_test_pass detection
# ---------------------------------------------------------------------------

class TestIsTestPass:
    def _check(self, text: str) -> bool:
        from app.core.nodes import _is_test_pass
        return _is_test_pass(text)

    def test_pytest_passed(self):
        assert self._check("5 passed in 1.23s") is True

    def test_all_tests_passed(self):
        assert self._check("All tests passed") is True

    def test_tests_passed(self):
        assert self._check("Tests passed") is True

    def test_fail_output_is_not_pass(self):
        assert self._check("1 failed, 5 passed") is False

    def test_empty_is_not_pass(self):
        assert self._check("") is False


# ---------------------------------------------------------------------------
# 4. _classify_test_output
# ---------------------------------------------------------------------------

class TestClassifyTestOutput:
    def _classify(self, text: str) -> str:
        from app.core.nodes import _classify_test_output
        return _classify_test_output(text)

    def test_env_failure_wins_over_pass(self):
        # Env failure takes priority
        assert self._classify("ModuleNotFoundError: No module named 'x'\n5 passed") == "env_failure"

    def test_pass_classified_correctly(self):
        assert self._classify("5 passed in 0.5s") == "pass"

    def test_test_failure_classified_correctly(self):
        assert self._classify("1 failed, 3 passed\nAssertionError: assert 1 == 2") == "test_failure"

    def test_empty_is_test_failure(self):
        # Unknown output → test_failure (conservative)
        assert self._classify("") == "test_failure"

    def test_command_not_found_is_env_failure(self):
        assert self._classify("pytest: command not found") == "env_failure"


# ---------------------------------------------------------------------------
# 5. tester_node env_failure path
# ---------------------------------------------------------------------------

class TestTesterNodeEnvFailure:
    def _run(self, llm_result: str, env_fix_attempts: int = 0) -> dict:
        from app.core.nodes import tester_node
        from app.core.events import clear_listeners

        item = _make_item()
        state = _make_state(
            todo_items=[item],
            current_item_index=0,
            env_fix_attempts=env_fix_attempts,
            active_coder="coder_a",
        )

        with patch("app.core.nodes._invoke_agent", return_value=llm_result), \
             patch("app.core.nodes._format_intelligence_summary_tester", return_value=""):
            result = tester_node(state)
        clear_listeners()
        return result

    def test_env_failure_sets_env_fixing_phase(self):
        result = self._run("ModuleNotFoundError: No module named 'fastapi'")
        assert result["phase"] == WorkflowPhase.ENV_FIXING

    def test_env_failure_does_not_stop_workflow(self):
        result = self._run("pytest: command not found")
        assert result.get("stop_reason") != "env_setup_failed"

    def test_env_failure_stores_last_test_result(self):
        output = "ModuleNotFoundError: No module named 'uvicorn'"
        result = self._run(output)
        assert result["last_test_result"] == output

    def test_env_failure_at_cap_stops_workflow(self):
        from app.core.nodes import _MAX_ENV_FIX_ATTEMPTS
        result = self._run(
            "pytest: command not found",
            env_fix_attempts=_MAX_ENV_FIX_ATTEMPTS,
        )
        assert result["phase"] == WorkflowPhase.STOPPED
        assert result["stop_reason"] == "env_setup_failed"

    def test_env_failure_below_cap_does_not_stop(self):
        from app.core.nodes import _MAX_ENV_FIX_ATTEMPTS
        result = self._run(
            "pytest: command not found",
            env_fix_attempts=_MAX_ENV_FIX_ATTEMPTS - 1,
        )
        assert result["phase"] == WorkflowPhase.ENV_FIXING

    def test_pass_resets_env_fix_attempts(self):
        result = self._run("5 passed in 0.5s\nVerdict: PASS")
        assert result.get("env_fix_attempts") == 0

    def test_test_failure_routes_to_coding(self):
        result = self._run("1 failed\nVerdict: FAIL\nAssertionError")
        assert result["phase"] == WorkflowPhase.CODING

    def test_test_pass_routes_to_deciding(self):
        result = self._run("5 passed\nVerdict: PASS")
        assert result["phase"] == WorkflowPhase.DECIDING


# ---------------------------------------------------------------------------
# 6. planner_env_fix_node
# ---------------------------------------------------------------------------

class TestPlannerEnvFixNode:
    def _run(self, last_test_result: str = "ModuleNotFoundError: No module named 'X'",
             env_fix_attempts: int = 0,
             items: list[TodoItem] | None = None,
             current_index: int = 0) -> dict:
        from app.core.nodes import planner_env_fix_node
        from app.core.events import clear_listeners

        if items is None:
            items = [_make_item(id="item_0", description="Add feature")]

        state = _make_state(
            phase=WorkflowPhase.ENV_FIXING,
            todo_items=items,
            current_item_index=current_index,
            last_test_result=last_test_result,
            env_fix_attempts=env_fix_attempts,
            active_coder="coder_a",
            repo_facts={"tech_stack": {"language": "python", "package_manager": "pip"}},
        )

        llm_json = '{"description": "Install missing package X", "command": "pip install X", "reason": "ModuleNotFoundError"}'
        with patch("app.core.nodes._invoke_agent", return_value=llm_json):
            result = planner_env_fix_node(state)
        clear_listeners()
        return result

    def test_returns_updated_todo_items(self):
        result = self._run()
        assert "todo_items" in result
        assert len(result["todo_items"]) == 2  # original + fix item

    def test_fix_item_prepended_before_current(self):
        result = self._run()
        fix_item = result["todo_items"][0]
        assert fix_item.id.startswith("env_fix_")

    def test_fix_item_task_type_ops(self):
        result = self._run()
        fix_item = result["todo_items"][0]
        assert fix_item.task_type == "ops"

    def test_current_item_index_unchanged(self):
        result = self._run(current_index=0)
        # index still points to position 0 which is now the fix item
        assert result["current_item_index"] == 0

    def test_env_fix_attempts_incremented(self):
        result = self._run(env_fix_attempts=0)
        assert result["env_fix_attempts"] == 1

    def test_phase_set_to_coding(self):
        result = self._run()
        assert result["phase"] == WorkflowPhase.CODING

    def test_peer_review_pre_approved(self):
        # Env-fix items skip peer review — planner pre-approves
        result = self._run()
        assert result["peer_review_verdict"] == "APPROVE"

    def test_fix_item_description_from_llm(self):
        result = self._run()
        fix_item = result["todo_items"][0]
        assert "Install" in fix_item.description or len(fix_item.description) > 0

    def test_multi_item_plan_fix_inserted_correctly(self):
        items = [
            _make_item(id="item_0", description="Feature A"),
            _make_item(id="item_1", description="Feature B"),
        ]
        result = self._run(items=items, current_index=1)
        # Fix item should be at index 1, Feature B at index 2
        assert result["todo_items"][1].id.startswith("env_fix_")
        assert result["todo_items"][2].id == "item_1"


# ---------------------------------------------------------------------------
# 7. Orchestrator routing
# ---------------------------------------------------------------------------

class TestEnvFixOrchestration:
    @pytest.fixture(autouse=True)
    def _reset_shutdown(self):
        from app.core.orchestrator import reset_shutdown
        reset_shutdown()
        yield
        reset_shutdown()

    def test_tester_routes_to_env_fix_on_env_fixing_phase(self):
        from app.core.orchestrator import _route_after_tester
        state = _make_state(phase=WorkflowPhase.ENV_FIXING)
        assert _route_after_tester(state) == "env_fix"

    def test_tester_routes_to_decide_on_deciding_phase(self):
        from app.core.orchestrator import _route_after_tester
        state = _make_state(phase=WorkflowPhase.DECIDING)
        assert _route_after_tester(state) == "decide"

    def test_tester_routes_to_coder_on_coding_phase(self):
        from app.core.orchestrator import _route_after_tester
        state = _make_state(phase=WorkflowPhase.CODING)
        assert _route_after_tester(state) == "coder"

    def test_tester_routes_to_stopped_on_stopped_phase(self):
        from app.core.orchestrator import _route_after_tester
        state = _make_state(phase=WorkflowPhase.STOPPED)
        assert _route_after_tester(state) == "stopped"

    def test_env_fix_routes_to_coder(self):
        from app.core.orchestrator import _route_after_env_fix
        state = _make_state(phase=WorkflowPhase.CODING)
        assert _route_after_env_fix(state) == "coder"

    def test_env_fix_routes_to_stopped_on_shutdown(self):
        from app.core.orchestrator import _route_after_env_fix, request_shutdown
        request_shutdown()
        state = _make_state(phase=WorkflowPhase.CODING)
        assert _route_after_env_fix(state) == "stopped"

    def test_coder_skips_peer_review_for_env_fix_item(self):
        from app.core.orchestrator import _route_after_coder
        fix_item = _make_item(id="env_fix_1", task_type="ops")
        state = _make_state(
            phase=WorkflowPhase.CODING,
            todo_items=[fix_item],
            current_item_index=0,
        )
        assert _route_after_coder(state) == "tester"

    def test_coder_uses_peer_review_for_normal_items(self):
        from app.core.orchestrator import _route_after_coder
        normal_item = _make_item(id="item_0", task_type="coding")
        state = _make_state(
            phase=WorkflowPhase.CODING,
            todo_items=[normal_item],
            current_item_index=0,
        )
        assert _route_after_coder(state) == "peer_review"

    def test_graph_contains_env_fix_node(self):
        from app.core.orchestrator import build_graph
        graph = build_graph()
        assert "env_fix" in graph.nodes

    def test_graph_compiles_with_env_fix(self):
        from app.core.orchestrator import compile_graph
        compiled = compile_graph()
        assert compiled is not None

    def test_resume_routes_to_env_fix_on_env_fixing_phase(self):
        from app.core.orchestrator import _route_after_resume
        state = _make_state(phase=WorkflowPhase.ENV_FIXING)
        assert _route_after_resume(state) == "env_fix"
