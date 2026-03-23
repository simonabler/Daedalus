"""Tester node — test execution and result classification."""
from __future__ import annotations

import platform
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import load_system_prompt
from app.core.config import get_settings
from app.core.events import (
    emit,
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_status,
    emit_verdict,
)
from app.core.logging import get_logger
from app.core.state import GraphState, ItemStatus, WorkflowPhase
from app.core.task_routing import record_agent_outcome
from app.core.token_budget import BudgetExceededException

from ._helpers import (
    TESTER_TOOLS,
    _coder_label,
    _invoke_with_budget,
    _os_note,
    _progress_meta,
    _save_checkpoint_snapshot,
)
from ._context_format import _format_intelligence_summary_tester

logger = get_logger("core.nodes.tester")

_ENV_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"'pytest'\s+is not recognized", re.IGNORECASE),
    re.compile(r'The term ["\']pytest["\'] is not recognized', re.IGNORECASE),
    re.compile(r"No module named\s+\S+", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"cannot import name", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),
    re.compile(r"error: externally-managed-environment", re.IGNORECASE),
    re.compile(r"Could not find.*executable", re.IGNORECASE),
    re.compile(r"No such file or directory.*python", re.IGNORECASE),
    re.compile(r"python.*not found", re.IGNORECASE),
    re.compile(r"node.*not found", re.IGNORECASE),
]

_TEST_PASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d+ passed", re.IGNORECASE),
    re.compile(r"All tests passed", re.IGNORECASE),
    re.compile(r"Tests passed", re.IGNORECASE),
    re.compile(r"OK\s*$", re.MULTILINE),
]

_MAX_ENV_FIX_ATTEMPTS = 2

def _is_env_failure(output: str) -> bool:
    """Return True if the output indicates a missing/broken test environment."""
    return any(pat.search(output) for pat in _ENV_FAILURE_PATTERNS)


def _is_test_pass(output: str) -> bool:
    """Return True if the output clearly indicates all tests passed (no failures)."""
    has_failure = re.search(r"\d+\s+failed", output, re.IGNORECASE)
    if has_failure:
        return False
    return any(pat.search(output) for pat in _TEST_PASS_PATTERNS)


def _classify_test_output(output: str) -> str:
    """Classify raw test runner output as 'pass', 'env_failure', or 'test_failure'."""
    if _is_env_failure(output):
        return "env_failure"
    if _is_test_pass(output):
        return "pass"
    return "test_failure"


def tester_node(state: GraphState) -> dict:
    """Run tests and verification.

    Classifies the test runner output into three categories:
    - pass        → advance to decide/human-gate
    - test_failure → route back to coder for a fix
    - env_failure  → route to planner_env_fix (never handled here)

    The tester's job is to write, run, and analyse tests.
    It never modifies the environment — that is the planner's responsibility.
    """
    item = state.current_item
    if not item:
        emit_error("system", "No item to test")
        return {"error_message": "No item to test", "phase": WorkflowPhase.STOPPED}

    emit_node_start("tester", "Testing", item_id=item.id, item_desc=item.description)
    emit_status("tester", f"🧪 Running tests for: {item.description}", **_progress_meta(state, "testing"))

    item.status = ItemStatus.TESTING

    prompt = (
        f"## Verification Task\n\n"
        f"**Item**: {item.id} — {item.description}\n"
        f"**Execution platform**: {state.execution_platform or platform.platform()}\n"
        f"**OS Note**: {_os_note(state.execution_platform or platform.platform())}\n"
    )
    if item.acceptance_criteria:
        prompt += "**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria) + "\n"
    if item.verification_commands:
        prompt += "**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands) + "\n"
    intelligence_ctx = _format_intelligence_summary_tester(state)
    if intelligence_ctx:
        prompt += f"\n\n{intelligence_ctx}"
    prompt += (
        "\nRun all tests, linters, and verification commands. "
        "Produce a structured test report with PASS or FAIL verdict. "
        "If the test runner itself cannot start (missing interpreter, missing module, "
        "command not found), report that clearly — do NOT attempt to install packages."
    )

    try:
        result, budget_update = _invoke_with_budget(
            state, "tester", [HumanMessage(content=prompt)], TESTER_TOOLS, node="tester",
        )
    except BudgetExceededException:
        return {"phase": WorkflowPhase.STOPPED, "stop_reason": "budget_hard_limit_exceeded"}

    # -- Classify the raw LLM output ----------------------------------------
    if "**Verdict**: PASS" in result or "Verdict: PASS" in result:
        llm_verdict = "PASS"
    elif "**Verdict**: FAIL" in result or "Verdict: FAIL" in result:
        llm_verdict = "FAIL"
    elif "PASS" in result.upper() and "FAIL" not in result.upper():
        llm_verdict = "PASS"
    else:
        llm_verdict = "FAIL"

    # Override: if output signals env failure, trust that over LLM verdict
    classification = _classify_test_output(result)

    item.test_report = result

    # -- ENV FAILURE: hand off to planner_env_fix ---------------------------
    if classification == "env_failure":
        if state.env_fix_attempts >= _MAX_ENV_FIX_ATTEMPTS:
            msg = (
                f"Environment setup failed after {state.env_fix_attempts} fix attempt(s). "
                "Human intervention required."
            )
            emit_status("tester", f"❌ {msg}", **_progress_meta(state, "stopped"))
            emit_node_end("tester", "Testing", msg)
            return {
                "last_test_result": result,
                "stop_reason": "env_setup_failed",
                "phase": WorkflowPhase.STOPPED,
                **budget_update,
            }

        emit_status(
            "tester",
            "⚠️ Tester: missing dependency or broken environment detected "
            "— handing to planner for auto-fix",
            **_progress_meta(state, "env_fixing"),
        )
        emit_node_end("tester", "Testing", "ENV_FAILURE — routing to planner_env_fix")
        return {
            "last_test_result": result,
            "phase": WorkflowPhase.ENV_FIXING,
            **budget_update,
        }

    # -- Normal PASS / FAIL -------------------------------------------------
    emit_verdict("tester", llm_verdict, detail=result, item_id=item.id)

    if llm_verdict == "FAIL":
        item.test_fail_count += 1
        item.status = ItemStatus.IN_PROGRESS
        record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=False)
        if item.test_fail_count >= get_settings().max_rework_cycles_per_item:
            msg = f"Item {item.id} failed tests {item.test_fail_count} times; stopping to avoid loop."
            emit_error("tester", msg)
            return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED, **budget_update}
        emit_status(
            "tester",
            f"❌ Tests FAILED - sending back to {_coder_label(state.active_coder)}",
            **_progress_meta(state, "coding"),
        )
    else:
        emit_status("tester", "✅ All tests PASSED", **_progress_meta(state, "deciding"))

    emit_node_end("tester", "Testing", f"Verdict: {llm_verdict}")

    updates = {
        "last_test_result": result,
        "env_fix_attempts": 0,  # reset on successful test run (pass or genuine fail)
        "phase": WorkflowPhase.DECIDING if llm_verdict == "PASS" else WorkflowPhase.CODING,
        **budget_update,
    }
    if llm_verdict == "PASS":
        _save_checkpoint_snapshot(state, updates, "test_pass")
    return updates


# ---------------------------------------------------------------------------
# NODE: planner_env_fix
# ---------------------------------------------------------------------------

