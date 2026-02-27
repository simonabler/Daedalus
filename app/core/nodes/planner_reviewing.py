"""Planner review/decision nodes."""

from __future__ import annotations

from ._helpers import *
from ._streaming import *
from ._prompt_enrichment import *

def planner_review_node(state: GraphState) -> dict:
    """Planner final review gate."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to review")
        return {"error_message": "No item to review", "phase": WorkflowPhase.STOPPED}

    emit_node_start("planner", "Final Review", item_id=item.id, item_desc=item.description)
    emit_status("planner", f"🎯 Planner final review: {item.description}", **_progress_meta(state, "reviewing"))

    impl_label = _coder_label(state.active_coder)
    rev_label = _reviewer_label(state.active_reviewer)

    prompt = (
        f"## Planner Final Review\n\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implemented by**: {impl_label}\n"
        f"**Peer-reviewed by**: {rev_label} — APPROVED\n\n"
        f"**Coder's Report**:\n{state.last_coder_result}\n\n"
        f"**Peer Review Notes**:\n{state.peer_review_notes}\n\n"
        f"Final review:\n"
        f"1. Use `git status` and `git diff` to verify the changes.\n"
        f"2. Confirm the peer review didn't miss anything.\n"
        f"3. Verify: minimal diff, clean architecture, tests present, docs updated.\n"
        f"4. Give your verdict: APPROVE or REWORK.\n"
        f"5. If APPROVE, confirm or adjust the suggested Conventional Commit message."
    )

    result, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=PLANNER_TOOLS, inject_memory=False, node="planner",
    )
    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"

    emit_verdict("planner", verdict, detail=result, item_id=item.id)

    if verdict == "REWORK":
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS
        if item.rework_count >= get_settings().max_rework_cycles_per_item:
            emit_status(
                "planner",
                "🔄 Planner forcing objective tester gate after repeated rework cycles",
                **_progress_meta(state, "testing"),
            )
            phase = WorkflowPhase.TESTING
        else:
            emit_status("planner", f"Planner REWORK - sending back to {impl_label}", **_progress_meta(state, "coding"))
            phase = WorkflowPhase.CODING
    else:
        emit_status("planner", "✅ Planner APPROVED - sending to Tester", **_progress_meta(state, "testing"))
        phase = WorkflowPhase.TESTING

    emit_node_end("planner", "Final Review", f"Verdict: {verdict}")

    return {
        "last_review_verdict": verdict,
        "review_notes": result,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# NODE: tester
# ---------------------------------------------------------------------------

# Patterns in test/runner output that indicate an environment problem,
# not a test failure. The tester detects these and hands off to planner_env_fix.
_ENV_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"'pytest'\s+is not recognized", re.IGNORECASE),
    re.compile(r'The term ["\']pytest["\'] is not recognized', re.IGNORECASE),
    re.compile(r"No module named\s+\S+", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"cannot import name", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),          # Node.js
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

# Maximum env-fix rounds before giving up and routing to human gate
_MAX_ENV_FIX_ATTEMPTS = 2

def planner_env_fix_node(state: GraphState) -> dict:
    """Create a single env-setup fix item and prepend it to the plan.

    Called when tester_node detects an environment failure (missing package,
    command not found, etc.). This node:
    1. Reads the tester's failure output from last_test_result
    2. Uses an LLM to identify the exact missing package/tool
    3. Creates one TodoItem: install the dependency or update config
    4. Prepends it to todo_items so the coder handles it next
    5. Routes to coder (skipping peer review — it's a mechanical fix)

    Does NOT re-plan the entire task. Only creates the minimal fix item.
    """
    emit_node_start("planner", "EnvFix", item_desc="Diagnosing environment failure")
    emit_status(
        "planner",
        "🔧 Planner: diagnosing missing dependency and creating fix item…",
        **_progress_meta(state, "env_fixing"),
    )

    failure_output = state.last_test_result or "(no output captured)"
    repo_facts_summary = ""
    if state.repo_facts:
        rf = state.repo_facts
        lang = rf.get("tech_stack", {}).get("language", "unknown")
        pm = rf.get("tech_stack", {}).get("package_manager", "unknown")
        repo_facts_summary = f"Language: {lang}, Package manager: {pm}"

    prompt = (
        "## Environment Fix Task\n\n"
        "The test runner failed due to a missing dependency or broken environment. "
        "Your job is to create a single fix TodoItem to install or configure what is missing.\n\n"
        f"**Test runner output (failure)**:\n```\n{failure_output[:3000]}\n```\n\n"
        f"**Repository info**: {repo_facts_summary or 'unknown'}\n\n"
        "Instructions:\n"
        "1. Identify the exact missing package, module, or tool from the output above.\n"
        "2. Determine the correct install command (pip install X, npm install X, "
        "poetry add X, update pyproject.toml, etc.).\n"
        "3. Respond with ONLY a JSON object in this exact format:\n"
        "```json\n"
        '{"description": "Install missing dependency X", '
        '"command": "pip install X", '
        '"reason": "ModuleNotFoundError: No module named X"}\n'
        "```\n"
        "Do not add explanation outside the JSON block."
    )

    raw, budget_update = _invoke_with_budget(
        state, "planner", [HumanMessage(content=prompt)],
        tools=None, inject_memory=False, node="env_fix",
    )

    # Parse JSON from LLM response
    fix_description = "Install missing test dependency"
    fix_command = ""
    try:
        import json as _json
        # Extract JSON block from response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = _json.loads(raw[json_start:json_end])
            fix_description = parsed.get("description", fix_description)
            fix_command = parsed.get("command", "")
    except Exception:
        logger.warning("planner_env_fix | could not parse LLM JSON response: %s", raw[:200])

    # Create the fix TodoItem with a unique id
    fix_item = TodoItem(
        id=f"env_fix_{state.env_fix_attempts + 1}",
        description=fix_description,
        task_type="ops",
        acceptance_criteria=["Test environment is functional and tests can run"],
        verification_commands=[fix_command] if fix_command else [],
    )

    # Assign the same coder that was active — env fixes skip peer review
    fix_item.assigned_agent = state.active_coder
    fix_item.assigned_reviewer = ""  # no review for mechanical env fixes

    # Prepend fix item before the current item (so coder handles it next)
    current_idx = state.current_item_index
    new_items = (
        list(state.todo_items[:current_idx])
        + [fix_item]
        + list(state.todo_items[current_idx:])
    )

    emit_status(
        "planner",
        f"🔧 Env fix item created: {fix_description}",
        **_progress_meta(state, "env_fixing"),
    )
    emit_node_end("planner", "EnvFix", f"Fix item queued: {fix_description}")

    return {
        "todo_items": new_items,
        "current_item_index": current_idx,  # points to fix_item now
        "env_fix_attempts": state.env_fix_attempts + 1,
        "phase": WorkflowPhase.CODING,
        # Skip peer review for env-fix items — coder goes straight back to tester
        "peer_review_verdict": "APPROVE",
        "peer_review_notes": "Auto-approved: environment fix item, no peer review needed.",
        **budget_update,
    }

def planner_decide_node(state: GraphState) -> dict:
    """Mark item done, prepare commit."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to decide on")
        return {"error_message": "No item to decide on", "phase": WorkflowPhase.STOPPED}

    item_num = state.current_item_index + 1
    total = len(state.todo_items)

    item.status = ItemStatus.DONE
    commit_msg = _extract_commit_message(state.peer_review_notes, state.review_notes, item.description)
    item.commit_message = commit_msg
    next_done = state.completed_items + 1

    emit_status(
        "planner",
        f"✅ Item {item_num}/{total} DONE: {item.description}",
        **_progress_meta(state, "deciding", done_override=next_done),
    )
    record_agent_outcome(state.repo_root, item.task_type, state.active_coder, success=True)

    try:
        _write_todo_file(state.todo_items, state.user_request)
    except Exception as e:
        logger.warning("Could not update todo.md: %s", e)

    return {
        "phase": WorkflowPhase.COMMITTING,
        "completed_items": next_done,
    }
