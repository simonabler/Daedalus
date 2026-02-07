"""LangGraph node implementations for the dual-coder orchestrator workflow.

New workflow:
  Planner → Coder (A|B) → Peer Review (B|A) → Planner Review → Tester → Decide → Commit

Coder assignment alternates per TODO item:
  - Even items (0, 2, 4…): Coder A implements, Reviewer B (gpt-5.2) reviews
  - Odd  items (1, 3, 5…): Coder B implements, Reviewer A (Claude)  reviews

On REWORK from peer review: the original coder re-implements (not the reviewer).
On REWORK from planner review or test failure: same — original coder fixes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase
from app.tools.filesystem import read_file, write_file, list_directory
from app.tools.git import git_command, git_create_branch, git_commit_and_push, git_status
from app.tools.shell import run_shell
from app.tools.build import run_tests, run_linter

logger = get_logger("core.nodes")


# ── Helper: invoke LLM with tools ────────────────────────────────────────

def _invoke_agent(role: str, messages: list, tools: list | None = None) -> str:
    """Invoke an LLM agent and handle tool calls in a loop."""
    llm = get_llm(role)
    system_prompt = load_system_prompt(role)

    all_messages = [SystemMessage(content=system_prompt)] + messages

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    max_tool_rounds = 15
    for _ in range(max_tool_rounds):
        response = llm_with_tools.invoke(all_messages)
        all_messages.append(response)

        if not response.tool_calls:
            return response.content if isinstance(response.content, str) else str(response.content)

        tool_map = {t.name: t for t in (tools or [])}
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                logger.info("tool_call  | %s(%s) -> %d chars", tc["name"], list(tc["args"].keys()), len(str(result)))
            else:
                result = f"Unknown tool: {tc['name']}"

            all_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "ERROR: Exceeded maximum tool call rounds."


# ── Tool sets ─────────────────────────────────────────────────────────────

PLANNER_TOOLS = [read_file, write_file, list_directory, git_status, run_shell]

CODER_TOOLS = [
    read_file, write_file, list_directory,
    run_shell, git_status, git_command,
    run_tests, run_linter,
]

# Peer reviewers get read-only tools + git diff (no write access)
REVIEWER_TOOLS = [read_file, list_directory, run_shell, git_status, git_command, run_tests, run_linter]

TESTER_TOOLS = [read_file, list_directory, run_shell, run_tests, run_linter, git_status]


# ── Helper: determine coder/reviewer pair for an item ────────────────────

def _assign_coder_pair(item_index: int) -> tuple[str, str]:
    """Return (active_coder, active_reviewer) for the given item index.

    Even items → coder_a implements, reviewer_b reviews.
    Odd  items → coder_b implements, reviewer_a reviews.
    """
    if item_index % 2 == 0:
        return ("coder_a", "reviewer_b")
    else:
        return ("coder_b", "reviewer_a")


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_plan
# ═══════════════════════════════════════════════════════════════════════════

def planner_plan_node(state: GraphState) -> dict:
    """Planner creates/updates the plan in tasks/todo.md."""
    logger.info("NODE: planner_plan | request: %s", state.user_request[:100])

    context_parts = [
        f"User request: {state.user_request}",
        f"Repository root: {state.repo_root}",
        f"Current branch: {state.branch_name or 'not set'}",
        "",
        "NOTE: This system uses two coders (Coder A = Claude, Coder B = gpt-5.2).",
        "They alternate: even-numbered items go to Coder A, odd to Coder B.",
        "Each coder's work is peer-reviewed by the other before testing.",
    ]

    try:
        lessons = read_file.invoke({"path": "tasks/lessons.md"})
        if not lessons.startswith("ERROR"):
            context_parts.append(f"Lessons learned:\n{lessons}")
    except Exception:
        pass

    try:
        todo = read_file.invoke({"path": "tasks/todo.md"})
        if not todo.startswith("ERROR"):
            context_parts.append(f"Current todo.md:\n{todo}")
    except Exception:
        pass

    prompt = (
        "Analyze the request and create a detailed plan.\n\n"
        + "\n\n".join(context_parts)
        + "\n\nInstructions:\n"
        "1. Use the list_directory tool to understand the project structure.\n"
        "2. Read key files (README, config, etc.) to understand the codebase.\n"
        "3. Create a detailed plan with checkboxes in tasks/todo.md.\n"
        "4. Each item needs: description, acceptance criteria, verification commands.\n"
        "5. Create a feature branch if not already on one.\n"
        "6. Return the plan summary and the ID of the first item to work on."
    )

    result = _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)

    items = _parse_plan_from_result(result)

    branch = state.branch_name
    if not branch or branch == "main" or branch == "master":
        slug = state.user_request[:30].lower().replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        branch = f"feature/{date}-{slug}"
        git_create_branch.invoke({"branch_name": branch})

    # Assign first coder pair
    coder, reviewer = _assign_coder_pair(0)

    return {
        "todo_items": items if items else state.todo_items,
        "current_item_index": 0 if items else state.current_item_index,
        "branch_name": branch,
        "phase": WorkflowPhase.CODING,
        "needs_replan": False,
        "active_coder": coder,
        "active_reviewer": reviewer,
    }


def _parse_plan_from_result(result: str) -> list[TodoItem]:
    """Extract TODO items from planner output. Best-effort parsing."""
    items = []
    lines = result.split("\n")
    current_id = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
            current_id += 1
            desc = stripped.replace("- [ ]", "").replace("- [x]", "").strip()
            if desc and desc[0].isdigit():
                parts = desc.split(":", 1)
                if len(parts) > 1:
                    desc = parts[1].strip()

            items.append(TodoItem(
                id=f"item-{current_id:03d}",
                description=desc or f"Task {current_id}",
                status=ItemStatus.PENDING,
            ))

    if not items:
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:4]:
                current_id += 1
                desc = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
                items.append(TodoItem(
                    id=f"item-{current_id:03d}",
                    description=desc,
                    status=ItemStatus.PENDING,
                ))

    return items


# ═══════════════════════════════════════════════════════════════════════════
# NODE: coder (dispatches to coder_a or coder_b based on state)
# ═══════════════════════════════════════════════════════════════════════════

def coder_node(state: GraphState) -> dict:
    """Dispatch coding to the active coder (A or B, alternating per item)."""
    item = state.current_item
    if not item:
        return {"error_message": "No current item to work on", "phase": WorkflowPhase.STOPPED}

    active = state.active_coder  # "coder_a" or "coder_b"
    logger.info("NODE: coder [%s] | item: %s — %s", active, item.id, item.description[:80])

    item.status = ItemStatus.IN_PROGRESS
    item.iteration_count += 1

    settings = get_settings()
    if item.iteration_count > settings.max_iterations_per_item:
        return {
            "stop_reason": f"Item {item.id} exceeded max iterations ({settings.max_iterations_per_item})",
            "phase": WorkflowPhase.STOPPED,
        }

    # Build prompt
    coder_label = "Coder A (Claude)" if active == "coder_a" else "Coder B (gpt-5.2)"
    reviewer_label = "Coder B (gpt-5.2)" if active == "coder_a" else "Coder A (Claude)"

    prompt_parts = [
        f"## Task Assignment — {coder_label}",
        f"**Item ID**: {item.id}",
        f"**Description**: {item.description}",
        f"**Your peer reviewer**: {reviewer_label} (they will review your work next)",
    ]
    if item.acceptance_criteria:
        prompt_parts.append("**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria))
    if item.verification_commands:
        prompt_parts.append("**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands))
    if item.review_notes:
        prompt_parts.append(f"**Rework Notes (from previous review)**:\n{item.review_notes}")
    if state.peer_review_notes and state.peer_review_verdict == "REWORK":
        prompt_parts.append(f"**Peer Review Feedback (REWORK requested)**:\n{state.peer_review_notes}")
    if item.test_report:
        prompt_parts.append(f"**Test Report (from previous round)**:\n{item.test_report}")

    prompt_parts.append(
        "\nImplement this task. Use tools to read the codebase, make changes, "
        "add tests, update docs. Keep diffs minimal. Follow Clean Architecture."
    )

    result = _invoke_agent(active, [HumanMessage(content="\n\n".join(prompt_parts))], CODER_TOOLS)

    return {
        "last_coder_result": result,
        "phase": WorkflowPhase.PEER_REVIEWING,
        "total_iterations": state.total_iterations + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: peer_review (the OTHER coder reviews)
# ═══════════════════════════════════════════════════════════════════════════

def peer_review_node(state: GraphState) -> dict:
    """The peer reviewer (opposite coder) reviews the implementation.

    If coder_a implemented → reviewer_b (gpt-5.2) reviews.
    If coder_b implemented → reviewer_a (Claude) reviews.
    """
    item = state.current_item
    if not item:
        return {"error_message": "No item to peer-review", "phase": WorkflowPhase.STOPPED}

    reviewer = state.active_reviewer  # "reviewer_a" or "reviewer_b"
    implementer = state.active_coder

    implementer_label = "Coder A (Claude)" if implementer == "coder_a" else "Coder B (gpt-5.2)"
    reviewer_label = "Reviewer A (Claude)" if reviewer == "reviewer_a" else "Reviewer B (gpt-5.2)"

    logger.info(
        "NODE: peer_review [%s reviewing %s's work] | item: %s",
        reviewer_label, implementer_label, item.id,
    )
    item.status = ItemStatus.IN_REVIEW

    prompt = (
        f"## Peer Code Review\n\n"
        f"**Reviewer**: {reviewer_label}\n"
        f"**Implementer**: {implementer_label}\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implementer's Report**:\n{state.last_coder_result}\n\n"
        f"Review the changes:\n"
        f"1. Use `git_command` with `git diff` to see the actual changes.\n"
        f"2. Use `git_command` with `git status` to see which files changed.\n"
        f"3. Read the modified files to understand the full context.\n"
        f"4. Run the test suite and linter to check for regressions.\n"
        f"5. Verify: correct logic, minimal diff, clean architecture, tests added, docs updated.\n"
        f"6. Give your verdict: APPROVE or REWORK.\n"
        f"7. If REWORK, provide specific actionable notes.\n"
        f"8. If APPROVE, suggest a Conventional Commit message."
    )

    result = _invoke_agent(reviewer, [HumanMessage(content=prompt)], REVIEWER_TOOLS)

    # Parse verdict
    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"
    if "**Verdict**: REWORK" in result or "Verdict: REWORK" in result:
        verdict = "REWORK"
    elif "**Verdict**: APPROVE" in result or "Verdict: APPROVE" in result:
        verdict = "APPROVE"

    if verdict == "REWORK":
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS

    return {
        "peer_review_verdict": verdict,
        "peer_review_notes": result,
        # On APPROVE → proceed to planner review. On REWORK → back to coder.
        "phase": WorkflowPhase.REVIEWING if verdict == "APPROVE" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_review (final gate before testing)
# ═══════════════════════════════════════════════════════════════════════════

def planner_review_node(state: GraphState) -> dict:
    """Planner does a final review after peer review passes."""
    item = state.current_item
    if not item:
        return {"error_message": "No item to review", "phase": WorkflowPhase.STOPPED}

    logger.info("NODE: planner_review | item: %s", item.id)

    implementer_label = "Coder A (Claude)" if state.active_coder == "coder_a" else "Coder B (gpt-5.2)"
    reviewer_label = "Reviewer B (gpt-5.2)" if state.active_reviewer == "reviewer_b" else "Reviewer A (Claude)"

    prompt = (
        f"## Planner Final Review\n\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implemented by**: {implementer_label}\n"
        f"**Peer-reviewed by**: {reviewer_label} — APPROVED\n\n"
        f"**Coder's Report**:\n{state.last_coder_result}\n\n"
        f"**Peer Review Notes**:\n{state.peer_review_notes}\n\n"
        f"Final review:\n"
        f"1. Use `git status` and `git diff` to verify the changes.\n"
        f"2. Confirm the peer review didn't miss anything.\n"
        f"3. Verify: minimal diff, clean architecture, tests present, docs updated.\n"
        f"4. Give your verdict: APPROVE or REWORK.\n"
        f"5. If APPROVE, confirm or adjust the suggested Conventional Commit message."
    )

    result = _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)

    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"

    if verdict == "REWORK":
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS

    return {
        "last_review_verdict": verdict,
        "review_notes": result,
        "phase": WorkflowPhase.TESTING if verdict == "APPROVE" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: tester
# ═══════════════════════════════════════════════════════════════════════════

def tester_node(state: GraphState) -> dict:
    """Tester runs verification checks."""
    item = state.current_item
    if not item:
        return {"error_message": "No item to test", "phase": WorkflowPhase.STOPPED}

    logger.info("NODE: tester | item: %s", item.id)
    item.status = ItemStatus.TESTING

    prompt = (
        f"## Verification Task\n\n"
        f"**Item**: {item.id} — {item.description}\n"
    )
    if item.acceptance_criteria:
        prompt += "**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria) + "\n"
    if item.verification_commands:
        prompt += "**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands) + "\n"

    prompt += (
        "\nRun all tests, linters, and verification commands. "
        "Produce a structured test report with PASS or FAIL verdict."
    )

    result = _invoke_agent("tester", [HumanMessage(content=prompt)], TESTER_TOOLS)

    verdict = "PASS" if "PASS" in result.upper() and "FAIL" not in result.upper() else "FAIL"
    if "**Verdict**: PASS" in result or "Verdict: PASS" in result:
        verdict = "PASS"
    elif "**Verdict**: FAIL" in result or "Verdict: FAIL" in result:
        verdict = "FAIL"

    item.test_report = result

    if verdict == "FAIL":
        item.status = ItemStatus.IN_PROGRESS

    return {
        "last_test_result": result,
        "phase": WorkflowPhase.DECIDING if verdict == "PASS" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_decide
# ═══════════════════════════════════════════════════════════════════════════

def planner_decide_node(state: GraphState) -> dict:
    """Planner decides: commit this item and move to next, or stop."""
    item = state.current_item
    if not item:
        return {"error_message": "No item to decide on", "phase": WorkflowPhase.STOPPED}

    logger.info("NODE: planner_decide | item: %s", item.id)

    item.status = ItemStatus.DONE

    # Try to extract commit message from reviews (peer first, then planner)
    commit_msg = _extract_commit_message(state.peer_review_notes, state.review_notes, item.description)
    item.commit_message = commit_msg

    # Update tasks/todo.md
    try:
        todo_content = read_file.invoke({"path": "tasks/todo.md"})
        if not todo_content.startswith("ERROR"):
            updated = todo_content.replace(f"- [ ] {item.description}", f"- [x] {item.description}")
            write_file.invoke({"path": "tasks/todo.md", "content": updated})
    except Exception as e:
        logger.warning("Could not update todo.md: %s", e)

    return {
        "phase": WorkflowPhase.COMMITTING,
        "completed_items": state.completed_items + 1,
    }


def _extract_commit_message(peer_notes: str, planner_notes: str, fallback_desc: str) -> str:
    """Try to extract a conventional commit message from review notes."""
    for source in [planner_notes, peer_notes]:
        for line in source.split("\n"):
            stripped = line.strip()
            for prefix in ["Suggested commit:", "Commit message:", "Commit:", "Suggested Conventional Commit message:"]:
                if prefix.lower() in stripped.lower():
                    stripped = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
                    break
            stripped = stripped.strip("`").strip('"').strip("'").strip()
            if any(stripped.startswith(p) for p in ["feat(", "fix(", "docs:", "test:", "refactor(", "chore("]):
                return stripped
    slug = fallback_desc[:50].lower()
    return f"feat: {slug}"


# ═══════════════════════════════════════════════════════════════════════════
# NODE: committer
# ═══════════════════════════════════════════════════════════════════════════

def committer_node(state: GraphState) -> dict:
    """Commit and push the completed item, then advance to next item."""
    item = state.current_item
    if not item:
        return {"error_message": "No item to commit", "phase": WorkflowPhase.STOPPED}

    logger.info("NODE: committer | %s | msg: %s", item.id, item.commit_message)

    result = git_commit_and_push.invoke({
        "message": item.commit_message,
        "push": True,
    })

    logger.info("commit result: %s", result[:200])

    # Move to next item and alternate coders
    next_index = state.current_item_index + 1
    has_more = next_index < len(state.todo_items)

    if has_more:
        next_coder, next_reviewer = _assign_coder_pair(next_index)
        return {
            "current_item_index": next_index,
            "phase": WorkflowPhase.CODING,
            "active_coder": next_coder,
            "active_reviewer": next_reviewer,
            "peer_review_notes": "",
            "peer_review_verdict": "",
        }
    else:
        return {
            "phase": WorkflowPhase.COMPLETE,
        }
