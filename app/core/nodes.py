"""LangGraph node implementations — dual-coder workflow with shared memory.

All coder and reviewer nodes receive shared long-term memory before each call.
After each peer review a learning step extracts insights into memory files.
The planner compresses memory at session start if files are too large.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agents.models import get_llm, load_system_prompt
from app.core.config import get_settings
from app.core.events import (
    emit_agent_result,
    emit_agent_thinking,
    emit_commit,
    emit_error,
    emit_node_end,
    emit_node_start,
    emit_plan,
    emit_status,
    emit_tool_call,
    emit_tool_result,
    emit_verdict,
)
from app.core.logging import get_logger
from app.core.memory import (
    LEARNING_EXTRACTION_PROMPT,
    append_memory,
    build_compression_prompt,
    ensure_memory_files,
    get_memory_stats,
    load_all_memory,
    memory_needs_compression,
    save_compressed,
)
from app.core.state import GraphState, ItemStatus, TodoItem, WorkflowPhase
from app.tools.build import run_linter, run_tests
from app.tools.filesystem import list_directory, read_file, write_file
from app.tools.git import git_command, git_commit_and_push, git_create_branch, git_status
from app.tools.shell import run_shell

logger = get_logger("core.nodes")


# ── Helper: invoke LLM with tools + event emission ───────────────────────

def _invoke_agent(role: str, messages: list, tools: list | None = None,
                  inject_memory: bool = False) -> str:
    """Invoke an LLM agent, handle tool calls, emit events.

    If inject_memory=True, the shared long-term memory is prepended to the
    system prompt so the agent can use established conventions.
    """
    llm = get_llm(role)
    system_prompt = load_system_prompt(role)

    # Inject shared memory into system prompt for coders/reviewers
    if inject_memory:
        memory_ctx = load_all_memory()
        if memory_ctx:
            system_prompt = system_prompt + "\n\n" + memory_ctx

    all_messages = [SystemMessage(content=system_prompt)] + messages

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    prompt_summary = messages[-1].content[:300] if messages else ""
    emit_agent_thinking(role, prompt_summary)

    max_tool_rounds = 15
    for round_num in range(max_tool_rounds):
        response = llm_with_tools.invoke(all_messages)
        all_messages.append(response)

        if not response.tool_calls:
            result = response.content if isinstance(response.content, str) else str(response.content)
            emit_agent_result(role, result)
            return result

        tool_map = {t.name: t for t in (tools or [])}
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in tc["args"].items())
            emit_tool_call(role, tc["name"], args_str)

            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                    emit_error(role, f"Tool {tc['name']} failed: {e}")
            else:
                result = f"Unknown tool: {tc['name']}"

            emit_tool_result(role, tc["name"], str(result))
            logger.info("tool_call  | %s(%s) -> %d chars", tc["name"], list(tc["args"].keys()), len(str(result)))
            all_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    emit_error(role, "Exceeded maximum tool call rounds (15)")
    return "ERROR: Exceeded maximum tool call rounds."


# ── Tool sets ─────────────────────────────────────────────────────────────

PLANNER_TOOLS = [read_file, write_file, list_directory, git_status, run_shell]

CODER_TOOLS = [
    read_file, write_file, list_directory,
    run_shell, git_status, git_command,
    run_tests, run_linter,
]

REVIEWER_TOOLS = [read_file, list_directory, run_shell, git_status, git_command, run_tests, run_linter]

TESTER_TOOLS = [read_file, list_directory, run_shell, run_tests, run_linter, git_status]


# ── Helper: coder pair assignment ─────────────────────────────────────────

def _assign_coder_pair(item_index: int) -> tuple[str, str]:
    """Even items → coder_a/reviewer_b. Odd items → coder_b/reviewer_a."""
    if item_index % 2 == 0:
        return ("coder_a", "reviewer_b")
    else:
        return ("coder_b", "reviewer_a")


def _coder_label(role: str) -> str:
    return {"coder_a": "Coder A (Claude)", "coder_b": "Coder B (GPT-5.3)"}.get(role, role)

def _reviewer_label(role: str) -> str:
    return {"reviewer_a": "Reviewer A (Claude)", "reviewer_b": "Reviewer B (GPT-5.3)"}.get(role, role)


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_plan
# ═══════════════════════════════════════════════════════════════════════════

def planner_plan_node(state: GraphState) -> dict:
    """Planner creates the plan. Also ensures memory files exist and compresses if needed."""
    emit_node_start("planner", "Planning", item_desc=state.user_request[:100])
    emit_status("planner", f"📝 Analyzing request: {state.user_request[:80]}…", phase="planning")

    # Ensure memory files exist in the target repo
    try:
        ensure_memory_files()
    except Exception as e:
        logger.warning("Could not ensure memory files: %s", e)

    # Check if memory needs compression (do it early)
    stats = get_memory_stats()
    for key, info in stats.items():
        if info["needs_compression"]:
            emit_status("planner", f"🗜 Compressing memory: {key} ({info['chars']} chars)", phase="planning")
            _compress_memory_file(key)

    # Log memory stats
    total_chars = sum(s["chars"] for s in stats.values())
    if total_chars > 0:
        emit_status("planner", f"🧠 Shared memory loaded: {total_chars} chars across {len(stats)} files", phase="planning")

    context_parts = [
        f"User request: {state.user_request}",
        f"Repository root: {state.repo_root}",
        f"Current branch: {state.branch_name or 'not set'}",
        "",
        "NOTE: This system uses two coders (Coder A = Claude, Coder B = GPT-5.3).",
        "They alternate: even-numbered items go to Coder A, odd to Coder B.",
        "Each coder's work is peer-reviewed by the other before testing.",
        "Both coders share long-term memory files in memory/.",
    ]

    # Load shared memory for planner context
    memory_ctx = load_all_memory()
    if memory_ctx:
        context_parts.append(f"\n{memory_ctx}")

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

    if items:
        plan_text = "## TODO Plan\n"
        for i, item in enumerate(items):
            coder, _ = _assign_coder_pair(i)
            label = _coder_label(coder)
            plan_text += f"  {i+1}. [ ] {item.description}  →  {label}\n"
        emit_plan("planner", plan_text, items_count=len(items))
        emit_status("planner", f"✅ Plan created with {len(items)} items", phase="planning", items_count=len(items))
    else:
        emit_status("planner", "⚠ Could not parse plan items from planner output", phase="planning")

    branch = state.branch_name
    if not branch or branch == "main" or branch == "master":
        slug = state.user_request[:30].lower().replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        branch = f"feature/{date}-{slug}"
        git_create_branch.invoke({"branch_name": branch})
        emit_status("planner", f"🌿 Created branch: {branch}", phase="planning")

    coder, reviewer = _assign_coder_pair(0)
    emit_node_end("planner", "Planning", f"Plan ready — starting with item 1 → {_coder_label(coder)}")

    return {
        "todo_items": items if items else state.todo_items,
        "current_item_index": 0 if items else state.current_item_index,
        "branch_name": branch,
        "phase": WorkflowPhase.CODING,
        "needs_replan": False,
        "active_coder": coder,
        "active_reviewer": reviewer,
    }


def _compress_memory_file(key: str) -> None:
    """Use the planner LLM to compress an oversized memory file."""
    prompt = build_compression_prompt(key)
    if not prompt:
        return
    try:
        result = _invoke_agent("planner", [HumanMessage(content=prompt)])
        # Clean up: strip markdown fences if present
        cleaned = result.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        save_compressed(key, cleaned)
        emit_status("planner", f"✅ Memory compressed: {key}", phase="planning")
    except Exception as e:
        logger.warning("Memory compression failed for %s: %s", key, e)


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
# NODE: coder  (reads shared memory)
# ═══════════════════════════════════════════════════════════════════════════

def coder_node(state: GraphState) -> dict:
    """Dispatch to active coder. Injects shared memory into context."""
    item = state.current_item
    if not item:
        emit_error("system", "No current item to work on")
        return {"error_message": "No current item to work on", "phase": WorkflowPhase.STOPPED}

    active = state.active_coder
    reviewer = state.active_reviewer
    item_num = state.current_item_index + 1
    total = len(state.todo_items)

    emit_node_start(active, "Coding", item_id=item.id, item_desc=item.description)
    emit_status(
        active,
        f"🔨 [{item_num}/{total}] {_coder_label(active)} implementing: {item.description}",
        phase="coding", item_id=item.id, iteration=item.iteration_count + 1,
    )

    item.status = ItemStatus.IN_PROGRESS
    item.iteration_count += 1

    settings = get_settings()
    if item.iteration_count > settings.max_iterations_per_item:
        msg = f"Item {item.id} exceeded max iterations ({settings.max_iterations_per_item})"
        emit_error(active, msg)
        return {"stop_reason": msg, "phase": WorkflowPhase.STOPPED}

    prompt_parts = [
        f"## Task Assignment — {_coder_label(active)}",
        f"**Item ID**: {item.id}",
        f"**Description**: {item.description}",
        f"**Your peer reviewer**: {_reviewer_label(reviewer)}",
    ]
    if item.acceptance_criteria:
        prompt_parts.append("**Acceptance Criteria**:\n" + "\n".join(f"- {ac}" for ac in item.acceptance_criteria))
    if item.verification_commands:
        prompt_parts.append("**Verification Commands**:\n" + "\n".join(f"- `{vc}`" for vc in item.verification_commands))
    if item.review_notes:
        prompt_parts.append(f"**Rework Notes (from previous review)**:\n{item.review_notes}")
    if state.peer_review_notes and state.peer_review_verdict == "REWORK":
        prompt_parts.append(f"**Peer Review Feedback (REWORK)**:\n{state.peer_review_notes}")
    if item.test_report:
        prompt_parts.append(f"**Test Report (previous)**:\n{item.test_report}")

    prompt_parts.append(
        "\nImplement this task. Use tools to read the codebase, make changes, "
        "add tests, update docs. Keep diffs minimal. Follow Clean Architecture.\n"
        "Follow the coding style and architecture decisions from shared memory."
    )

    # inject_memory=True → shared memory is appended to system prompt
    result = _invoke_agent(active, [HumanMessage(content="\n\n".join(prompt_parts))],
                           CODER_TOOLS, inject_memory=True)

    emit_node_end(active, "Coding", f"Implementation complete — handing to {_reviewer_label(reviewer)} for peer review")

    return {
        "last_coder_result": result,
        "phase": WorkflowPhase.PEER_REVIEWING,
        "total_iterations": state.total_iterations + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: peer_review  (reads shared memory)
# ═══════════════════════════════════════════════════════════════════════════

def peer_review_node(state: GraphState) -> dict:
    """Cross-coder peer review with shared memory context."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to peer-review")
        return {"error_message": "No item to peer-review", "phase": WorkflowPhase.STOPPED}

    reviewer = state.active_reviewer
    implementer = state.active_coder
    impl_label = _coder_label(implementer)
    rev_label = _reviewer_label(reviewer)

    emit_node_start(reviewer, "Peer Review", item_id=item.id, item_desc=item.description)
    emit_status(reviewer, f"🔍 {rev_label} reviewing {impl_label}'s work on: {item.description}",
                phase="peer_reviewing", item_id=item.id)

    item.status = ItemStatus.IN_REVIEW

    prompt = (
        f"## Peer Code Review\n\n"
        f"**Reviewer**: {rev_label}\n"
        f"**Implementer**: {impl_label}\n"
        f"**Item**: {item.id} — {item.description}\n\n"
        f"**Implementer's Report**:\n{state.last_coder_result}\n\n"
        f"Review the changes. Also verify consistency with the shared memory "
        f"(coding style, architecture decisions, insights):\n"
        f"1. Use `git_command` with `git diff` to see the actual changes.\n"
        f"2. Use `git_command` with `git status` to see which files changed.\n"
        f"3. Read the modified files to understand the full context.\n"
        f"4. Run the test suite and linter to check for regressions.\n"
        f"5. Verify: correct logic, minimal diff, clean architecture, tests added, docs updated.\n"
        f"6. Check: does the code follow established patterns from shared memory?\n"
        f"7. Give your verdict: APPROVE or REWORK.\n"
        f"8. If REWORK, provide specific actionable notes.\n"
        f"9. If APPROVE, suggest a Conventional Commit message.\n"
        f"10. Note any NEW patterns, conventions, or insights discovered during review\n"
        f"    (these will be added to shared memory for future tasks)."
    )

    # inject_memory=True → reviewer sees established conventions
    result = _invoke_agent(reviewer, [HumanMessage(content=prompt)],
                           REVIEWER_TOOLS, inject_memory=True)

    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"
    if "**Verdict**: REWORK" in result or "Verdict: REWORK" in result:
        verdict = "REWORK"
    elif "**Verdict**: APPROVE" in result or "Verdict: APPROVE" in result:
        verdict = "APPROVE"

    emit_verdict(reviewer, verdict, detail=result, item_id=item.id)

    if verdict == "REWORK":
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS
        emit_status(reviewer, f"🔄 Peer review REWORK — sending back to {impl_label}", phase="coding", item_id=item.id)
    else:
        emit_status(reviewer, f"✅ Peer review APPROVED — extracting learnings → Planner review", phase="reviewing", item_id=item.id)

    emit_node_end(reviewer, "Peer Review", f"Verdict: {verdict}")

    return {
        "peer_review_verdict": verdict,
        "peer_review_notes": result,
        "phase": WorkflowPhase.REVIEWING if verdict == "APPROVE" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: learn_from_review  (extracts insights → memory files)
# ═══════════════════════════════════════════════════════════════════════════

def learn_from_review_node(state: GraphState) -> dict:
    """Extract learnings from the peer review and append to shared memory.

    Runs after peer review (regardless of APPROVE/REWORK) and before planner review.
    Uses a cheap LLM call (planner model) to extract structured insights.
    """
    item = state.current_item
    if not item:
        return {}

    review_text = state.peer_review_notes
    if not review_text:
        return {}

    emit_node_start("system", "Learning", item_id=item.id, item_desc="Extracting insights from review")

    prompt = (
        f"{LEARNING_EXTRACTION_PROMPT}\n\n"
        f"---\n\n"
        f"## Review to Analyze\n"
        f"**Item**: {item.id} — {item.description}\n"
        f"**Verdict**: {state.peer_review_verdict}\n\n"
        f"**Review Text**:\n{review_text}\n"
    )

    try:
        result = _invoke_agent("planner", [HumanMessage(content=prompt)])

        # Parse JSON from result
        learnings = _parse_learnings(result)
        total_added = 0

        for key in ["coding_style", "architecture", "insights"]:
            entries = learnings.get(key, [])
            for entry in entries:
                if entry and len(entry.strip()) > 5:
                    append_memory(key, entry.strip(), item_id=item.id)
                    total_added += 1

        if total_added > 0:
            emit_status("system",
                        f"🧠 Learned {total_added} new insight(s) from peer review of {item.id}",
                        phase="learning", item_id=item.id)
            logger.info("Learned %d insights from review of %s", total_added, item.id)
        else:
            emit_status("system",
                        f"🧠 No new generalizable insights from review of {item.id}",
                        phase="learning", item_id=item.id)

    except Exception as e:
        logger.warning("Learning extraction failed: %s", e)
        emit_status("system", f"⚠ Learning extraction skipped: {e}", phase="learning")

    emit_node_end("system", "Learning")

    # Don't change phase — we continue to wherever peer_review set us
    return {}


def _parse_learnings(result: str) -> dict:
    """Extract JSON from LLM result. Handles markdown fences and partial output."""
    text = result.strip()
    # Remove markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse learnings JSON: %s", text[:200])
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_review
# ═══════════════════════════════════════════════════════════════════════════

def planner_review_node(state: GraphState) -> dict:
    """Planner final review gate."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to review")
        return {"error_message": "No item to review", "phase": WorkflowPhase.STOPPED}

    emit_node_start("planner", "Final Review", item_id=item.id, item_desc=item.description)
    emit_status("planner", f"🎯 Planner final review: {item.description}", phase="reviewing", item_id=item.id)

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

    result = _invoke_agent("planner", [HumanMessage(content=prompt)], PLANNER_TOOLS)
    verdict = "APPROVE" if "APPROVE" in result.upper() else "REWORK"

    emit_verdict("planner", verdict, detail=result, item_id=item.id)

    if verdict == "REWORK":
        item.review_notes = result
        item.status = ItemStatus.IN_PROGRESS
        emit_status("planner", f"🔄 Planner REWORK — sending back to {impl_label}", phase="coding", item_id=item.id)
    else:
        emit_status("planner", f"✅ Planner APPROVED — sending to Tester", phase="testing", item_id=item.id)

    emit_node_end("planner", "Final Review", f"Verdict: {verdict}")

    return {
        "last_review_verdict": verdict,
        "review_notes": result,
        "phase": WorkflowPhase.TESTING if verdict == "APPROVE" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: tester
# ═══════════════════════════════════════════════════════════════════════════

def tester_node(state: GraphState) -> dict:
    """Run tests and verification."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to test")
        return {"error_message": "No item to test", "phase": WorkflowPhase.STOPPED}

    emit_node_start("tester", "Testing", item_id=item.id, item_desc=item.description)
    emit_status("tester", f"🧪 Running tests for: {item.description}", phase="testing", item_id=item.id)

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

    emit_verdict("tester", verdict, detail=result, item_id=item.id)

    if verdict == "FAIL":
        item.status = ItemStatus.IN_PROGRESS
        emit_status("tester", f"❌ Tests FAILED — sending back to {_coder_label(state.active_coder)}", phase="coding", item_id=item.id)
    else:
        emit_status("tester", f"✅ All tests PASSED", phase="deciding", item_id=item.id)

    emit_node_end("tester", "Testing", f"Verdict: {verdict}")

    return {
        "last_test_result": result,
        "phase": WorkflowPhase.DECIDING if verdict == "PASS" else WorkflowPhase.CODING,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: planner_decide
# ═══════════════════════════════════════════════════════════════════════════

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

    emit_status("planner", f"✅ Item {item_num}/{total} DONE: {item.description}", phase="deciding", item_id=item.id)

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
    return f"feat: {fallback_desc[:50].lower()}"


# ═══════════════════════════════════════════════════════════════════════════
# NODE: committer
# ═══════════════════════════════════════════════════════════════════════════

def committer_node(state: GraphState) -> dict:
    """Commit, push, and advance to next item."""
    item = state.current_item
    if not item:
        emit_error("system", "No item to commit")
        return {"error_message": "No item to commit", "phase": WorkflowPhase.STOPPED}

    emit_status("system", f"📦 Committing: {item.commit_message}", phase="committing", item_id=item.id)

    result = git_commit_and_push.invoke({"message": item.commit_message, "push": True})
    emit_commit(item.commit_message, item_id=item.id)
    logger.info("commit result: %s", result[:200])

    next_index = state.current_item_index + 1
    has_more = next_index < len(state.todo_items)

    if has_more:
        next_coder, next_reviewer = _assign_coder_pair(next_index)
        emit_status(
            "planner",
            f"➡ Moving to item {next_index + 1}/{len(state.todo_items)}: "
            f"{state.todo_items[next_index].description}  →  {_coder_label(next_coder)}",
            phase="coding",
        )
        return {
            "current_item_index": next_index,
            "phase": WorkflowPhase.CODING,
            "active_coder": next_coder,
            "active_reviewer": next_reviewer,
            "peer_review_notes": "",
            "peer_review_verdict": "",
        }
    else:
        # Log final memory stats
        stats = get_memory_stats()
        total = sum(s["chars"] for s in stats.values())
        emit_status("planner", f"🧠 Final memory: {total} chars across {len(stats)} files", phase="complete")
        emit_status("planner", f"🎉 All {len(state.todo_items)} items completed! Branch: {state.branch_name}", phase="complete")
        return {"phase": WorkflowPhase.COMPLETE}
