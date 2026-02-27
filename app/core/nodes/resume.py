"""Resume node and TODO resume parsing helpers."""

from __future__ import annotations

from ._helpers import *

def _parse_todo_for_resume(todo_text: str) -> list[TodoItem]:
    """Parse tasks/todo.md and recover structured todo items."""
    items: list[TodoItem] = []
    current: TodoItem | None = None

    for raw_line in todo_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        match = CHECKBOX_RE.match(stripped)
        if match:
            if current:
                items.append(current)
            mark = match.group("mark").lower()
            desc = match.group("desc").strip()
            status = ItemStatus.DONE if mark == "x" else ItemStatus.PENDING
            current = TodoItem(
                id=f"item-{len(items) + 1:03d}",
                description=desc,
                task_type=classify_task_type(desc),
                status=status,
            )
            continue

        if not current:
            continue

        if stripped.startswith("- Type:"):
            current.task_type = stripped.split(":", 1)[1].strip().lower() or current.task_type
        elif stripped.startswith("- Owner:"):
            current.assigned_agent = _owner_to_agent(stripped.split(":", 1)[1].strip())
            if current.assigned_agent:
                current.assigned_reviewer = _reviewer_for_worker(current.assigned_agent)
        elif stripped.startswith("- AC:"):
            current.acceptance_criteria.append(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- Verify:"):
            verify = stripped.split(":", 1)[1].strip().strip("`")
            if verify:
                current.verification_commands.append(verify)

    if current:
        items.append(current)
    return items

def _resume_from_saved_todo(state: GraphState) -> dict | None:
    """Resume a previously planned workflow from tasks/todo.md."""
    todo_raw = read_file.invoke({"path": "tasks/todo.md"})
    if todo_raw.startswith("ERROR"):
        return None

    items = _parse_todo_for_resume(todo_raw)
    if not items:
        return None

    pending_indexes = [idx for idx, item in enumerate(items) if item.status != ItemStatus.DONE]
    if not pending_indexes:
        return None

    current_index = pending_indexes[0]
    completed = sum(1 for item in items if item.status == ItemStatus.DONE)
    current_item = items[current_index]

    if not current_item.assigned_agent:
        candidates = _candidate_agents_for_task(current_item.task_type)
        assigned_agent, _history = select_agent_thompson(state.repo_root, current_item.task_type, candidates)
        current_item.assigned_agent = assigned_agent
        current_item.assigned_reviewer = _reviewer_for_worker(assigned_agent)

    branch = state.branch_name
    if not branch:
        branch_result = git_command.invoke({"command": "git rev-parse --abbrev-ref HEAD"})
        for line in branch_result.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("[") and "stderr" not in candidate.lower():
                branch = candidate
                break

    emit_status(
        "planner",
        f"Resumed workflow at item {current_index + 1}/{len(items)}: {current_item.description}",
        phase="coding",
        items_total=len(items),
        items_done=completed,
        current_index=current_index + 1,
        branch=branch or "",
        platform=state.execution_platform or platform.platform(),
    )

    return {
        "todo_items": items,
        "current_item_index": current_index,
        "completed_items": completed,
        "branch_name": branch or "",
        "phase": WorkflowPhase.CODING,
        "active_coder": current_item.assigned_agent or "coder_a",
        "active_reviewer": (
            current_item.assigned_reviewer
            or _reviewer_for_worker(current_item.assigned_agent or "coder_a")
        ),
    }

def resume_node(state: GraphState) -> dict:
    """Resume branch using checkpoints first, then TODO fallback."""
    emit_node_start("planner", "Resume", item_desc=state.user_request[:100])

    restored = checkpoint_manager.load_checkpoint(repo_root=state.repo_root)
    if restored is not None:
        payload = restored.model_dump()
        payload["input_intent"] = "resume"
        payload["resumed_from_checkpoint"] = True

        # ── Plan approval was written into the checkpoint by mark_latest_plan_approval ──
        if payload.get("plan_approved") and not payload.get("needs_plan_approval"):
            feedback = payload.get("plan_approval_feedback", "")
            if feedback:
                # Revision requested — route back to planner for one more round
                payload["phase"] = WorkflowPhase.PLANNING
                payload["stop_reason"] = ""
                emit_node_end("planner", "Resume", "Plan revision requested — replanning")
            else:
                # Pure GO — route directly to coder, bypass gate
                payload["phase"] = WorkflowPhase.CODING
                payload["stop_reason"] = ""
                emit_node_end("planner", "Resume", "Plan approved — routing to coder")
            return payload

        # ── Commit-gate approval written by mark_latest_approval ──
        pending = payload.get("pending_approval", {})
        approved = isinstance(pending, dict) and bool(pending.get("approved"))
        if payload.get("needs_human_approval") and not approved:
            payload["phase"] = WorkflowPhase.WAITING_FOR_APPROVAL
            payload["stop_reason"] = "human_approval_required"
            emit_node_end("planner", "Resume", "Checkpoint restored; awaiting human approval")
        elif approved:
            payload["needs_human_approval"] = False
            payload["phase"] = WorkflowPhase.COMMITTING
            payload["stop_reason"] = ""
            emit_node_end("planner", "Resume", "Checkpoint restored; approval detected, resuming commit")
        else:
            emit_node_end("planner", "Resume", "Checkpoint restored")
        return payload

    # ── If an active plan is already in memory, resume from it directly ────
    # This covers the case where the workflow is mid-flight (e.g. waiting for
    # plan approval, or between items) and the user types "continue" or "resume".
    if state.todo_items:
        pending_items = [i for i in state.todo_items if i.status != ItemStatus.DONE]
        if pending_items:
            current_idx = next(
                (idx for idx, i in enumerate(state.todo_items) if i.status != ItemStatus.DONE),
                state.current_item_index,
            )
            emit_status(
                "planner",
                f"Resuming from in-memory plan — {len(pending_items)} item(s) remaining.",
                **_progress_meta(state, "coding"),
            )
            emit_node_end("planner", "Resume", f"Resumed in-memory plan at item {current_idx + 1}")
            return {
                "input_intent": "resume",
                "resumed_from_checkpoint": False,
                "phase": WorkflowPhase.CODING,
                "current_item_index": current_idx,
                "stop_reason": "",
            }

    resumed = _resume_from_saved_todo(state)
    if resumed is not None:
        resumed["input_intent"] = "resume"
        resumed["resumed_from_checkpoint"] = False
        emit_node_end("planner", "Resume", "Resumed workflow from tasks/todo.md")
        return resumed

    emit_status("planner", "No resumable TODO list found.", **_progress_meta(state, "complete"))
    emit_node_end("planner", "Resume", "Nothing to resume")
    return {
        "planner_response": "No resumable TODO items found in tasks/todo.md.",
        "phase": WorkflowPhase.COMPLETE,
        "stop_reason": "resume_not_found",
        "input_intent": "resume",
    }


# ---------------------------------------------------------------------------
# NODE: planner_plan
# ---------------------------------------------------------------------------
