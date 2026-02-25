"""Telegram bot interface for Daedalus.

Commands:
  /task <text>  — submit a new coding task
  /status       — show current workflow status
  /logs         — show recent log entries (last 20)
  /approve      — approve a pending human-gate commit
  /reject       — reject a pending human-gate commit
  /stop         — request workflow stop

Notifications (sent automatically):
  • Approval required  — when the human gate fires during any workflow
  • Workflow complete   — when a task finishes (COMPLETE or STOPPED)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.core.approval_registry import registry as approval_registry
from app.core.config import get_settings
from app.core.events import EventCategory, WorkflowEvent, subscribe_sync
from app.core.logging import get_logger
from app.core.orchestrator import run_workflow
from app.core.state import GraphState, WorkflowPhase
logger = get_logger("telegram.bot")

# ── Module-level state ───────────────────────────────────────────────────
_current_state: GraphState | None = None
_telegram_app: Application | None = None   # set by create_telegram_app()

# ── Helpers ──────────────────────────────────────────────────────────────


def _is_allowed(user_id: int) -> bool:
    """Return True if the user is in the allow-list (empty list = allow all)."""
    settings = get_settings()
    allowed = settings.allowed_telegram_ids
    return not allowed or user_id in allowed


def _allowed_chat_ids() -> list[int]:
    """Return the configured allowed Telegram user IDs."""
    return get_settings().allowed_telegram_ids


async def _notify_all(text: str, reply_markup=None) -> None:
    """Send a message to every allowed Telegram user."""
    if _telegram_app is None:
        return
    for uid in _allowed_chat_ids():
        try:
            await _telegram_app.bot.send_message(
                chat_id=uid,
                text=text,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
        except Exception as exc:
            logger.warning("Failed to notify Telegram user %d: %s", uid, exc)


# ── Event bus listener ───────────────────────────────────────────────────

def _on_workflow_event(event: WorkflowEvent) -> None:
    """Sync listener on the event bus — schedules Telegram notifications."""
    if _telegram_app is None:
        return

    try:
        loop = _telegram_app.bot.loop  # type: ignore[attr-defined]
    except AttributeError:
        return

    if event.category == EventCategory.APPROVAL_NEEDED:
        asyncio.run_coroutine_threadsafe(
            _send_approval_notification(event.metadata),
            loop,
        )
    elif event.category == EventCategory.PLAN_APPROVAL:
        asyncio.run_coroutine_threadsafe(
            _send_plan_approval_notification(event.metadata),
            loop,
        )
    elif event.category == EventCategory.CODER_QUESTION:
        asyncio.run_coroutine_threadsafe(
            _send_question_notification(event.metadata),
            loop,
        )
    elif event.category == EventCategory.STATUS:
        phase = (event.metadata or {}).get("phase", "")
        if phase in ("complete", "stopped"):
            asyncio.run_coroutine_threadsafe(
                _send_completion_notification(event.title, phase),
                loop,
            )


async def _send_approval_notification(meta: dict) -> None:
    """Send an approval-request message with inline APPROVE / REJECT buttons."""
    summary    = meta.get("summary", "changes pending")
    files      = meta.get("files", [])
    triggers   = meta.get("triggers", [])
    git_status = meta.get("git_status", "")

    trigger_lines = "\n".join(
        f"  \u2022 {t.get('reason', t.get('type', '?'))}" for t in triggers
    )
    file_lines = "\n".join(f"  `{f}`" for f in files[:15])
    if len(files) > 15:
        file_lines += f"\n  \u2026 and {len(files) - 15} more"

    status_snippet = ""
    if git_status:
        snippet = git_status[:600]
        status_snippet = f"\n\n*Git status:*\n```\n{snippet}\n```"

    text = (
        f"\u26a0\ufe0f *HUMAN APPROVAL REQUIRED*\n\n"
        f"*Summary:* {summary}\n\n"
        f"*Reasons:*\n{trigger_lines}\n\n"
        f"*Changed files:*\n{file_lines}"
        f"{status_snippet}\n\n"
        f"Use the buttons below or /approve / /reject."
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 APPROVE", callback_data="approval:approve"),
            InlineKeyboardButton("\u274c REJECT",  callback_data="approval:reject"),
        ]
    ])

    await _notify_all(text, reply_markup=keyboard)


async def _send_question_notification(meta: dict) -> None:
    """Send a coder question to all allowed Telegram users with inline option buttons."""
    question   = meta.get("question", "(no question)")
    context    = meta.get("context", "")
    options    = meta.get("options", [])
    asked_by   = meta.get("asked_by", "coder")
    agent_label = "Coder 1" if asked_by == "coder_a" else "Coder 2"

    context_part = ""
    if context:
        context_part = f"\n\n*Context:*\n_{context[:600]}_"

    text = (
        f"\U0001f914 *{agent_label} is asking a question*\n\n"
        f"{question}"
        f"{context_part}\n\n"
        f"Reply with /answer <your answer> or use the buttons below."
    )

    # Build inline keyboard from options (max 4 buttons)
    buttons = []
    for i, opt in enumerate(options[:4]):
        short = opt[:40]
        buttons.append(InlineKeyboardButton(short, callback_data=f"answer:{i}:{short}"))

    keyboard = InlineKeyboardMarkup([buttons]) if buttons else None

    # Store option texts so callback can resolve index → full text
    # We encode the full option text directly in the callback_data (truncated)
    await _notify_all(text, reply_markup=keyboard)


async def _send_completion_notification(title: str, phase: str) -> None:
    """Send a workflow-completion summary to all allowed users."""
    icon = "\U0001f389" if phase == "complete" else "\U0001f6d1"
    lines = [f"{icon} *Workflow {phase.upper()}*", "", title]

    if _current_state:
        done  = _current_state.completed_items
        total = len(_current_state.todo_items)
        lines.append(f"Progress: {done}/{total} items")
        if _current_state.branch_name:
            lines.append(f"Branch: `{_current_state.branch_name}`")
        if _current_state.stop_reason and _current_state.stop_reason != "user_rejected":
            lines.append(f"Stop reason: {_current_state.stop_reason}")

    await _notify_all("\n".join(lines))


# ── Command handlers ─────────────────────────────────────────────────────

async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /task — submit a new coding task."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    task_text = " ".join(context.args) if context.args else ""
    if not task_text:
        await update.message.reply_text("Usage: /task <description of the coding task>")
        return

    await update.message.reply_text(
        f"\U0001f4cb Task received: {task_text[:200]}\n\nStarting workflow\u2026"
    )

    global _current_state
    try:
        settings = get_settings()
        # repo_ref may be passed as part of the task text in future; for now
        # fall back to the static TARGET_REPO_PATH.
        repo_path = settings.target_repo_path
        repo_ref  = ""
        _current_state = GraphState(
            user_request=task_text,
            repo_root=repo_path,
            repo_ref=repo_ref,
            phase=WorkflowPhase.PLANNING,
        )
        final_state = await run_workflow(task_text, repo_path, repo_ref=repo_ref)
        _current_state = final_state

        done  = final_state.completed_items
        total = len(final_state.todo_items)
        icon  = "\u2705" if final_state.phase == WorkflowPhase.COMPLETE else "\U0001f6d1"

        summary = (
            f"{icon} Workflow finished\n"
            f"Phase: {final_state.phase.value}\n"
            f"Progress: {done}/{total} items done\n"
            f"Branch: {final_state.branch_name or 'n/a'}"
        )
        if final_state.stop_reason:
            summary += f"\nStop reason: {final_state.stop_reason}"
        if final_state.error_message:
            summary += f"\nError: {final_state.error_message}"

        await update.message.reply_text(summary)

    except Exception as exc:
        logger.error("Task failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"\u274c Task failed: {exc}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show current workflow status."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    if not _current_state:
        await update.message.reply_text("\U0001f4a4 No active task.")
        return

    pending_note = ""
    if _current_state.needs_human_approval:
        pending_note = "\n\n\u26a0\ufe0f *Waiting for your approval.* Use /approve or /reject."

    await update.message.reply_text(
        f"\U0001f4ca *Status*\n"
        f"Phase: `{_current_state.phase.value}`\n"
        f"Branch: `{_current_state.branch_name or 'n/a'}`\n"
        f"Progress: {_current_state.completed_items}/{len(_current_state.todo_items)} items\n"
        f"{_current_state.get_progress_summary()}"
        f"{pending_note}",
        parse_mode="Markdown",
    )


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /logs — show recent log file entries."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    settings = get_settings()
    try:
        log_path = Path(settings.log_file)
        if log_path.exists():
            lines = log_path.read_text().strip().split("\n")[-20:]
            text  = "\n".join(lines)
            if len(text) > 3800:
                text = text[-3800:]
            await update.message.reply_text(
                f"\U0001f4dc Recent logs:\n```\n{text}\n```",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("No log file found.")
    except Exception as exc:
        await update.message.reply_text(f"Error reading logs: {exc}")


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /approve — approve a pending human-gate commit."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    if not approval_registry.is_pending:
        await update.message.reply_text(
            "\u2139\ufe0f Nothing is pending approval right now."
        )
        return

    resolved = approval_registry.approve(approved=True)
    if resolved:
        await update.message.reply_text(
            "\u2705 *Approved.* The workflow will continue to commit.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "\u26a0\ufe0f Could not resolve — may have already been handled."
        )


async def cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reject — reject a pending human-gate commit."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    if not approval_registry.is_pending:
        await update.message.reply_text(
            "\u2139\ufe0f Nothing is pending approval right now."
        )
        return

    resolved = approval_registry.approve(approved=False)
    if resolved:
        await update.message.reply_text(
            "\u274c *Rejected.* The workflow has been stopped.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "\u26a0\ufe0f Could not resolve — may have already been handled."
        )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop — request a graceful workflow stop."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    from app.core.orchestrator import request_shutdown
    request_shutdown()
    await update.message.reply_text(
        "\U0001f6d1 Stop requested. The workflow will halt after the current step."
    )


async def cmd_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /answer <text> — deliver a human answer to the waiting coder."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return

    answer_text = " ".join(context.args).strip() if context.args else ""
    if not answer_text:
        await update.message.reply_text(
            "Usage: /answer <your answer>\n"
            "Example: /answer Use Redis — we already have it in infrastructure."
        )
        return

    if not approval_registry.is_question_pending:
        await update.message.reply_text(
            "\u2139\ufe0f No coder question is pending right now."
        )
        return

    delivered = approval_registry.deliver_answer(answer_text)
    if delivered:
        await update.message.reply_text(
            f"\U0001f4ac *Answer delivered:* {answer_text}\n\nThe coder will continue.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "\u26a0\ufe0f Could not deliver — question may have already been answered."
        )


async def handle_inline_answer(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle inline keyboard option button presses for coder questions."""
    query = update.callback_query
    await query.answer()

    if not _is_allowed(query.from_user.id):
        await query.edit_message_text("\u26d4 Not authorized.")
        return

    data = query.data or ""
    # Format: "answer:<index>:<text>"
    parts = data.split(":", 2)
    answer_text = parts[2] if len(parts) == 3 else ""

    if not answer_text:
        await query.edit_message_text("\u26a0\ufe0f Could not parse option.")
        return

    if not approval_registry.is_question_pending:
        await query.edit_message_text(
            "\u2139\ufe0f This question has already been answered."
        )
        return

    delivered = approval_registry.deliver_answer(answer_text)
    if delivered:
        await query.edit_message_text(
            f"\U0001f4ac *Answer delivered:* {answer_text}\n\nThe coder will continue.",
            parse_mode="Markdown",
        )
    else:
        await query.edit_message_text(
            "\u26a0\ufe0f Could not deliver — may have already been answered."
        )


async def _send_plan_approval_notification(meta: dict) -> None:
    """Send a plan-ready message with inline Approve / Revise / Cancel buttons."""
    items      = meta.get("items", [])
    count      = meta.get("items_count", len(items))

    item_lines = "\n".join(
        f"  {i + 1}. [{item.get('task_type', 'coding')}] {item.get('description', '')}"
        for i, item in enumerate(items[:20])
    )
    if len(items) > 20:
        item_lines += f"\n  … and {len(items) - 20} more"

    text = (
        f"\U0001f4cb *PLAN READY — YOUR APPROVAL REQUIRED*\n\n"
        f"*{count} item{'s' if count != 1 else ''}:*\n"
        f"{item_lines}\n\n"
        f"Tap a button below, or use /plan\\_approve, /plan\\_revise, or /plan\\_cancel."
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 Approve",        callback_data="plan_approval:approve"),
            InlineKeyboardButton("\u274c Cancel",          callback_data="plan_approval:cancel"),
        ],
        [
            InlineKeyboardButton("\u270f\ufe0f Approve with note", callback_data="plan_approval:revise"),
        ],
    ])

    await _notify_all(text, reply_markup=keyboard, parse_mode="Markdown")


async def handle_inline_plan_approval(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle inline keyboard buttons for plan approval."""
    query = update.callback_query
    await query.answer()

    if not _is_allowed(query.from_user.id):
        await query.edit_message_text("\u26d4 Not authorized.")
        return

    data   = query.data or ""
    action = data.split(":", 1)[1] if ":" in data else ""

    if action == "cancel":
        await _call_plan_approve(approved=False, feedback="")
        await query.edit_message_text("\u274c Task cancelled by user.")
        return

    if action == "approve":
        result = await _call_plan_approve(approved=True, feedback="")
        if result:
            await query.edit_message_text(
                "\u2705 *Plan approved — \U0001f680 coding started!*", parse_mode="Markdown"
            )
        else:
            await query.edit_message_text("\u26a0\ufe0f Could not submit approval — workflow may have moved on.")
        return

    if action == "revise":
        # Ask user to send their revision note as a reply
        await query.edit_message_text(
            "\u270f\ufe0f *Send your revision note*\n\n"
            "Reply with /plan\\_revise <your note> to submit a revision request.",
            parse_mode="Markdown",
        )


async def _call_plan_approve(approved: bool, feedback: str) -> bool:
    """POST to /api/plan-approve on the local server."""
    import aiohttp
    settings = get_settings()
    host = getattr(settings, "web_host", "127.0.0.1")
    port = getattr(settings, "web_port", 8420)
    url  = f"http://{host}:{port}/api/plan-approve"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"approved": approved, "feedback": feedback},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception as exc:
        logger.warning("_call_plan_approve failed: %s", exc)
        return False


async def cmd_plan_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /plan_approve — approve the plan as-is."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return
    ok = await _call_plan_approve(approved=True, feedback="")
    msg = "\u2705 Plan approved — coding started!" if ok else "\u26a0\ufe0f No plan awaiting approval."
    await update.message.reply_text(msg)


async def cmd_plan_revise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /plan_revise <note> — approve with a revision note."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return
    note = " ".join(context.args or []).strip()
    if not note:
        await update.message.reply_text("Usage: /plan\\_revise <your revision note>", parse_mode="Markdown")
        return
    ok = await _call_plan_approve(approved=True, feedback=note)
    msg = f"\u270f\ufe0f Revision sent — planner will update the plan." if ok else "\u26a0\ufe0f No plan awaiting approval."
    await update.message.reply_text(msg)


async def cmd_plan_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /plan_cancel — cancel the task."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("\u26d4 Not authorized.")
        return
    ok = await _call_plan_approve(approved=False, feedback="")
    msg = "\u274c Task cancelled." if ok else "\u26a0\ufe0f No plan awaiting approval."
    await update.message.reply_text(msg)


async def handle_inline_approval(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle inline keyboard APPROVE / REJECT button presses."""
    query = update.callback_query
    await query.answer()

    if not _is_allowed(query.from_user.id):
        await query.edit_message_text("\u26d4 Not authorized.")
        return

    data     = query.data or ""
    action   = data.split(":", 1)[1] if ":" in data else ""
    approved = action == "approve"

    if not approval_registry.is_pending:
        await query.edit_message_text(
            "\u2139\ufe0f This approval has already been resolved."
        )
        return

    resolved = approval_registry.approve(approved=approved)
    if resolved:
        result = (
            "\u2705 *Approved.* Workflow continuing to commit\u2026"
            if approved
            else "\u274c *Rejected.* Workflow stopped."
        )
        await query.edit_message_text(result, parse_mode="Markdown")
    else:
        await query.edit_message_text(
            "\u26a0\ufe0f Could not resolve — may have already been handled."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Treat plain-text messages as task submissions."""
    if not _is_allowed(update.effective_user.id):
        return
    text = update.message.text.strip()
    if text:
        context.args = text.split()
        await cmd_task(update, context)


# ── App factory ──────────────────────────────────────────────────────────

def create_telegram_app() -> Optional[Application]:
    """Build and configure the Telegram Application.  Returns None if no token."""
    global _telegram_app

    settings = get_settings()
    if not settings.telegram_bot_token:
        logger.info("Telegram bot token not set — skipping Telegram integration")
        return None

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("task",         cmd_task))
    app.add_handler(CommandHandler("status",       cmd_status))
    app.add_handler(CommandHandler("logs",         cmd_logs))
    app.add_handler(CommandHandler("approve",      cmd_approve))
    app.add_handler(CommandHandler("reject",       cmd_reject))
    app.add_handler(CommandHandler("answer",       cmd_answer))
    app.add_handler(CommandHandler("stop",         cmd_stop))
    app.add_handler(CommandHandler("plan_approve", cmd_plan_approve))
    app.add_handler(CommandHandler("plan_revise",  cmd_plan_revise))
    app.add_handler(CommandHandler("plan_cancel",  cmd_plan_cancel))
    app.add_handler(CallbackQueryHandler(handle_inline_approval,      pattern=r"^approval:"))
    app.add_handler(CallbackQueryHandler(handle_inline_plan_approval, pattern=r"^plan_approval:"))
    app.add_handler(CallbackQueryHandler(handle_inline_answer,        pattern=r"^answer:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Subscribe to the shared event bus for push notifications
    subscribe_sync(_on_workflow_event)

    _telegram_app = app
    logger.info(
        "Telegram bot configured — commands: /task /status /logs /approve /reject "
        "/answer /plan_approve /plan_revise /plan_cancel /stop"
    )
    return app


# ── Runner ───────────────────────────────────────────────────────────────

async def run_telegram_bot() -> None:
    """Start the Telegram bot polling loop (runs until cancelled)."""
    telegram_app = create_telegram_app()
    if telegram_app is None:
        return

    logger.info("Starting Telegram bot polling\u2026")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
