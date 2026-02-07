"""Telegram bot interface for AI Dev Worker.

Commands:
  /task <text>  — submit a new task
  /status       — show current workflow status
  /logs         — show recent log entries (last 10)
  /stop         — request workflow stop
"""

from __future__ import annotations

import asyncio

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.orchestrator import run_workflow
from app.core.state import GraphState, WorkflowPhase

logger = get_logger("telegram.bot")

_current_state: GraphState | None = None
_stop_requested = False


def _is_allowed(user_id: int) -> bool:
    """Check if user is in the allowed list (empty = allow all)."""
    settings = get_settings()
    allowed = settings.allowed_telegram_ids
    return not allowed or user_id in allowed


async def cmd_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /task command — submit a new coding task."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    task_text = " ".join(context.args) if context.args else ""
    if not task_text:
        await update.message.reply_text("Usage: /task <description of the coding task>")
        return

    await update.message.reply_text(f"📋 Task received: {task_text[:200]}\n\nStarting workflow...")

    global _current_state, _stop_requested
    _stop_requested = False

    try:
        settings = get_settings()
        _current_state = GraphState(
            user_request=task_text,
            repo_root=settings.target_repo_path,
            phase=WorkflowPhase.PLANNING,
        )

        final_state = await run_workflow(task_text, settings.target_repo_path)
        _current_state = final_state

        # Summary
        done = final_state.completed_items
        total = len(final_state.todo_items)
        status_emoji = "✅" if final_state.phase == WorkflowPhase.COMPLETE else "🛑"

        summary = (
            f"{status_emoji} Workflow finished\n"
            f"Phase: {final_state.phase.value}\n"
            f"Progress: {done}/{total} items done\n"
            f"Branch: {final_state.branch_name or 'n/a'}"
        )
        if final_state.stop_reason:
            summary += f"\nStop reason: {final_state.stop_reason}"
        if final_state.error_message:
            summary += f"\nError: {final_state.error_message}"

        await update.message.reply_text(summary)

    except Exception as e:
        logger.error("Task failed: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Task failed: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    if not _current_state:
        await update.message.reply_text("💤 No active task.")
        return

    await update.message.reply_text(
        f"📊 Status\n"
        f"Phase: {_current_state.phase.value}\n"
        f"Branch: {_current_state.branch_name or 'n/a'}\n"
        f"Progress: {_current_state.completed_items}/{len(_current_state.todo_items)} items\n"
        f"{_current_state.get_progress_summary()}"
    )


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /logs command — show recent activity."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    # Read from log file (last 20 lines)
    settings = get_settings()
    try:
        from pathlib import Path
        log_path = Path(settings.log_file)
        if log_path.exists():
            lines = log_path.read_text().strip().split("\n")[-20:]
            text = "\n".join(lines)
            if len(text) > 4000:
                text = text[-4000:]
            await update.message.reply_text(f"📜 Recent logs:\n```\n{text}\n```", parse_mode="Markdown")
        else:
            await update.message.reply_text("No log file found.")
    except Exception as e:
        await update.message.reply_text(f"Error reading logs: {e}")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stop command — request workflow stop."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorized.")
        return

    global _stop_requested
    _stop_requested = True
    await update.message.reply_text("🛑 Stop requested. The workflow will halt after the current step.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages as task submissions."""
    if not _is_allowed(update.effective_user.id):
        return

    text = update.message.text.strip()
    if text:
        # Treat as a task
        context.args = text.split()
        await cmd_task(update, context)


def create_telegram_app() -> Application | None:
    """Create and configure the Telegram bot application. Returns None if token not set."""
    settings = get_settings()
    if not settings.telegram_bot_token:
        logger.info("Telegram bot token not set — skipping Telegram integration")
        return None

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("task", cmd_task))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Telegram bot configured")
    return app


async def run_telegram_bot():
    """Start the Telegram bot polling loop."""
    telegram_app = create_telegram_app()
    if telegram_app is None:
        return

    logger.info("Starting Telegram bot polling...")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
