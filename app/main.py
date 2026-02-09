"""Daedalus main entry point.

Starts the FastAPI web server, Telegram bot, and background task processor.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading

import uvicorn

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging


def main():
    """Entry point: starts all services."""
    setup_logging()
    logger = get_logger("main")
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Daedalus starting")
    logger.info("=" * 60)

    if not settings.target_repo_path:
        logger.warning("TARGET_REPO_PATH not set - you'll need to specify repo_path per task")

    if not settings.openai_api_key.strip():
        logger.error("OPENAI_API_KEY not set - OpenAI-backed roles will fail")

    if not settings.anthropic_api_key.strip():
        logger.error("ANTHROPIC_API_KEY not set - Anthropic-backed roles will fail")

    logger.info("Web UI: http://%s:%d", settings.web_host, settings.web_port)

    if settings.telegram_bot_token:
        logger.info("Telegram bot: enabled")
    else:
        logger.info("Telegram bot: disabled (no TELEGRAM_BOT_TOKEN)")

    config = uvicorn.Config(
        "app.web.server:app",
        host=settings.web_host,
        port=settings.web_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
    server = uvicorn.Server(config)

    # Track Ctrl+C presses for escalating shutdown
    _ctrl_c_count = 0

    def _force_exit_after(seconds: float) -> None:
        """Force-kill the process after a grace period."""
        threading.Event().wait(seconds)
        logger.warning("Grace period expired — forcing exit")
        os._exit(1)

    def _handle_signal(signum, frame):
        nonlocal _ctrl_c_count
        _ctrl_c_count += 1

        from app.core.orchestrator import request_shutdown
        request_shutdown()

        if _ctrl_c_count == 1:
            logger.info("Ctrl+C received — shutting down gracefully (press again to force)")
            # Start a daemon thread that force-kills after 15s
            t = threading.Thread(target=_force_exit_after, args=(15,), daemon=True)
            t.start()
            # Let uvicorn handle the first signal normally
            server.should_exit = True
        else:
            logger.warning("Second Ctrl+C — forcing immediate exit")
            os._exit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    async def _run_all():
        """Run web server and telegram bot concurrently."""
        tasks = [server.serve()]

        if settings.telegram_bot_token:
            from app.telegram.bot import run_telegram_bot

            tasks.append(run_telegram_bot())

        await asyncio.gather(*tasks)

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
