"""Daedalus main entry point.

Starts the FastAPI web server, Telegram bot, and background task processor.
"""

from __future__ import annotations

import asyncio

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
