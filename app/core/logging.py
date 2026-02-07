"""Centralized logging configuration for daedalus."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings

_configured = False


def setup_logging() -> logging.Logger:
    """Configure root logger with console + rotating file handlers. Idempotent."""
    global _configured
    if _configured:
        return logging.getLogger("daedalus")

    settings = get_settings()
    logger = logging.getLogger("daedalus")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (rotating, 5 MB × 3 backups)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _configured = True
    logger.info("Logging initialised (level=%s, file=%s)", settings.log_level, settings.log_file)
    return logger


def get_logger(name: str = "daedalus") -> logging.Logger:
    """Get a child logger. Always call setup_logging() at startup first."""
    return logging.getLogger(name)
