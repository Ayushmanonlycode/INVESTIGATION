"""Structured logging configuration using Loguru.

Provides a centralized logging setup for the Investigation Engine.
All modules should use `from loguru import logger` directly —
Loguru's global logger is configured once at engine startup.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from investigation_engine.config.settings import LoggingSettings


_CONFIGURED: bool = False


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Configure the global Loguru logger.

    Should be called once at engine initialization. Subsequent calls
    are no-ops to prevent duplicate handlers.

    Args:
        settings: Logging configuration. Uses defaults if None.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    # Remove default handler
    logger.remove()

    if settings is None:
        from investigation_engine.config.settings import LoggingSettings
        settings = LoggingSettings()

    # Console handler
    logger.add(
        sys.stderr,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Optional file handler
    if settings.log_file:
        logger.add(
            settings.log_file,
            format=settings.log_format,
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info("Investigation Engine logging configured | level={}", settings.log_level)
    _CONFIGURED = True


def reset_logging() -> None:
    """Reset logging configuration. Used primarily in testing.

    Removes all handlers and resets the configured flag,
    allowing configure_logging to be called again.
    """
    global _CONFIGURED
    logger.remove()
    _CONFIGURED = False
