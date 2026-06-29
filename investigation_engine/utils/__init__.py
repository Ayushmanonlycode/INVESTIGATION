"""Utilities package — shared helpers for logging, timing, and statistics."""

from investigation_engine.utils.logging import configure_logging
from investigation_engine.utils.timing import timed

__all__ = ["configure_logging", "timed"]
