"""Performance timing utilities.

Provides decorators and context managers for measuring execution time
of investigation modules and engine operations.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that logs the execution time of a function.

    Logs at INFO level with the function name and duration in seconds.
    Useful for tracking investigation module performance.

    Args:
        func: The function to time.

    Returns:
        Wrapped function with timing instrumentation.

    Example:
        @timed
        def investigate(self, df, dataset_info, config):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                "{}.{} completed in {:.3f}s",
                func.__module__,
                func.__qualname__,
                elapsed,
            )
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error(
                "{}.{} failed after {:.3f}s",
                func.__module__,
                func.__qualname__,
                elapsed,
            )
            raise

    return wrapper


@contextmanager
def timer(operation_name: str) -> Any:
    """Context manager for timing a block of code.

    Args:
        operation_name: Human-readable name for the operation being timed.

    Yields:
        A dictionary with 'elapsed' key that is populated after the block completes.

    Example:
        with timer("loading dataset") as t:
            df = pd.read_csv("data.csv")
        print(f"Took {t['elapsed']:.3f}s")
    """
    result: dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        logger.info("{} completed in {:.3f}s", operation_name, result["elapsed"])
