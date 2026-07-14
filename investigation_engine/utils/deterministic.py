"""Deterministic hashing helpers for reproducible reasoning artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_normalize(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    return value


def stable_payload(value: Any) -> str:
    """Serialize an object deterministically for hashing or comparison."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_id(prefix: str, value: Any, *, length: int = 12) -> str:
    """Generate a deterministic short identifier from structured content."""
    digest = hashlib.sha1(stable_payload(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def stable_datetime(prefix: str, value: Any) -> datetime:
    """Generate a deterministic UTC timestamp from structured content."""
    digest = hashlib.sha1(stable_payload({"prefix": prefix, "value": value}).encode("utf-8")).hexdigest()
    seconds = int(digest[:10], 16) % (365 * 24 * 60 * 60)
    microseconds = int(digest[10:15], 16) % 1_000_000
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(
        seconds=seconds,
        microseconds=microseconds,
    )
