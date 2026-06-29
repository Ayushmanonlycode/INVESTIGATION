"""Severity levels for investigation findings.

Ordered from most critical to informational. Used by all investigation modules
to classify the urgency of a finding. The IPS engine will consume these levels
as part of its ranking algorithm.
"""

from __future__ import annotations

from enum import Enum


class Severity(str, Enum):
    """Investigation finding severity classification.

    Levels are ordered by investigative urgency:
        CRITICAL — Requires immediate analyst attention. Data integrity at risk.
        HIGH     — Significant anomaly or pattern that likely impacts analysis.
        MEDIUM   — Notable finding that warrants review.
        LOW      — Minor observation, useful context for deeper investigation.
        INFO     — Informational only, no action required.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def numeric_weight(self) -> float:
        """Return a numeric weight for IPS scoring.

        Higher values indicate greater investigative importance.
        """
        weights: dict[Severity, float] = {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.8,
            Severity.MEDIUM: 0.5,
            Severity.LOW: 0.2,
            Severity.INFO: 0.0,
        }
        return weights[self]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) < order.index(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) <= order.index(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) > order.index(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = list(Severity)
        return order.index(self) >= order.index(other)
