"""Investigation Priority Score (IPS) — interface definition.

Phase 1: Abstract interface ONLY. No implementation.

The IPS system will rank findings by their investigative importance,
considering multiple dimensions from each finding's metadata:

    Scoring Dimensions (expected in Finding.metadata):
    ──────────────────────────────────────────────────
    anomaly_strength       — Magnitude of statistical deviation
    statistical_significance — P-value or equivalent measure
    rarity                 — How unusual the pattern is (0.0–1.0)
    correlation_strength   — Strength of discovered relationship
    impact                 — Estimated impact on downstream analysis (0.0–1.0)
    confidence_interval    — Tuple of (lower, upper) CI bounds

    Additional Scoring Inputs:
    ──────────────────────────
    finding.severity       — Severity enum weight
    finding.confidence     — Module-reported confidence
    finding.column_count   — Scope of impact (how many columns)
    finding.row_count      — Scope of impact (how many rows)

The IPS should produce a single normalized score (0.0–1.0) per finding,
enabling global ranking across heterogeneous module outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from investigation_engine.models.finding import Finding


class IPSCalculator(ABC):
    """Abstract base class for Investigation Priority Score calculation.

    Implementations should:
        1. Extract scoring dimensions from Finding.metadata
        2. Normalize each dimension to a common scale
        3. Apply weighted combination
        4. Produce a single 0.0–1.0 score

    The scoring function must be:
        - Deterministic (same finding → same score)
        - Monotonic in each dimension
        - Transparent (weights and formula should be explainable)
    """

    @abstractmethod
    def score(self, finding: Finding) -> float:
        """Compute the Investigation Priority Score for a single finding.

        Args:
            finding: The finding to score.

        Returns:
            IPS score between 0.0 (lowest priority) and 1.0 (highest priority).
        """
        ...

    @abstractmethod
    def rank(self, findings: list[Finding]) -> list[Finding]:
        """Rank findings by their Investigation Priority Score.

        Args:
            findings: List of findings to rank.

        Returns:
            New list of findings sorted by IPS score (highest first).
            The original list is not modified.
        """
        ...

    @abstractmethod
    def explain(self, finding: Finding) -> dict[str, float]:
        """Explain the IPS score breakdown for a finding.

        Returns the contribution of each scoring dimension to the
        final score, enabling transparency and debugging.

        Args:
            finding: The finding to explain.

        Returns:
            Dictionary mapping dimension names to their weighted contributions.
        """
        ...
