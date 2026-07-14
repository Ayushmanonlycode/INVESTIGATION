"""Deterministic confidence propagation across the reasoning pipeline."""

from __future__ import annotations

from statistics import mean

from investigation_engine.reasoning.evidence.models import EvidenceUnit


class ConfidencePropagationEngine:
    """Computes explainable confidence scores from deterministic factors."""

    _EVIDENCE_WEIGHT = 0.38
    _COMMUNITY_WEIGHT = 0.27
    _CROSS_INVESTIGATOR_WEIGHT = 0.22
    _CONTRADICTION_WEIGHT = 0.08
    _UNKNOWN_WEIGHT = 0.05

    def score(
        self,
        supporting: list[EvidenceUnit],
        *,
        contradicting: list[EvidenceUnit] | None = None,
        unknown_evidence: list[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Return a confidence score and its weighted factor breakdown."""
        contradicting = contradicting or []
        unknown_evidence = unknown_evidence or []

        evidence_strength = (
            mean(evidence.strength for evidence in supporting) / 100.0 if supporting else 0.0
        )
        community_strength = (
            mean(evidence.community_strength for evidence in supporting) if supporting else 0.0
        )
        investigators = {
            module
            for evidence in supporting
            for module in evidence.provenance.get("investigators", [])
        }
        cross_investigator_agreement = min(1.0, len(investigators) / 2.0)
        contradiction_factor = min(
            1.0,
            (
                mean(evidence.strength for evidence in contradicting) / 100.0
                if contradicting
                else 0.0
            ),
        )
        unknown_factor = min(1.0, len(set(unknown_evidence)) / 5.0)

        breakdown = {
            "evidence_strength": round(evidence_strength * self._EVIDENCE_WEIGHT, 2),
            "community_strength": round(community_strength * self._COMMUNITY_WEIGHT, 2),
            "cross_investigator_agreement": round(
                cross_investigator_agreement * self._CROSS_INVESTIGATOR_WEIGHT,
                2,
            ),
            "contradicting_evidence": round(
                contradiction_factor * -self._CONTRADICTION_WEIGHT,
                2,
            ),
            "unknown_evidence": round(unknown_factor * -self._UNKNOWN_WEIGHT, 2),
        }
        confidence = round(
            max(0.0, min(1.0, sum(breakdown.values()))),
            2,
        )
        return confidence, breakdown
