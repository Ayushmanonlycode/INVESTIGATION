"""Hypothesis generation engine."""

from __future__ import annotations

from loguru import logger

from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker


class HypothesisGenerationEngine:
    """Generates deterministic, evidence-backed hypotheses."""

    def generate(
        self,
        knowledge_objects: list[KnowledgeObject],
        evidence_units: list[EvidenceUnit],
    ) -> list[Hypothesis]:
        if not knowledge_objects:
            return []

        evidence_lookup = {evidence.evidence_id: evidence for evidence in evidence_units}
        hypotheses = [
            self._build_hypothesis(knowledge, evidence_lookup)
            for knowledge in knowledge_objects
        ]
        logger.info(
            "HypothesisGenerationEngine: generated {} hypotheses from {} knowledge objects",
            len(hypotheses),
            len(knowledge_objects),
        )
        return sorted(hypotheses, key=lambda item: (-item.confidence, item.title))

    def _build_hypothesis(
        self,
        knowledge: KnowledgeObject,
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> Hypothesis:
        supporting = [
            evidence_lookup[evidence_id]
            for evidence_id in knowledge.supporting_evidence
            if evidence_id in evidence_lookup
        ]
        lead = max(supporting, key=lambda evidence: (evidence.strength, evidence.confidence))
        contradicting = self._find_contradictions(knowledge, supporting, evidence_lookup)
        title, statement = self._compose_statement(knowledge, lead)
        confidence = round(
            max(0.0, min(
                1.0,
                sum(evidence.confidence for evidence in supporting) / max(len(supporting), 1)
                - 0.1 * len(contradicting),
            )),
            2,
        )

        hypothesis = Hypothesis(
            title=title,
            statement=statement,
            supporting_evidence=[evidence.evidence_id for evidence in supporting],
            contradicting_evidence=[evidence.evidence_id for evidence in contradicting],
            unknowns=self._unknowns_for(lead),
            confidence=confidence,
            plausible_causes=self._causes_for(lead),
            validation_steps=self._validation_steps_for(lead, knowledge),
            provenance={},
            metadata={
                "dominant_category": lead.category,
                "affected_columns": knowledge.metadata.get("affected_columns", []),
                "knowledge_concept": knowledge.concept,
            },
        )
        hypothesis.provenance = ProvenanceTracker.for_hypothesis(
            hypothesis,
            knowledge,
            evidence_lookup,
        )
        return hypothesis

    def _find_contradictions(
        self,
        knowledge: KnowledgeObject,
        supporting: list[EvidenceUnit],
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> list[EvidenceUnit]:
        supporting_categories = {evidence.category for evidence in supporting}
        contradicting: list[EvidenceUnit] = []
        for evidence in evidence_lookup.values():
            if evidence.evidence_id in knowledge.supporting_evidence:
                continue
            if not set(evidence.affected_columns).intersection(
                knowledge.metadata.get("affected_columns", [])
            ):
                continue
            if evidence.category not in supporting_categories:
                contradicting.append(evidence)
        return sorted(contradicting, key=lambda item: (-item.strength, item.title))[:3]

    def _compose_statement(
        self,
        knowledge: KnowledgeObject,
        lead: EvidenceUnit,
    ) -> tuple[str, str]:
        concept = knowledge.concept
        if lead.category == "Systematic Missingness":
            return (
                f"{concept} appear systematically unavailable",
                f"Available evidence indicates that {concept.lower()} are "
                "systematically unavailable in this dataset.",
            )
        if lead.category == "Correlation Community":
            return (
                f"{concept} behave as one related signal family",
                f"Available evidence indicates that {concept.lower()} move together "
                "strongly enough to warrant joint investigation.",
            )
        if lead.category == "Identifier Integrity Failure":
            return (
                f"{concept} may be compromising record linkage",
                f"Available evidence indicates that {concept.lower()} include "
                "identifier integrity failures that can compromise joins or deduplication.",
            )
        if lead.category == "Constant Feature Group":
            return (
                f"{concept} may be structurally non-informative",
                f"Available evidence indicates that {concept.lower()} contain "
                "non-varying features with little analytical value.",
            )
        if lead.category == "Low Information Feature Group":
            return (
                f"{concept} may contribute limited analytical signal",
                f"Available evidence indicates that {concept.lower()} provide limited "
                "information because variation is minimal.",
            )
        return (
            f"{concept} deserve investigation",
            f"Available evidence indicates that {concept.lower()} exhibit a coherent "
            "pattern deserving analyst review.",
        )

    def _causes_for(self, evidence: EvidenceUnit) -> list[str]:
        mapping = {
            "Systematic Missingness": [
                "Data collection may be conditional on an operational state.",
                "Upstream telemetry or ingestion may be omitting this feature family.",
            ],
            "Correlation Community": [
                "Several columns may be measuring the same latent factor.",
                "Derived or duplicated features may be re-expressing the same source signal.",
            ],
            "Identifier Integrity Failure": [
                "Primary-key generation or record matching may be unstable.",
                "Duplicate ingestion or join fan-out may be present.",
            ],
            "Constant Feature Group": [
                "The dataset may capture only one operating regime for this feature family.",
                "A configuration default may have overwritten genuine variation.",
            ],
            "Low Information Feature Group": [
                "The feature family may be dominated by defaults or rare-event behavior.",
                "The signal may be too sparse or too imbalanced for downstream use.",
            ],
        }
        return mapping.get(
            evidence.category,
            ["A localized structural or statistical issue may be present."],
        )

    def _validation_steps_for(
        self,
        evidence: EvidenceUnit,
        knowledge: KnowledgeObject,
    ) -> list[str]:
        columns = knowledge.metadata.get("affected_columns", [])
        return [
            f"Inspect the detailed findings and source values for columns {columns}.",
            "Check whether the pattern persists across relevant dataset segments or "
            "time windows.",
            "Confirm whether the observed pattern reflects collection behavior, "
            "preprocessing, or true domain structure.",
        ]

    def _unknowns_for(self, evidence: EvidenceUnit) -> list[str]:
        return [
            "Causal direction cannot be established from these structured observations alone.",
            "Temporal ordering of the observed pattern is not available in the current evidence.",
        ]
