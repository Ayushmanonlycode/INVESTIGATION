"""Knowledge construction engine."""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker


class KnowledgeConstructionEngine:
    """Transforms evidence units into reusable semantic concepts."""

    def construct(self, evidence_units: list[EvidenceUnit]) -> list[KnowledgeObject]:
        if not evidence_units:
            return []

        groups: dict[str, list[EvidenceUnit]] = defaultdict(list)
        for evidence in evidence_units:
            concept_key = str(evidence.metadata.get("concept_key", evidence.evidence_id))
            groups[concept_key].append(evidence)

        knowledge_objects = [self._build_knowledge(group) for group in groups.values()]
        self._attach_related_concepts(knowledge_objects)
        logger.info(
            "KnowledgeConstructionEngine: constructed {} knowledge objects from {} evidence units",
            len(knowledge_objects),
            len(evidence_units),
        )
        return sorted(knowledge_objects, key=lambda item: (-item.confidence, item.concept))

    def _build_knowledge(self, evidence_units: list[EvidenceUnit]) -> KnowledgeObject:
        lead = max(evidence_units, key=lambda evidence: (evidence.strength, evidence.confidence))
        semantic_label = str(lead.metadata.get("semantic_label", lead.title))
        description = (
            f"{semantic_label} is supported by {len(evidence_units)} evidence unit(s) spanning "
            f"categories {sorted({evidence.category for evidence in evidence_units})}."
        )
        confidence = round(
            sum(evidence.confidence for evidence in evidence_units) / len(evidence_units),
            2,
        )
        provenance = ProvenanceTracker.for_evidence(
            evidence_units,
            {
                finding_id: None  # type: ignore[dict-item]
                for evidence in evidence_units
                for finding_id in evidence.supporting_findings
            },
        )
        provenance["concept_key"] = lead.metadata.get("concept_key")

        return KnowledgeObject(
            concept=semantic_label,
            description=description,
            supporting_evidence=[evidence.evidence_id for evidence in evidence_units],
            provenance=provenance,
            confidence=confidence,
            metadata={
                "concept_key": lead.metadata.get("concept_key"),
                "categories": sorted({evidence.category for evidence in evidence_units}),
                "affected_columns": sorted({
                    column for evidence in evidence_units for column in evidence.affected_columns
                }),
            },
        )

    def _attach_related_concepts(self, knowledge_objects: list[KnowledgeObject]) -> None:
        for knowledge in knowledge_objects:
            related: list[str] = []
            columns = set(knowledge.metadata.get("affected_columns", []))
            for other in knowledge_objects:
                if other.knowledge_id == knowledge.knowledge_id:
                    continue
                other_columns = set(other.metadata.get("affected_columns", []))
                if columns and other_columns and columns.intersection(other_columns):
                    related.append(other.concept)
            knowledge.related_concepts = sorted(set(related))

