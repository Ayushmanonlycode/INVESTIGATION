"""Investigation prioritization engine."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject


class InvestigationQueue(BaseModel):
    """Prioritized queue of investigations."""

    investigations: list[Investigation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, float | int | str] = Field(default_factory=dict)


class InvestigationPrioritizer:
    """Ranks hypotheses without depending on raw finding count."""

    def prioritize(
        self,
        hypotheses: list[Hypothesis],
        knowledge_objects: list[KnowledgeObject],
        evidence_units: list[EvidenceUnit],
    ) -> InvestigationQueue:
        evidence_lookup = {evidence.evidence_id: evidence for evidence in evidence_units}
        knowledge_lookup = {
            knowledge.knowledge_id: knowledge
            for knowledge in knowledge_objects
        }

        investigations = [
            self._build_investigation(hypothesis, evidence_lookup, knowledge_lookup)
            for hypothesis in hypotheses
        ]
        investigations.sort(key=lambda item: (-item.priority, -item.confidence, item.title))
        return InvestigationQueue(
            investigations=investigations,
            metadata={
                "investigation_count": len(investigations),
                "hypothesis_count": len(hypotheses),
                "evidence_unit_count": len(evidence_units),
            },
        )

    def _build_investigation(
        self,
        hypothesis: Hypothesis,
        evidence_lookup: dict[str, EvidenceUnit],
        knowledge_lookup: dict[str, KnowledgeObject],
    ) -> Investigation:
        supporting = [
            evidence_lookup[evidence_id]
            for evidence_id in hypothesis.supporting_evidence
            if evidence_id in evidence_lookup
        ]
        contradicting = [
            evidence_lookup[evidence_id]
            for evidence_id in hypothesis.contradicting_evidence
            if evidence_id in evidence_lookup
        ]
        evidence_strength = (
            sum(evidence.strength for evidence in supporting) / max(len(supporting), 1)
        )
        semantic_coverage = min(
            1.0,
            len({
                column for evidence in supporting for column in evidence.affected_columns
            }) / 10.0,
        )
        severity_score = max(
            self._severity_weight(str(evidence.metadata.get("max_severity", "low")))
            for evidence in supporting
        ) if supporting else 0.2
        agreement = min(
            1.0,
            len({
                module
                for evidence in supporting
                for module in evidence.metadata.get("source_modules", [])
            }) / 3.0,
        )
        contradiction_penalty = (
            sum(evidence.strength for evidence in contradicting) / max(len(contradicting), 1)
        ) / 100.0 if contradicting else 0.0

        priority = round(max(0.0, min(
            100.0,
            evidence_strength * 0.35
            + hypothesis.confidence * 25.0
            + semantic_coverage * 15.0
            + severity_score * 15.0
            + agreement * 10.0
            - contradiction_penalty * 20.0,
        )), 1)

        covered_columns = len({
            column for evidence in supporting for column in evidence.affected_columns
        })

        explanation = [
            f"Evidence strength averaged {evidence_strength:.1f}.",
            f"Hypothesis confidence is {hypothesis.confidence:.0%}.",
            f"Semantic coverage spans {covered_columns} column(s).",
            f"Independent investigator agreement is {agreement:.0%}.",
        ]
        if contradicting:
            explanation.append(
                f"{len(contradicting)} contradicting evidence unit(s) reduced priority."
            )
        else:
            explanation.append("No contradicting evidence was found for the same concept scope.")

        finding_ids = sorted({
            finding_id
            for evidence in supporting
            for finding_id in evidence.supporting_findings
        })
        affected_columns = sorted({
            column for evidence in supporting for column in evidence.affected_columns
        })
        lead_knowledge_id = str(hypothesis.provenance.get("knowledge_id", ""))
        knowledge = knowledge_lookup.get(lead_knowledge_id)

        return Investigation(
            title=hypothesis.title,
            summary=hypothesis.statement,
            hypothesis=hypothesis.statement,
            supporting_findings=finding_ids,
            supporting_evidence=hypothesis.supporting_evidence,
            supporting_hypotheses=[hypothesis.hypothesis_id],
            contradicting_evidence=hypothesis.contradicting_evidence,
            evidence_score=round(evidence_strength, 1),
            confidence=hypothesis.confidence,
            priority=priority,
            possible_causes=hypothesis.plausible_causes,
            recommended_next_steps=hypothesis.validation_steps,
            priority_explanation=explanation,
            affected_columns=affected_columns,
            provenance={
                "hypothesis_id": hypothesis.hypothesis_id,
                "knowledge_id": lead_knowledge_id,
                "evidence_ids": hypothesis.supporting_evidence,
                "finding_ids": finding_ids,
                "investigators": sorted({
                    module
                    for evidence in supporting
                    for module in evidence.metadata.get("source_modules", [])
                }),
            },
            metadata={
                "reasoning_layer": "investigation_prioritization",
                "knowledge_concept": knowledge.concept if knowledge is not None else None,
                "dominant_category": hypothesis.metadata.get("dominant_category"),
                "semantic_coverage": semantic_coverage,
                "agreement": agreement,
                "contradiction_penalty": contradiction_penalty,
            },
        )

    def _severity_weight(self, severity: str) -> float:
        mapping = {
            Severity.CRITICAL.value: 1.0,
            Severity.HIGH.value: 0.8,
            Severity.MEDIUM.value: 0.5,
            Severity.LOW.value: 0.2,
            Severity.INFO.value: 0.0,
        }
        return mapping.get(severity, 0.2)
