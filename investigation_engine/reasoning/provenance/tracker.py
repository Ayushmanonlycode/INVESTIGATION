"""Provenance tracker for deterministic lineage preservation."""

from __future__ import annotations

from typing import Any

from investigation_engine.models.finding import Finding
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject


class ProvenanceTracker:
    """Builds explicit lineage chains across the reasoning pipeline."""

    @staticmethod
    def for_findings(findings: list[Finding]) -> dict[str, Any]:
        return {
            "finding_ids": [finding.id for finding in findings],
            "investigators": sorted({finding.module for finding in findings}),
            "categories": sorted({
                str(finding.metadata.get("category", "unknown")) for finding in findings
            }),
        }

    @staticmethod
    def for_evidence(
        evidence_units: list[EvidenceUnit],
        finding_lookup: dict[str, Finding | None],
    ) -> dict[str, Any]:
        finding_ids = sorted({
            finding_id
            for evidence in evidence_units
            for finding_id in evidence.supporting_findings
        })
        investigators = sorted({
            finding_lookup[finding_id].module
            for finding_id in finding_ids
            if finding_id in finding_lookup and finding_lookup[finding_id] is not None
        })
        return {
            "evidence_ids": [evidence.evidence_id for evidence in evidence_units],
            "finding_ids": finding_ids,
            "investigators": investigators,
        }

    @staticmethod
    def for_knowledge(
        knowledge: KnowledgeObject,
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> dict[str, Any]:
        evidence_ids = knowledge.supporting_evidence
        finding_ids = sorted({
            finding_id
            for evidence_id in evidence_ids
            for finding_id in evidence_lookup[evidence_id].supporting_findings
            if evidence_id in evidence_lookup
        })
        investigators = sorted({
            investigator
            for evidence_id in evidence_ids
            for investigator in evidence_lookup[evidence_id].provenance.get("investigators", [])
            if evidence_id in evidence_lookup
        })
        return {
            "knowledge_id": knowledge.knowledge_id,
            "evidence_ids": evidence_ids,
            "finding_ids": finding_ids,
            "investigators": investigators,
        }

    @staticmethod
    def for_hypothesis(
        hypothesis: Hypothesis,
        knowledge: KnowledgeObject,
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> dict[str, Any]:
        knowledge_provenance = ProvenanceTracker.for_knowledge(knowledge, evidence_lookup)
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "knowledge_id": knowledge.knowledge_id,
            **knowledge_provenance,
            "supporting_evidence": hypothesis.supporting_evidence,
            "contradicting_evidence": hypothesis.contradicting_evidence,
        }
