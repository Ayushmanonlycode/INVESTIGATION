"""Render complete reasoning lineage trees."""

from __future__ import annotations

from typing import Any

from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation import InvestigationResult
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject


class ProvenanceTreeRenderer:
    """Builds deterministic provenance trees for investigations."""

    def build(self, result: InvestigationResult) -> list[dict[str, Any]]:
        finding_lookup = {finding.id: finding for finding in result.findings}
        evidence_lookup = {evidence.evidence_id: evidence for evidence in result.evidence_units}
        knowledge_lookup = {
            knowledge.knowledge_id: knowledge for knowledge in result.knowledge_objects
        }
        hypothesis_lookup = {
            hypothesis.hypothesis_id: hypothesis for hypothesis in result.hypotheses
        }

        return [
            self._investigation_node(
                investigation,
                hypothesis_lookup,
                knowledge_lookup,
                evidence_lookup,
                finding_lookup,
            )
            for investigation in sorted(
                result.investigations,
                key=lambda item: (-item.priority, -item.confidence, item.title, item.investigation_id),
            )
        ]

    def render_terminal(self, trees: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for tree in trees:
            self._append_lines(tree, lines, depth=0)
            lines.append("")
        return "\n".join(lines).strip()

    def _append_lines(self, node: dict[str, Any], lines: list[str], *, depth: int) -> None:
        indent = "  " * depth
        label = node["level"].replace("_", " ").title()
        title = node.get("title") or node["id"]
        confidence = node.get("confidence")
        if confidence is None:
            header = f"{indent}{label} [{node['id']}] {title}"
        else:
            header = f"{indent}{label} [{node['id']}] {title} (confidence={confidence:.2f})"
        lines.append(header)
        summary = node.get("summary")
        if summary:
            lines.append(f"{indent}  Summary: {summary}")
        supporting_objects = node.get("supporting_objects", [])
        if supporting_objects:
            lines.append(f"{indent}  Supporting: {', '.join(supporting_objects)}")
        contradicting_objects = node.get("contradicting_objects", [])
        if contradicting_objects:
            lines.append(f"{indent}  Contradicting: {', '.join(contradicting_objects)}")
        unknown_evidence = node.get("unknown_evidence", [])
        if unknown_evidence:
            lines.append(f"{indent}  Unknown evidence:")
            for item in unknown_evidence:
                lines.append(f"{indent}    - {item}")
        confidence_breakdown = node.get("confidence_breakdown")
        if isinstance(confidence_breakdown, dict) and confidence_breakdown:
            lines.append(f"{indent}  Confidence breakdown:")
            for key, value in confidence_breakdown.items():
                lines.append(
                    f"{indent}    {key.replace('_', ' ').title()}: {value:+.2f}"
                )
        evidence_explainability = node.get("evidence_explainability")
        if isinstance(evidence_explainability, dict) and evidence_explainability:
            lines.append(f"{indent}  Evidence explainability:")
            for key, value in evidence_explainability.items():
                if isinstance(value, list):
                    lines.append(f"{indent}    {key.replace('_', ' ').title()}:")
                    for item in value:
                        lines.append(f"{indent}      - {item}")
                else:
                    lines.append(f"{indent}    {key.replace('_', ' ').title()}: {value}")
        for child in node.get("children", []):
            self._append_lines(child, lines, depth=depth + 1)

    def _investigation_node(
        self,
        investigation: Any,
        hypothesis_lookup: dict[str, Hypothesis],
        knowledge_lookup: dict[str, KnowledgeObject],
        evidence_lookup: dict[str, EvidenceUnit],
        finding_lookup: dict[str, Finding],
    ) -> dict[str, Any]:
        children = [
            self._hypothesis_node(
                hypothesis_lookup[hypothesis_id],
                knowledge_lookup,
                evidence_lookup,
                finding_lookup,
            )
            for hypothesis_id in sorted(investigation.supporting_hypotheses)
            if hypothesis_id in hypothesis_lookup
        ]
        return {
            "level": "investigation",
            "id": investigation.investigation_id,
            "title": investigation.title,
            "confidence": investigation.confidence,
            "summary": investigation.summary,
            "affected_columns": sorted(investigation.affected_columns),
            "supporting_objects": sorted(investigation.supporting_hypotheses),
            "contradicting_objects": sorted(investigation.contradicting_evidence),
            "unknown_evidence": sorted(investigation.unknown_evidence),
            "confidence_breakdown": investigation.confidence_breakdown,
            "likely_causes": [cause.model_dump(mode="json") for cause in investigation.likely_causes],
            "evidence_strength": (
                investigation.evidence_strength_details.model_dump(mode="json")
                if investigation.evidence_strength_details is not None
                else None
            ),
            "provenance": investigation.provenance,
            "children": children,
        }

    def _hypothesis_node(
        self,
        hypothesis: Hypothesis,
        knowledge_lookup: dict[str, KnowledgeObject],
        evidence_lookup: dict[str, EvidenceUnit],
        finding_lookup: dict[str, Finding],
    ) -> dict[str, Any]:
        children = [
            self._knowledge_node(
                knowledge_lookup[knowledge_id],
                evidence_lookup,
                finding_lookup,
            )
            for knowledge_id in sorted(hypothesis.supporting_knowledge_objects)
            if knowledge_id in knowledge_lookup
        ]
        return {
            "level": "hypothesis",
            "id": hypothesis.hypothesis_id,
            "title": hypothesis.title,
            "confidence": hypothesis.confidence,
            "summary": hypothesis.statement,
            "affected_columns": sorted(hypothesis.metadata.get("affected_columns", [])),
            "supporting_objects": sorted(hypothesis.supporting_knowledge_objects),
            "contradicting_objects": sorted(
                set(hypothesis.contradicting_knowledge_objects + hypothesis.contradicting_evidence)
            ),
            "unknown_evidence": sorted(set(hypothesis.unknown_evidence + hypothesis.unknowns)),
            "confidence_breakdown": hypothesis.confidence_breakdown,
            "provenance": hypothesis.provenance,
            "children": children,
        }

    def _knowledge_node(
        self,
        knowledge: KnowledgeObject,
        evidence_lookup: dict[str, EvidenceUnit],
        finding_lookup: dict[str, Finding],
    ) -> dict[str, Any]:
        children = [
            self._evidence_node(evidence_lookup[evidence_id], finding_lookup)
            for evidence_id in sorted(knowledge.supporting_evidence)
            if evidence_id in evidence_lookup
        ]
        return {
            "level": "knowledge_object",
            "id": knowledge.knowledge_id,
            "title": knowledge.concept,
            "confidence": knowledge.confidence,
            "summary": knowledge.description,
            "affected_columns": sorted(knowledge.metadata.get("affected_columns", [])),
            "supporting_objects": sorted(knowledge.supporting_evidence),
            "contradicting_objects": [],
            "unknown_evidence": [],
            "related_concepts": sorted(knowledge.related_concepts),
            "provenance": knowledge.provenance,
            "children": children,
        }

    def _evidence_node(
        self,
        evidence: EvidenceUnit,
        finding_lookup: dict[str, Finding],
    ) -> dict[str, Any]:
        children = [
            self._finding_node(finding_lookup[finding_id])
            for finding_id in sorted(evidence.supporting_findings)
            if finding_id in finding_lookup
        ]
        return {
            "level": "evidence_unit",
            "id": evidence.evidence_id,
            "title": evidence.title,
            "confidence": evidence.confidence,
            "summary": evidence.summary,
            "category": evidence.category,
            "strength": evidence.strength,
            "affected_columns": sorted(evidence.affected_columns),
            "community_id": evidence.community_id,
            "supporting_objects": sorted(evidence.supporting_findings),
            "contradicting_objects": [],
            "unknown_evidence": [],
            "evidence_explainability": {
                "structural_similarity": f"{evidence.structural_similarity:.4f}",
                "statistical_similarity": f"{evidence.statistical_similarity:.4f}",
                "semantic_similarity": f"{evidence.semantic_similarity:.4f}",
                "community_strength": f"{evidence.community_strength:.4f}",
                "merge_explanation": evidence.merge_explanation,
            },
            "provenance": evidence.provenance,
            "children": children,
        }

    def _finding_node(self, finding: Finding) -> dict[str, Any]:
        investigator_node = {
            "level": "investigator",
            "id": finding.module,
            "title": finding.module.replace("_", " ").title(),
            "confidence": None,
            "summary": "Source investigator contributing this finding.",
            "children": [],
        }
        return {
            "level": "finding",
            "id": finding.id,
            "title": finding.title,
            "confidence": finding.confidence,
            "summary": finding.description,
            "supporting_objects": [finding.module],
            "contradicting_objects": [],
            "unknown_evidence": [],
            "children": [investigator_node],
        }
