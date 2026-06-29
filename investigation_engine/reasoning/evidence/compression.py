"""Evidence compression engine."""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from investigation_engine.fusion.graph import EvidenceGraph
from investigation_engine.models.finding import Finding
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.evidence.rules import (
    BaseEvidenceCompressionRule,
    ClusterCommunityRule,
    ConfigurationArtifactRule,
    ConstantFeatureGroupRule,
    CorrelationCommunityRule,
    GenericEvidenceRule,
    IdentifierIntegrityFailureRule,
    LowInformationFeatureGroupRule,
    OutlierCommunityRule,
    RedundantFeatureGroupRule,
    SystematicMissingnessRule,
)


class EvidenceCompressionEngine:
    """Compresses raw findings into a smaller set of evidence units."""

    def __init__(self) -> None:
        self._graph = EvidenceGraph()
        self._rules: list[BaseEvidenceCompressionRule] = [
            SystematicMissingnessRule(),
            IdentifierIntegrityFailureRule(),
            RedundantFeatureGroupRule(),
            ConstantFeatureGroupRule(),
            LowInformationFeatureGroupRule(),
            ConfigurationArtifactRule(),
            CorrelationCommunityRule(),
            OutlierCommunityRule(),
            ClusterCommunityRule(),
            GenericEvidenceRule(),
        ]

    def compress(self, findings: list[Finding]) -> list[EvidenceUnit]:
        if not findings:
            return []

        self._graph.build_from_findings(findings)
        components = self._graph.get_connected_subgraphs()
        provisional = [self._compress_component(component) for component in components]
        merged = self._merge_related_evidence(provisional)
        logger.info(
            "EvidenceCompressionEngine: compressed {} findings into {} evidence units",
            len(findings),
            len(merged),
        )
        return merged

    def _compress_component(self, findings: list[Finding]) -> EvidenceUnit:
        for rule in self._rules:
            if rule.matches(findings):
                return rule.compress(findings)
        raise RuntimeError("No evidence compression rule matched the findings cluster.")

    def _merge_related_evidence(self, evidence_units: list[EvidenceUnit]) -> list[EvidenceUnit]:
        grouped: dict[tuple[str, str], list[EvidenceUnit]] = defaultdict(list)
        for evidence in evidence_units:
            concept_key = str(evidence.metadata.get("concept_key", evidence.category)).lower()
            grouped[(evidence.category, concept_key)].append(evidence)

        merged_units: list[EvidenceUnit] = []
        for (category, concept_key), units in grouped.items():
            if len(units) == 1:
                merged_units.append(units[0])
                continue

            first = units[0]
            merged_units.append(EvidenceUnit(
                category=category,
                title=first.title,
                summary=(
                    f"{first.summary} Additional related evidence units were merged so the "
                    "reasoning "
                    "layer reflects distinct evidence, not raw detector verbosity."
                ),
                supporting_findings=sorted({
                    finding_id
                    for unit in units
                    for finding_id in unit.supporting_findings
                }),
                provenance={
                    "merged_evidence_ids": [unit.evidence_id for unit in units],
                    "investigators": sorted({
                        investigator
                        for unit in units
                        for investigator in unit.provenance.get("investigators", [])
                    }),
                    "finding_ids": sorted({
                        finding_id
                        for unit in units
                        for finding_id in unit.supporting_findings
                    }),
                    "compression_rule": "merged_related_evidence",
                },
                confidence=round(max(unit.confidence for unit in units), 2),
                strength=round(max(unit.strength for unit in units), 1),
                affected_columns=sorted({
                    column for unit in units for column in unit.affected_columns
                }),
                affected_rows=sorted({
                    row
                    for unit in units
                    for row in (unit.affected_rows or [])
                }) or None,
                metadata={
                    **first.metadata,
                    "concept_key": concept_key,
                    "merged_unit_count": len(units),
                    "source_modules": sorted({
                        module
                        for unit in units
                        for module in unit.metadata.get("source_modules", [])
                    }),
                },
            ))

        return sorted(merged_units, key=lambda unit: (-unit.strength, -unit.confidence, unit.title))
