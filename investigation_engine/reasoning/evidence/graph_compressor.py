"""Graph-based evidence compression engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean

import networkx as nx
from loguru import logger

from investigation_engine.config.settings import EvidenceCompressionSettings
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.evidence.community_detection import (
    EvidenceCommunityDetector,
)
from investigation_engine.reasoning.evidence.graph_builder import EvidenceGraphBuilder
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker
from investigation_engine.utils.deterministic import stable_datetime, stable_id


@dataclass(frozen=True, slots=True)
class EvidenceCommunity:
    """Internal representation of a detected evidence community."""

    community_id: str
    finding_ids: list[str]
    findings: list[Finding]
    subgraph: nx.Graph


@dataclass(frozen=True, slots=True)
class GraphCompressionArtifacts:
    """Complete graph compression output for downstream reuse."""

    graph: nx.Graph
    communities: list[list[str]]
    evidence_units: list[EvidenceUnit]


class GraphBasedEvidenceCompressionEngine:
    """Compresses findings into evidence units via graph community detection."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._config = config
        self._builder = EvidenceGraphBuilder(config)
        self._detector = EvidenceCommunityDetector(config)

    def compress(self, findings: list[Finding]) -> list[EvidenceUnit]:
        return self.build_artifacts(findings).evidence_units

    def build_artifacts(self, findings: list[Finding]) -> GraphCompressionArtifacts:
        if not findings:
            return GraphCompressionArtifacts(
                graph=nx.Graph(),
                communities=[],
                evidence_units=[],
            )

        graph = self._builder.build_graph(findings)
        communities = self._detector.detect(graph)
        evidence_units = [
            self._community_to_evidence_unit(
                EvidenceCommunity(
                    community_id=f"community_{index + 1:04d}",
                    finding_ids=community_ids,
                    findings=[graph.nodes[node_id]["finding"] for node_id in community_ids],
                    subgraph=graph.subgraph(community_ids).copy(),
                )
            )
            for index, community_ids in enumerate(communities)
        ]
        evidence_units.sort(key=lambda unit: (-unit.strength, -unit.community_strength, unit.title))
        logger.info(
            "GraphBasedEvidenceCompressionEngine: compressed {} findings into {} evidence units",
            len(findings),
            len(evidence_units),
        )
        return GraphCompressionArtifacts(
            graph=graph,
            communities=communities,
            evidence_units=evidence_units,
        )

    @property
    def builder(self) -> EvidenceGraphBuilder:
        return self._builder

    @property
    def detector(self) -> EvidenceCommunityDetector:
        return self._detector

    def _community_to_evidence_unit(self, community: EvidenceCommunity) -> EvidenceUnit:
        findings = community.findings
        columns = self._union_columns(findings)
        rows = self._union_rows(findings)
        category = self._infer_category(findings)
        representative_columns = self._representative_columns(community.subgraph, findings)
        semantic_label = self._semantic_label(findings, representative_columns or columns)
        title = self._title_for(category, semantic_label)
        explanation = self._merge_explanation(community.subgraph)
        structural_similarity = self._average_edge_metric(
            community.subgraph,
            "structural_similarity",
        )
        statistical_similarity = self._average_edge_metric(
            community.subgraph,
            "statistical_similarity",
        )
        semantic_similarity = self._average_edge_metric(
            community.subgraph,
            "semantic_similarity",
        )
        summary = self._summary_for(category, semantic_label, findings, representative_columns)
        confidence = round(mean(finding.confidence for finding in findings), 2)
        community_strength = round(self._community_strength(community.subgraph), 4)
        severity = self._max_severity(findings)
        strength = round(min(
            100.0,
            severity.numeric_weight * 55.0 + confidence * 25.0 + community_strength * 20.0,
        ), 1)
        source_categories = sorted({
            str(finding.metadata.get("category", "unknown")) for finding in findings
        })
        source_modules = sorted({finding.module for finding in findings})
        provenance = {
            **ProvenanceTracker.for_findings(findings),
            "community_id": community.community_id,
            "community_algorithm": self._config.algorithm,
            "graph_node_count": community.subgraph.number_of_nodes(),
            "graph_edge_count": community.subgraph.number_of_edges(),
            "structural_similarity": structural_similarity,
            "statistical_similarity": statistical_similarity,
            "semantic_similarity": semantic_similarity,
        }

        evidence_id = stable_id(
            "evidence",
            {
                "category": category,
                "findings": [finding.id for finding in findings],
                "columns": columns,
            },
        )
        return EvidenceUnit(
            evidence_id=evidence_id,
            category=category,
            title=title,
            summary=summary,
            supporting_findings=[finding.id for finding in findings],
            community_id=community.community_id,
            provenance=provenance,
            confidence=confidence,
            strength=strength,
            community_strength=community_strength,
            structural_similarity=structural_similarity,
            statistical_similarity=statistical_similarity,
            semantic_similarity=semantic_similarity,
            merge_explanation=explanation,
            affected_columns=columns,
            representative_columns=representative_columns,
            affected_rows=rows,
            metadata={
                "compression_engine": "graph_based",
                "community_algorithm": self._config.algorithm,
                "merge_explanation": explanation,
                "source_categories": source_categories,
                "source_modules": source_modules,
                "semantic_label": semantic_label,
                "concept_key": self._concept_key(representative_columns or columns, semantic_label),
                "max_severity": severity.value,
                "graph_density": self._graph_density(community.subgraph),
                "community_size": len(findings),
            },
            created_at=stable_datetime(
                "evidence",
                {
                    "evidence_id": evidence_id,
                    "community_id": community.community_id,
                },
            ),
        )

    def _infer_category(self, findings: list[Finding]) -> str:
        categories = {str(finding.metadata.get("category", "unknown")) for finding in findings}
        if "identifiers" in categories:
            return "Identifier Integrity Failure"
        if "duplicate_features" in categories or "redundant_features" in categories:
            return "Redundant Feature Group"
        if "missing_values" in categories or "missingness_relationships" in categories:
            return "Systematic Missingness"
        if "constant_features" in categories:
            return "Constant Feature Group"
        if "near_constant_features" in categories or "cardinality" in categories:
            return "Low Information Feature Group"
        if "datatype_integrity" in categories:
            return "Configuration Artifact"
        if any("outlier" in category for category in categories):
            return "Outlier Community"
        if any("cluster" in category for category in categories):
            return "Cluster Community"
        return "Correlation Community"

    def _summary_for(
        self,
        category: str,
        semantic_label: str,
        findings: list[Finding],
        representative_columns: list[str],
    ) -> str:
        return (
            f"{semantic_label} were compressed into one {category.lower()} from "
            f"{len(findings)} finding(s). Representative columns: "
            f"{representative_columns or ['dataset-wide']}."
        )

    def _title_for(self, category: str, semantic_label: str) -> str:
        mapping = {
            "Correlation Community": f"{semantic_label} form a correlation community",
            "Redundant Feature Group": f"{semantic_label} contain redundant features",
            "Systematic Missingness": f"{semantic_label} appear systematically unavailable",
            "Identifier Integrity Failure": f"{semantic_label} may compromise record linkage",
            "Constant Feature Group": f"{semantic_label} contain constant features",
            "Configuration Artifact": f"{semantic_label} suggest a configuration artifact",
            "Low Information Feature Group": f"{semantic_label} provide limited information",
            "Outlier Community": f"{semantic_label} show related outlier behavior",
            "Cluster Community": f"{semantic_label} exhibit clustered behavior",
        }
        return mapping.get(category, f"{semantic_label} deserve investigation")

    def _semantic_label(self, findings: list[Finding], columns: list[str]) -> str:
        family = None
        for finding in findings:
            family = self._builder.scorer.feature_family(finding)
            if family is not None:
                break
        if family is not None:
            return family.replace("_", " ").title() + " Measurements"
        if columns:
            return columns[0].replace("_", " ").title() + " Feature Set"
        return "Analytical Evidence"

    def _merge_explanation(self, subgraph: nx.Graph) -> list[str]:
        reasons = Counter()
        for _left_id, _right_id, edge_data in subgraph.edges(data=True):
            for reason in edge_data.get("reasons", []):
                signal_name = reason.split("=")[0]
                reasons[signal_name] += 1
        if not reasons:
            return [
                "Merged because the finding was isolated and formed its own "
                "evidence community."
            ]
        return [
            f"Merged because of strong {reason.replace('_', ' ')} similarity."
            for reason, _count in reasons.most_common(4)
        ]

    def _representative_columns(
        self,
        subgraph: nx.Graph,
        findings: list[Finding],
    ) -> list[str]:
        frequency = Counter(
            column
            for finding in findings
            for column in finding.affected_columns
        )
        if frequency:
            return [column for column, _count in frequency.most_common(5)]

        centrality = nx.degree_centrality(subgraph) if subgraph.number_of_nodes() > 1 else {}
        ranked_findings = sorted(
            findings,
            key=lambda finding: (-centrality.get(finding.id, 0.0), finding.id),
        )
        for finding in ranked_findings:
            if finding.affected_columns:
                return finding.affected_columns[:5]
        return []

    @staticmethod
    def _community_strength(subgraph: nx.Graph) -> float:
        if subgraph.number_of_nodes() <= 1:
            return 1.0
        weights = [
            float(edge_data.get("weight", 0.0))
            for *_rest, edge_data in subgraph.edges(data=True)
        ]
        if not weights:
            return 0.0
        density = nx.density(subgraph)
        return min(1.0, mean(weights) * density)

    @staticmethod
    def _graph_density(subgraph: nx.Graph) -> float:
        if subgraph.number_of_nodes() <= 1:
            return 0.0
        return float(nx.density(subgraph))

    @staticmethod
    def _average_edge_metric(subgraph: nx.Graph, metric: str) -> float:
        values = [
            float(edge_data.get(metric, 0.0))
            for *_rest, edge_data in subgraph.edges(data=True)
        ]
        return round(mean(values), 4) if values else 0.0

    @staticmethod
    def _union_columns(findings: list[Finding]) -> list[str]:
        return sorted({column for finding in findings for column in finding.affected_columns})

    @staticmethod
    def _union_rows(findings: list[Finding]) -> list[int] | None:
        rows = sorted({
            row
            for finding in findings
            for row in (finding.affected_rows or [])
        })
        return rows or None

    @staticmethod
    def _max_severity(findings: list[Finding]) -> Severity:
        return max(findings, key=lambda finding: finding.severity.numeric_weight).severity

    @staticmethod
    def _concept_key(columns: list[str], semantic_label: str) -> str:
        if columns:
            return columns[0].lower()
        return semantic_label.lower().replace(" ", "_")
