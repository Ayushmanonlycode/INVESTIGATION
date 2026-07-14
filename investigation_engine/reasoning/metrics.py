"""Reusable reasoning metrics engine."""

from __future__ import annotations
import resource

from statistics import mean
from typing import Any

import networkx as nx

from investigation_engine.config.settings import EvidenceCompressionSettings
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.reasoning.evidence.community_detection import EvidenceCommunityDetector
from investigation_engine.reasoning.evidence.graph_builder import EvidenceGraphBuilder
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject


class ReasoningMetricsEngine:
    """Computes deterministic metrics across the reasoning hierarchy."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._builder = EvidenceGraphBuilder(config)
        self._detector = EvidenceCommunityDetector(config)

    def build(
        self,
        findings: list[Finding],
        evidence_units: list[EvidenceUnit],
        knowledge_objects: list[KnowledgeObject],
        hypotheses: list[Hypothesis],
        investigations: list[Investigation],
        *,
        graph: nx.Graph | None = None,
        communities: list[list[str]] | None = None,
        layer_timings: dict[str, float] | None = None,
        total_runtime_seconds: float | None = None,
        dataset_memory_bytes: int | None = None,
    ) -> dict[str, Any]:
        evidence_graph = graph or self._builder.build_graph(findings)
        detected_communities = communities or self._detector.detect(evidence_graph)
        modularity = self._detector.modularity_score(evidence_graph, detected_communities)
        community_sizes = [len(community) for community in detected_communities]
        total_runtime = round(
            total_runtime_seconds
            if total_runtime_seconds is not None
            else sum((layer_timings or {}).values()),
            6,
        )
        process_memory_bytes = self._process_memory_bytes()

        return {
            "counts": {
                "findings": len(findings),
                "evidence_units": len(evidence_units),
                "knowledge_objects": len(knowledge_objects),
                "hypotheses": len(hypotheses),
                "investigations": len(investigations),
            },
            "compression": {
                "findings_to_evidence": round(len(evidence_units) / max(len(findings), 1), 4),
                "evidence_to_knowledge": round(
                    len(knowledge_objects) / max(len(evidence_units), 1),
                    4,
                ),
                "knowledge_to_hypotheses": round(
                    len(hypotheses) / max(len(knowledge_objects), 1),
                    4,
                ),
                "hypotheses_to_investigations": round(
                    len(investigations) / max(len(hypotheses), 1),
                    4,
                ),
                "overall_compression_ratio": round(
                    len(investigations) / max(len(findings), 1),
                    4,
                ),
                "findings_per_evidence_unit": round(
                    len(findings) / max(len(evidence_units), 1),
                    4,
                ),
                "evidence_per_knowledge_object": round(
                    len(evidence_units) / max(len(knowledge_objects), 1),
                    4,
                ),
                "knowledge_objects_per_hypothesis": round(
                    len(knowledge_objects) / max(len(hypotheses), 1),
                    4,
                ),
                "hypotheses_per_investigation": round(
                    len(hypotheses) / max(len(investigations), 1),
                    4,
                ),
                "overall_compression_factor": round(
                    len(findings) / max(len(investigations), 1),
                    4,
                ),
            },
            "graph": {
                "nodes": evidence_graph.number_of_nodes(),
                "edges": evidence_graph.number_of_edges(),
                "communities": len(detected_communities),
                "average_community_size": round(mean(community_sizes), 4) if community_sizes else 0.0,
                "largest_community": max(community_sizes, default=0),
                "density": round(
                    float(nx.density(evidence_graph))
                    if evidence_graph.number_of_nodes() > 1
                    else 0.0,
                    6,
                ),
                "average_degree": round(
                    (
                        sum(degree for _node, degree in evidence_graph.degree())
                        / evidence_graph.number_of_nodes()
                    )
                    if evidence_graph.number_of_nodes() > 0
                    else 0.0,
                    4,
                ),
                "modularity": round(modularity, 6),
            },
            "reasoning": {
                "average_evidence_per_hypothesis": round(
                    mean(len(h.supporting_evidence) for h in hypotheses) if hypotheses else 0.0,
                    4,
                ),
                "average_knowledge_per_hypothesis": round(
                    mean(len(h.supporting_knowledge_objects) for h in hypotheses)
                    if hypotheses
                    else 0.0,
                    4,
                ),
                "average_hypotheses_per_investigation": round(
                    mean(len(i.supporting_hypotheses) for i in investigations)
                    if investigations
                    else 0.0,
                    4,
                ),
                "runtime_per_layer": layer_timings or {},
                "reasoning_depth": 4,
                "average_confidence": round(
                    mean(item.confidence for item in investigations)
                    if investigations
                    else mean(item.confidence for item in hypotheses)
                    if hypotheses
                    else mean(item.confidence for item in evidence_units)
                    if evidence_units
                    else 0.0,
                    4,
                ),
            },
            "runtime": {
                "runtime_per_layer": layer_timings or {},
                "total_runtime_seconds": total_runtime,
            },
            "memory": {
                "dataset_memory_bytes": dataset_memory_bytes or 0,
                "dataset_memory_megabytes": round((dataset_memory_bytes or 0) / (1024 * 1024), 4),
                "process_memory_bytes": process_memory_bytes,
                "process_memory_megabytes": round(process_memory_bytes / (1024 * 1024), 4),
            },
            "dashboard": {
                "overall_compression_factor": round(
                    len(findings) / max(len(investigations), 1),
                    4,
                ),
                "graph_density": round(
                    float(nx.density(evidence_graph))
                    if evidence_graph.number_of_nodes() > 1
                    else 0.0,
                    6,
                ),
                "communities": len(detected_communities),
                "average_confidence": round(
                    mean(item.confidence for item in investigations)
                    if investigations
                    else 0.0,
                    4,
                ),
                "runtime_seconds": total_runtime,
            },
            "reasoning_flow": {
                "stages": [
                    {
                        "label": "Findings",
                        "count": len(findings),
                        "reduction_percent": round((1.0 - (len(evidence_units) / max(len(findings), 1))) * 100, 2) if findings else 0.0,
                        "compression_factor": round(len(findings) / max(len(evidence_units), 1), 4) if evidence_units else float(len(findings) or 0),
                    },
                    {
                        "label": "Evidence Units",
                        "count": len(evidence_units),
                        "reduction_percent": round((1.0 - (len(knowledge_objects) / max(len(evidence_units), 1))) * 100, 2) if evidence_units else 0.0,
                        "compression_factor": round(len(evidence_units) / max(len(knowledge_objects), 1), 4) if knowledge_objects else float(len(evidence_units) or 0),
                    },
                    {
                        "label": "Knowledge Objects",
                        "count": len(knowledge_objects),
                        "reduction_percent": round((1.0 - (len(hypotheses) / max(len(knowledge_objects), 1))) * 100, 2) if knowledge_objects else 0.0,
                        "compression_factor": round(len(knowledge_objects) / max(len(hypotheses), 1), 4) if hypotheses else float(len(knowledge_objects) or 0),
                    },
                    {
                        "label": "Hypotheses",
                        "count": len(hypotheses),
                        "reduction_percent": round((1.0 - (len(investigations) / max(len(hypotheses), 1))) * 100, 2) if hypotheses else 0.0,
                        "compression_factor": round(len(hypotheses) / max(len(investigations), 1), 4) if investigations else float(len(hypotheses) or 0),
                    },
                    {
                        "label": "Investigations",
                        "count": len(investigations),
                        "reduction_percent": 0.0,
                        "compression_factor": 1.0 if investigations else 0.0,
                    },
                ],
                "overall_reduction_percent": round((1.0 - (len(investigations) / max(len(findings), 1))) * 100, 2) if findings else 0.0,
                "overall_compression_factor": round(len(findings) / max(len(investigations), 1), 4) if findings else 0.0,
            },
        }

    @staticmethod
    def _process_memory_bytes() -> int:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_maxrss * 1024)
