"""Benchmarking utilities for evidence compression engines."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from statistics import mean
import tracemalloc

import networkx as nx

from investigation_engine.config.settings import EvidenceCompressionSettings
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.reasoning.evidence.community_detection import (
    EvidenceCommunityDetector,
)
from investigation_engine.reasoning.evidence.graph_builder import EvidenceGraphBuilder
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reports.exporters import latex_table, markdown_table


@dataclass(frozen=True, slots=True)
class CompressionBenchmarkReport:
    """Benchmark summary for one compression engine."""

    engine_name: str
    runtime_seconds: float
    evidence_unit_count: int
    compression_ratio: float
    redundancy_reduction: float
    average_findings_per_unit: float
    average_community_size: float
    largest_community: int
    graph_density: float
    community_modularity: float
    reasoning_compression_factor: float
    memory_usage_bytes: int
    provenance_preserved: bool
    deterministic: bool


@dataclass(frozen=True, slots=True)
class ReasoningBenchmarkMetrics:
    """Compression and abstraction metrics across the reasoning pipeline."""

    semantic_compression_ratio: float
    knowledge_reduction_ratio: float
    hypothesis_compression_ratio: float
    investigation_compression_ratio: float
    reasoning_depth: int
    average_provenance_length: float
    average_confidence: float


class EvidenceCompressionBenchmark:
    """Compares compression engines on shared findings."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._config = config
        self._graph_builder = EvidenceGraphBuilder(config)
        self._detector = EvidenceCommunityDetector(config)

    def benchmark(
        self,
        findings: list[Finding],
        engines: dict[str, object],
        *,
        graph: nx.Graph | None = None,
        communities: list[list[str]] | None = None,
        graph_density: float | None = None,
        community_modularity: float | None = None,
    ) -> dict[str, CompressionBenchmarkReport]:
        evidence_graph = graph
        detected_communities = communities
        if evidence_graph is None and (graph_density is None or community_modularity is None):
            evidence_graph = self._graph_builder.build_graph(findings)
        if detected_communities is None and evidence_graph is not None and community_modularity is None:
            detected_communities = self._detector.detect(evidence_graph)

        modularity_score = (
            community_modularity
            if community_modularity is not None
            else self._detector.modularity_score(
                evidence_graph if evidence_graph is not None else nx.Graph(),
                detected_communities or [],
            )
        )
        density = (
            graph_density
            if graph_density is not None
            else float(nx.density(evidence_graph)) if evidence_graph is not None and evidence_graph.number_of_nodes() > 1 else 0.0
        )

        reports: dict[str, CompressionBenchmarkReport] = {}
        for engine_name, engine in engines.items():
            runtimes: list[float] = []
            memory_usages: list[int] = []
            evidence_units: list[EvidenceUnit] = []
            for _ in range(self._config.benchmark_repeat_runs):
                tracemalloc.start()
                start = time.perf_counter()
                evidence_units = engine.compress(findings)
                runtimes.append(time.perf_counter() - start)
                _current, peak = tracemalloc.get_traced_memory()
                memory_usages.append(int(peak))
                tracemalloc.stop()

            unit_count = len(evidence_units)
            compression_ratio = round(unit_count / max(len(findings), 1), 4)
            community_sizes = [len(unit.supporting_findings) for unit in evidence_units]
            reports[engine_name] = CompressionBenchmarkReport(
                engine_name=engine_name,
                runtime_seconds=round(mean(runtimes), 6),
                evidence_unit_count=unit_count,
                compression_ratio=compression_ratio,
                redundancy_reduction=round(1.0 - compression_ratio, 4),
                average_findings_per_unit=round(
                    mean(len(unit.supporting_findings) for unit in evidence_units),
                    4,
                ) if evidence_units else 0.0,
                average_community_size=round(
                    mean(len(unit.supporting_findings) for unit in evidence_units),
                    4,
                ) if evidence_units else 0.0,
                largest_community=max(community_sizes, default=0),
                graph_density=round(density, 6),
                community_modularity=round(modularity_score, 6),
                reasoning_compression_factor=round(
                    len(findings) / max(unit_count, 1),
                    4,
                ),
                memory_usage_bytes=int(mean(memory_usages)) if memory_usages else 0,
                provenance_preserved=self._provenance_preserved(findings, evidence_units),
                deterministic=self._deterministic(engine, findings),
            )
        return reports

    def reasoning_metrics(
        self,
        findings: list[Finding],
        evidence_units: list[EvidenceUnit],
        knowledge_objects: list[KnowledgeObject],
        hypotheses: list[Hypothesis],
        investigations: list[Investigation],
    ) -> ReasoningBenchmarkMetrics:
        semantic_compression_ratio = round(
            len(evidence_units) / max(len(findings), 1),
            4,
        )
        knowledge_reduction_ratio = round(
            len(knowledge_objects) / max(len(evidence_units), 1),
            4,
        )
        hypothesis_compression_ratio = round(
            len(hypotheses) / max(len(knowledge_objects), 1),
            4,
        )
        investigation_compression_ratio = round(
            len(investigations) / max(len(hypotheses), 1),
            4,
        )
        provenance_lengths = [
            len(investigation.provenance.get("finding_ids", []))
            + len(investigation.provenance.get("evidence_ids", []))
            + len(investigation.provenance.get("knowledge_ids", []))
            + len(investigation.provenance.get("hypothesis_ids", []))
            for investigation in investigations
        ]
        average_provenance_length = (
            round(mean(provenance_lengths), 4)
            if provenance_lengths else 0.0
        )

        return ReasoningBenchmarkMetrics(
            semantic_compression_ratio=semantic_compression_ratio,
            knowledge_reduction_ratio=knowledge_reduction_ratio,
            hypothesis_compression_ratio=hypothesis_compression_ratio,
            investigation_compression_ratio=investigation_compression_ratio,
            reasoning_depth=4,
            average_provenance_length=average_provenance_length,
            average_confidence=round(
                mean(investigation.confidence for investigation in investigations)
                if investigations
                else 0.0,
                4,
            ),
        )

    @staticmethod
    def _provenance_preserved(findings: list[Finding], evidence_units: list[EvidenceUnit]) -> bool:
        original_ids = {finding.id for finding in findings}
        compressed_ids = {
            finding_id
            for unit in evidence_units
            for finding_id in unit.supporting_findings
        }
        return original_ids == compressed_ids

    @staticmethod
    def _deterministic(engine: object, findings: list[Finding]) -> bool:
        first = engine.compress(findings)
        second = engine.compress(findings)
        first_signature = [
            (
                unit.evidence_id,
                unit.title,
                tuple(unit.supporting_findings),
            )
            for unit in first
        ]
        second_signature = [
            (
                unit.evidence_id,
                unit.title,
                tuple(unit.supporting_findings),
            )
            for unit in second
        ]
        return first_signature == second_signature

    @staticmethod
    def to_json(reports: dict[str, CompressionBenchmarkReport]) -> dict[str, dict[str, object]]:
        return {name: asdict(report) for name, report in reports.items()}

    @staticmethod
    def to_rows(reports: dict[str, CompressionBenchmarkReport], *, humanize: bool = False) -> list[dict[str, object]]:
        rows = [asdict(report) for report in reports.values()]
        if not humanize:
            return rows
        labels = {
            "engine_name": "Engine",
            "runtime_seconds": "Runtime (s)",
            "evidence_unit_count": "Evidence Unit Count",
            "compression_ratio": "Compression Ratio",
            "redundancy_reduction": "Redundancy Reduction",
            "average_findings_per_unit": "Average Findings Per Unit",
            "average_community_size": "Average Community Size",
            "largest_community": "Largest Community",
            "graph_density": "Graph Density",
            "community_modularity": "Community Modularity",
            "reasoning_compression_factor": "Reasoning Compression Factor",
            "memory_usage_bytes": "Memory Usage Bytes",
            "provenance_preserved": "Provenance Preservation",
            "deterministic": "Determinism",
        }
        return [
            {labels.get(key, key): value for key, value in row.items()}
            for row in rows
        ]

    @staticmethod
    def to_markdown(reports: dict[str, CompressionBenchmarkReport]) -> str:
        return markdown_table(EvidenceCompressionBenchmark.to_rows(reports, humanize=True))

    @staticmethod
    def to_latex(reports: dict[str, CompressionBenchmarkReport]) -> str:
        return latex_table(
            EvidenceCompressionBenchmark.to_rows(reports),
            caption="Compression benchmark comparison",
            label="tab:compression_benchmark",
        )
