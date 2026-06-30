"""Comparative reasoning experiments for publication-oriented reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

from investigation_engine.config.settings import Settings
from investigation_engine.models.finding import Finding
from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.generator import HypothesisGenerationEngine
from investigation_engine.reasoning.knowledge.constructor import KnowledgeConstructionEngine
from investigation_engine.reasoning.metrics import ReasoningMetricsEngine
from investigation_engine.reasoning.prioritization.prioritizer import InvestigationPrioritizer
from investigation_engine.utils.deterministic import stable_id


@dataclass(frozen=True, slots=True)
class MethodComparisonReport:
    method: str
    findings: int
    review_items: int
    evidence_units: int
    knowledge_objects: int
    hypotheses: int
    investigations: int
    runtime_seconds: float
    compression_factor: float
    duplicate_reduction: float
    average_evidence_per_hypothesis: float
    average_knowledge_per_hypothesis: float
    average_confidence: float
    average_community_size: float
    largest_community: int
    graph_density: float
    reasoning_depth: int
    deterministic: bool
    summary: str


class ReasoningMethodComparator:
    """Compares analyst-facing workload across raw, rule-based, and GBEC flows."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or Settings()
        self._knowledge = KnowledgeConstructionEngine()
        self._hypotheses = HypothesisGenerationEngine()
        self._prioritizer = InvestigationPrioritizer(self._config)
        self._metrics = ReasoningMetricsEngine(self._config.evidence_compression)

    def compare(self, findings: list[Finding]) -> list[MethodComparisonReport]:
        actionable_findings = [
            finding
            for finding in findings
            if finding.metadata.get("category") != "structural_health_score"
        ]
        return [
            self._raw_output(actionable_findings),
            self._compressed_output(
                actionable_findings,
                label="Rule-Based Compression",
                engine=RuleBasedEvidenceCompressionEngine(),
                summary="Legacy rule-driven evidence grouping.",
            ),
            self._compressed_output(
                actionable_findings,
                label="GBEC (Graph-Based Evidence Compression)",
                engine=GraphBasedEvidenceCompressionEngine(self._config.evidence_compression),
                summary="Graph-based evidence compression with deterministic community detection.",
            ),
        ]

    @staticmethod
    def to_rows(reports: list[MethodComparisonReport], *, humanize: bool = False) -> list[dict[str, object]]:
        rows = [asdict(report) for report in reports]
        if not humanize:
            return rows
        labels = {
            "method": "Method",
            "findings": "Findings",
            "review_items": "Review Items",
            "evidence_units": "Evidence Units",
            "knowledge_objects": "Knowledge Objects",
            "hypotheses": "Hypotheses",
            "investigations": "Investigations",
            "runtime_seconds": "Runtime (s)",
            "compression_factor": "Compression Factor",
            "duplicate_reduction": "Duplicate Reduction",
            "average_evidence_per_hypothesis": "Average Evidence per Hypothesis",
            "average_knowledge_per_hypothesis": "Average Knowledge per Hypothesis",
            "average_confidence": "Average Confidence",
            "average_community_size": "Average Community Size",
            "largest_community": "Largest Community",
            "graph_density": "Graph Density",
            "reasoning_depth": "Reasoning Depth",
            "deterministic": "Deterministic",
            "summary": "Summary",
        }
        return [
            {labels.get(key, key): value for key, value in row.items()}
            for row in rows
        ]

    def _raw_output(self, findings: list[Finding]) -> MethodComparisonReport:
        count = len(findings)
        return MethodComparisonReport(
            method="Raw Investigator Output",
            findings=count,
            review_items=count,
            evidence_units=count,
            knowledge_objects=count,
            hypotheses=count,
            investigations=count,
            runtime_seconds=0.0,
            compression_factor=1.0 if count else 0.0,
            duplicate_reduction=0.0,
            average_evidence_per_hypothesis=1.0 if count else 0.0,
            average_knowledge_per_hypothesis=1.0 if count else 0.0,
            average_confidence=round(
                sum(finding.confidence for finding in findings) / count,
                4,
            ) if count else 0.0,
            average_community_size=1.0 if count else 0.0,
            largest_community=1 if count else 0,
            graph_density=0.0,
            reasoning_depth=1 if count else 0,
            deterministic=True,
            summary="Direct investigator findings with no compression or reasoning reduction.",
        )

    def _compressed_output(
        self,
        findings: list[Finding],
        *,
        label: str,
        engine: object,
        summary: str,
    ) -> MethodComparisonReport:
        started = time.perf_counter()
        evidence_units = engine.compress(findings)
        knowledge_objects = self._knowledge.construct(evidence_units)
        hypotheses = self._hypotheses.generate(knowledge_objects, evidence_units)
        queue = self._prioritizer.prioritize(hypotheses, knowledge_objects, evidence_units)
        metrics = self._metrics.build(
            findings,
            evidence_units,
            knowledge_objects,
            hypotheses,
            queue.investigations,
        )
        runtime_seconds = round(time.perf_counter() - started, 6)

        return MethodComparisonReport(
            method=label,
            findings=len(findings),
            review_items=len(evidence_units),
            evidence_units=len(evidence_units),
            knowledge_objects=len(knowledge_objects),
            hypotheses=len(hypotheses),
            investigations=len(queue.investigations),
            runtime_seconds=runtime_seconds,
            compression_factor=round(len(findings) / max(len(evidence_units), 1), 4) if findings else 0.0,
            duplicate_reduction=round(1.0 - (len(evidence_units) / max(len(findings), 1)), 4) if findings else 0.0,
            average_evidence_per_hypothesis=float(metrics["reasoning"]["average_evidence_per_hypothesis"]),
            average_knowledge_per_hypothesis=float(metrics["reasoning"]["average_knowledge_per_hypothesis"]),
            average_confidence=float(metrics["reasoning"]["average_confidence"]),
            average_community_size=float(metrics["graph"]["average_community_size"]),
            largest_community=int(metrics["graph"]["largest_community"]),
            graph_density=float(metrics["graph"]["density"]),
            reasoning_depth=int(metrics["reasoning"]["reasoning_depth"]),
            deterministic=self._deterministic(engine, findings),
            summary=summary,
        )

    @staticmethod
    def _deterministic(engine: object, findings: list[Finding]) -> bool:
        first = engine.compress(findings)
        second = engine.compress(findings)
        return ReasoningMethodComparator._signature(first) == ReasoningMethodComparator._signature(second)

    @staticmethod
    def _signature(evidence_units: list[EvidenceUnit]) -> list[tuple[str, str, tuple[str, ...]]]:
        return [
            (
                unit.evidence_id or stable_id("evidence", {"title": unit.title}),
                unit.title,
                tuple(unit.supporting_findings),
            )
            for unit in evidence_units
        ]
