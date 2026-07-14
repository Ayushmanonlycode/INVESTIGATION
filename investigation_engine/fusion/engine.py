"""Evidence Fusion Engine — compatibility adapter over the reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
import networkx as nx

from loguru import logger

from investigation_engine.config.settings import Settings
from investigation_engine.models.finding import Finding
from investigation_engine.models.investigation_hypothesis import Investigation
from investigation_engine.reasoning.evidence.compression import EvidenceCompressionEngine
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.generator import HypothesisGenerationEngine
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.constructor import KnowledgeConstructionEngine
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.prioritization.prioritizer import (
    InvestigationPrioritizer,
    InvestigationQueue,
)
from investigation_engine.utils.deterministic import stable_id


@dataclass(slots=True)
class ReasoningArtifacts:
    """All intermediate outputs of the reasoning pipeline."""

    evidence_units: list[EvidenceUnit]
    evidence_graph: nx.Graph
    evidence_communities: list[list[str]]
    knowledge_objects: list[KnowledgeObject]
    hypotheses: list[Hypothesis]
    investigation_queue: InvestigationQueue
    layer_timings: dict[str, float]


class EvidenceFusionEngine:
    """Backward-compatible facade for the multi-layer reasoning system."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()
        self._compression = EvidenceCompressionEngine(self.config)
        self._knowledge = KnowledgeConstructionEngine()
        self._hypotheses = HypothesisGenerationEngine()
        self._prioritizer = InvestigationPrioritizer(self.config)

    def build_reasoning_artifacts(self, findings: list[Finding]) -> ReasoningArtifacts:
        """Run the full reasoning pipeline and return all intermediate artifacts."""
        filtered_findings = [
            finding
            for finding in findings
            if finding.metadata.get("category") != "structural_health_score"
        ]
        summary_findings = [
            finding
            for finding in findings
            if finding.metadata.get("category") == "structural_health_score"
        ]

        timings: dict[str, float] = {}

        started = time.perf_counter()
        compression_artifacts = self._compression.build_artifacts(filtered_findings)
        evidence_units = compression_artifacts.evidence_units
        timings["evidence_compression"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        knowledge_objects = self._knowledge.construct(evidence_units)
        timings["knowledge_construction"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        hypotheses = self._hypotheses.generate(knowledge_objects, evidence_units)
        timings["hypothesis_generation"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        investigation_queue = self._prioritizer.prioritize(
            hypotheses,
            knowledge_objects,
            evidence_units,
        )
        timings["investigation_prioritization"] = round(time.perf_counter() - started, 6)

        if summary_findings:
            investigation_queue.investigations.extend(
                self._wrap_summary_findings(summary_findings)
            )
            investigation_queue.investigations.sort(key=lambda item: -item.priority)

        logger.info(
            "EvidenceFusionEngine: {} findings -> {} evidence units -> {} knowledge objects -> "
            "{} hypotheses -> {} investigations",
            len(findings),
            len(evidence_units),
            len(knowledge_objects),
            len(hypotheses),
            len(investigation_queue.investigations),
        )
        return ReasoningArtifacts(
            evidence_units=evidence_units,
            evidence_graph=compression_artifacts.graph,
            evidence_communities=compression_artifacts.communities,
            knowledge_objects=knowledge_objects,
            hypotheses=hypotheses,
            investigation_queue=investigation_queue,
            layer_timings=timings,
        )

    def fuse_findings(self, findings: list[Finding]) -> list[Investigation]:
        """Compatibility wrapper returning only prioritized investigations."""
        return self.build_reasoning_artifacts(findings).investigation_queue.investigations

    def _wrap_summary_findings(self, findings: list[Finding]) -> list[Investigation]:
        investigations: list[Investigation] = []
        for finding in findings:
            investigations.append(Investigation(
                investigation_id=stable_id(
                    "investigation",
                    {"summary_finding_id": finding.id, "title": finding.title},
                ),
                title=finding.title,
                summary=finding.description,
                hypothesis="Unified structural integrity assessment summary.",
                supporting_findings=[finding.id],
                evidence_score=round(finding.severity.numeric_weight * 100.0, 1),
                confidence=finding.confidence,
                priority=0.0,
                possible_causes=[
                    "Composite structural integrity summary from investigator output."
                ],
                recommended_next_steps=[finding.recommendation],
                priority_explanation=[
                    "Priority is fixed at zero because this item is a summary, not an "
                    "investigative queue entry."
                ],
                affected_columns=finding.affected_columns,
                provenance={
                    "finding_ids": [finding.id],
                    "investigators": [finding.module],
                },
                metadata={
                    "reasoning_layer": "summary_passthrough",
                    "original_category": "structural_health_score",
                },
            ))
        return investigations
