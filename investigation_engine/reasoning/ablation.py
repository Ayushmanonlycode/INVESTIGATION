"""Ablation utilities for deterministic reasoning-stage comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
import time

import networkx as nx

from investigation_engine.config.settings import Settings
from investigation_engine.fusion.engine import EvidenceFusionEngine
from investigation_engine.models.finding import Finding
from investigation_engine.reasoning.evidence.compression import EvidenceCompressionEngine
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.generator import HypothesisGenerationEngine
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.constructor import KnowledgeConstructionEngine
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.metrics import ReasoningMetricsEngine
from investigation_engine.reasoning.prioritization.prioritizer import InvestigationPrioritizer, InvestigationQueue
from investigation_engine.utils.deterministic import stable_id


@dataclass(frozen=True, slots=True)
class AblationReport:
    variant: str
    runtime_seconds: float
    findings: int
    evidence_units: int
    knowledge_objects: int
    hypotheses: int
    investigations: int
    compression_factor: float
    average_confidence: float
    graph_density: float
    community_count: int
    average_community_size: float
    largest_community: int
    modularity: float
    reasoning_depth: int
    note: str


class AblationEngine:
    """Runs deterministic reasoning-stage ablations without altering the main pipeline."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or Settings()
        self._fusion = EvidenceFusionEngine(self._config)
        self._compression = EvidenceCompressionEngine(self._config)
        self._knowledge = KnowledgeConstructionEngine()
        self._hypotheses = HypothesisGenerationEngine()
        self._prioritizer = InvestigationPrioritizer(self._config)
        self._metrics = ReasoningMetricsEngine(self._config.evidence_compression)

    def run(self, findings: list[Finding]) -> list[AblationReport]:
        findings = [
            finding
            for finding in findings
            if finding.metadata.get("category") != "structural_health_score"
        ]
        variants = {
            "baseline": self._baseline,
            "no_graph_compression": self._no_graph_compression,
            "no_knowledge_layer": self._no_knowledge_layer,
            "no_hypothesis_layer": self._no_hypothesis_layer,
            "no_semantic_splitting": self._no_semantic_splitting,
            "no_confidence_propagation": self._no_confidence_propagation,
        }
        reports: list[AblationReport] = []
        for name, runner in variants.items():
            started = time.perf_counter()
            evidence_units, knowledge_objects, hypotheses, queue, graph, communities, note = runner(findings)
            runtime_seconds = round(time.perf_counter() - started, 6)
            metrics = self._metrics.build(
                findings,
                evidence_units,
                knowledge_objects,
                hypotheses,
                queue.investigations,
                graph=graph,
                communities=communities,
                total_runtime_seconds=runtime_seconds,
            )
            reports.append(AblationReport(
                variant=name,
                runtime_seconds=runtime_seconds,
                findings=len(findings),
                evidence_units=len(evidence_units),
                knowledge_objects=len(knowledge_objects),
                hypotheses=len(hypotheses),
                investigations=len(queue.investigations),
                compression_factor=float(metrics["compression"]["overall_compression_factor"]),
                average_confidence=float(metrics["reasoning"]["average_confidence"]),
                graph_density=float(metrics["graph"]["density"]),
                community_count=int(metrics["graph"]["communities"]),
                average_community_size=float(metrics["graph"]["average_community_size"]),
                largest_community=int(metrics["graph"]["largest_community"]),
                modularity=float(metrics["graph"]["modularity"]),
                reasoning_depth=int(metrics["reasoning"]["reasoning_depth"]),
                note=note,
            ))
        return reports

    @staticmethod
    def to_rows(reports: list[AblationReport]) -> list[dict[str, object]]:
        return [asdict(report) for report in reports]

    def _baseline(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        artifacts = self._fusion.build_reasoning_artifacts(findings)
        return (
            artifacts.evidence_units,
            artifacts.knowledge_objects,
            artifacts.hypotheses,
            artifacts.investigation_queue,
            artifacts.evidence_graph,
            artifacts.evidence_communities,
            "Full reasoning pipeline.",
        )

    def _no_graph_compression(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        evidence_units = [self._finding_to_evidence(finding) for finding in findings]
        knowledge_objects = self._knowledge.construct(evidence_units)
        hypotheses = self._hypotheses.generate(knowledge_objects, evidence_units)
        queue = self._prioritizer.prioritize(hypotheses, knowledge_objects, evidence_units)
        graph = nx.Graph()
        for evidence in evidence_units:
            graph.add_node(evidence.evidence_id, evidence=evidence)
        communities = [[evidence.evidence_id] for evidence in evidence_units]
        return (
            evidence_units,
            knowledge_objects,
            hypotheses,
            queue,
            graph,
            communities,
            "Each finding becomes its own evidence unit.",
        )

    def _no_knowledge_layer(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        artifacts = self._compression.build_artifacts(findings)
        knowledge_objects = [self._evidence_to_knowledge(evidence) for evidence in artifacts.evidence_units]
        hypotheses = self._hypotheses.generate(knowledge_objects, artifacts.evidence_units)
        queue = self._prioritizer.prioritize(hypotheses, knowledge_objects, artifacts.evidence_units)
        return (
            artifacts.evidence_units,
            knowledge_objects,
            hypotheses,
            queue,
            artifacts.graph,
            artifacts.communities,
            "Semantic abstraction is reduced to one knowledge object per evidence unit.",
        )

    def _no_hypothesis_layer(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        artifacts = self._compression.build_artifacts(findings)
        knowledge_objects = self._knowledge.construct(artifacts.evidence_units)
        evidence_lookup = {item.evidence_id: item for item in artifacts.evidence_units}
        hypotheses = [self._knowledge_to_hypothesis(knowledge, evidence_lookup) for knowledge in knowledge_objects]
        queue = self._prioritizer.prioritize(hypotheses, knowledge_objects, artifacts.evidence_units)
        return (
            artifacts.evidence_units,
            knowledge_objects,
            hypotheses,
            queue,
            artifacts.graph,
            artifacts.communities,
            "Each knowledge object becomes its own investigation hypothesis.",
        )

    def _no_semantic_splitting(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        artifacts = self._fusion.build_reasoning_artifacts(findings)
        config = self._config.model_copy(deep=True)
        config.engine = config.engine.model_copy(update={
            "investigation_semantic_cohesion_threshold": 0.0,
            "max_investigation_evidence_share": 1.0,
        })
        prioritizer = InvestigationPrioritizer(config)
        queue = prioritizer.prioritize(
            artifacts.hypotheses,
            artifacts.knowledge_objects,
            artifacts.evidence_units,
        )
        return (
            artifacts.evidence_units,
            artifacts.knowledge_objects,
            artifacts.hypotheses,
            queue,
            artifacts.evidence_graph,
            artifacts.evidence_communities,
            "Investigation group splitting guardrails are disabled.",
        )

    def _no_confidence_propagation(
        self,
        findings: list[Finding],
    ) -> tuple[list[EvidenceUnit], list[KnowledgeObject], list[Hypothesis], InvestigationQueue, nx.Graph, list[list[str]], str]:
        artifacts = self._fusion.build_reasoning_artifacts(findings)
        evidence_lookup = {item.evidence_id: item for item in artifacts.evidence_units}
        hypotheses = []
        for hypothesis in artifacts.hypotheses:
            supporting = [evidence_lookup[evidence_id] for evidence_id in hypothesis.supporting_evidence if evidence_id in evidence_lookup]
            average_confidence = round(mean(item.confidence for item in supporting), 4) if supporting else 0.0
            hypotheses.append(hypothesis.model_copy(update={
                "confidence": average_confidence,
                "confidence_breakdown": {"uniform_average": average_confidence},
            }))
        queue = self._prioritizer.prioritize(hypotheses, artifacts.knowledge_objects, artifacts.evidence_units)
        investigations = []
        for investigation in queue.investigations:
            supporting = [evidence_lookup[evidence_id] for evidence_id in investigation.supporting_evidence if evidence_id in evidence_lookup]
            average_confidence = round(mean(item.confidence for item in supporting), 4) if supporting else 0.0
            investigations.append(investigation.model_copy(update={
                "confidence": average_confidence,
                "confidence_breakdown": {"uniform_average": average_confidence},
            }))
        queue = queue.model_copy(update={"investigations": investigations})
        return (
            artifacts.evidence_units,
            artifacts.knowledge_objects,
            hypotheses,
            queue,
            artifacts.evidence_graph,
            artifacts.evidence_communities,
            "Confidence is reduced to uniform averages over supporting evidence.",
        )

    @staticmethod
    def _finding_to_evidence(finding: Finding) -> EvidenceUnit:
        category = str(finding.metadata.get("category", "standalone_finding")).replace("_", " ").title()
        return EvidenceUnit(
            evidence_id=stable_id("evidence", {"finding_id": finding.id}),
            category=category,
            title=finding.title,
            summary=finding.description,
            supporting_findings=[finding.id],
            community_id=f"community_{finding.id}",
            provenance={
                "finding_ids": [finding.id],
                "investigators": [finding.module],
            },
            confidence=finding.confidence,
            strength=round(finding.severity.numeric_weight * 100.0, 1),
            community_strength=1.0,
            structural_similarity=0.0,
            statistical_similarity=0.0,
            semantic_similarity=0.0,
            merge_explanation=["No graph compression: finding preserved as standalone evidence unit."],
            affected_columns=finding.affected_columns,
            representative_columns=finding.affected_columns[:3],
            affected_rows=finding.affected_rows,
            metadata={
                "compression_engine": "ablation_passthrough",
                "max_severity": finding.severity.value,
                "source_categories": [str(finding.metadata.get("category", "standalone_finding"))],
            },
        )

    @staticmethod
    def _evidence_to_knowledge(evidence: EvidenceUnit) -> KnowledgeObject:
        concept = evidence.title.replace(" form a correlation community", "").replace(" deserve investigation", "")
        return KnowledgeObject(
            knowledge_id=stable_id("knowledge", {"evidence_id": evidence.evidence_id}),
            concept=concept,
            description=evidence.summary,
            supporting_evidence=[evidence.evidence_id],
            knowledge_graph_id=f"knowledge_{evidence.evidence_id}",
            provenance=evidence.provenance,
            confidence=evidence.confidence,
            abstraction_level=0.35,
            metadata={
                "categories": evidence.metadata.get("source_categories", [evidence.category]),
                "affected_columns": evidence.affected_columns,
                "semantic_terms": evidence.representative_columns or evidence.affected_columns,
                "creation_explanation": "Ablation surrogate knowledge object created directly from evidence.",
            },
        )

    @staticmethod
    def _knowledge_to_hypothesis(knowledge: KnowledgeObject, evidence_lookup: dict[str, EvidenceUnit]) -> Hypothesis:
        supporting = [evidence_lookup[evidence_id] for evidence_id in knowledge.supporting_evidence if evidence_id in evidence_lookup]
        confidence = round(mean(item.confidence for item in supporting), 4) if supporting else knowledge.confidence
        return Hypothesis(
            hypothesis_id=stable_id("hypothesis", {"knowledge_id": knowledge.knowledge_id}),
            title=f"{knowledge.concept} hypothesis",
            statement=knowledge.description,
            supporting_evidence=knowledge.supporting_evidence,
            supporting_knowledge_objects=[knowledge.knowledge_id],
            confidence=confidence,
            confidence_breakdown={"uniform_average": confidence},
            plausible_causes=["Ablation surrogate hypothesis inherits the semantic concept directly."],
            validation_steps=["Inspect the supporting evidence without cross-concept grouping."],
            metadata={
                "knowledge_concepts": [knowledge.concept],
                "dominant_category": next(iter(knowledge.metadata.get("categories", ["unknown"])), "unknown"),
                "workstream": "ablation_direct_hypothesis",
                "feature_families": [],
                "affected_columns": knowledge.metadata.get("affected_columns", []),
            },
        )
