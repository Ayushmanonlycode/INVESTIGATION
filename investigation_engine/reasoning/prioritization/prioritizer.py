"""Investigation prioritization engine."""

from __future__ import annotations

from datetime import UTC, datetime

import networkx as nx
from pydantic import BaseModel, Field
from investigation_engine.config.settings import Settings
from investigation_engine.models.investigation_hypothesis import (
    EvidenceStrengthAssessment,
    Investigation,
    LikelyCause,
)
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.confidence import ConfidencePropagationEngine
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.prioritization.refinement import (
    InvestigationScopeRefiner,
    InvestigationTitleGenerator,
)
from investigation_engine.utils.deterministic import stable_datetime, stable_id


class InvestigationQueue(BaseModel):
    """Prioritized queue of investigations."""

    investigations: list[Investigation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, float | int | str] = Field(default_factory=dict)


class InvestigationPrioritizer:
    """Ranks merged hypotheses as analyst work items."""

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or Settings()
        self._confidence_engine = ConfidencePropagationEngine()
        self._scope_refiner = InvestigationScopeRefiner(
            self._config.engine.investigation_semantic_cohesion_threshold
        )
        self._title_generator = InvestigationTitleGenerator()

    def prioritize(
        self,
        hypotheses: list[Hypothesis],
        knowledge_objects: list[KnowledgeObject],
        evidence_units: list[EvidenceUnit],
    ) -> InvestigationQueue:
        evidence_lookup = {evidence.evidence_id: evidence for evidence in evidence_units}
        knowledge_lookup = {knowledge.knowledge_id: knowledge for knowledge in knowledge_objects}
        hypothesis_lookup = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}

        graph = self._build_investigation_graph(hypotheses)
        communities = [
            sorted(component)
            for component in nx.connected_components(graph)
        ]
        communities.sort(key=lambda component: (component[0], len(component)))
        investigations: list[Investigation] = []
        for community in communities:
            initial_group = [hypothesis_lookup[hypothesis_id] for hypothesis_id in community]
            refined_groups = self._scope_refiner.split(
                initial_group,
                graph.subgraph(community).copy(),
                total_evidence_count=len(evidence_units),
                max_evidence_share=self._config.engine.max_investigation_evidence_share,
            )
            for refined_group in refined_groups:
                refined_ids = [hypothesis.hypothesis_id for hypothesis in refined_group]
                investigations.append(
                    self._build_investigation(
                        refined_group,
                        graph.subgraph(refined_ids).copy(),
                        evidence_lookup,
                        knowledge_lookup,
                    )
                )
        investigations.sort(
            key=lambda item: (-item.priority, -item.confidence, item.title, item.investigation_id)
        )

        return InvestigationQueue(
            investigations=investigations,
            metadata={
                "investigation_count": len(investigations),
                "hypothesis_count": len(hypotheses),
                "evidence_unit_count": len(evidence_units),
            },
        )

    def _build_investigation_graph(self, hypotheses: list[Hypothesis]) -> nx.Graph:
        graph = nx.Graph()
        for hypothesis in hypotheses:
            graph.add_node(hypothesis.hypothesis_id, hypothesis=hypothesis)

        for index, left in enumerate(hypotheses):
            for right in hypotheses[index + 1:]:
                score, reasons = self._hypothesis_link_score(left, right)
                if score < 0.28:
                    continue
                graph.add_edge(
                    left.hypothesis_id,
                    right.hypothesis_id,
                    weight=score,
                    reasons=reasons,
                )

        return graph

    def _hypothesis_link_score(
        self,
        left: Hypothesis,
        right: Hypothesis,
    ) -> tuple[float, list[str]]:
        left_columns = set(left.metadata.get("affected_columns", []))
        right_columns = set(right.metadata.get("affected_columns", []))
        left_concepts = set(left.metadata.get("knowledge_concepts", []))
        right_concepts = set(right.metadata.get("knowledge_concepts", []))
        left_evidence = set(left.supporting_evidence)
        right_evidence = set(right.supporting_evidence)
        left_families = set(left.metadata.get("feature_families", []))
        right_families = set(right.metadata.get("feature_families", []))

        shared_columns = self._jaccard(left_columns, right_columns)
        shared_concepts = self._jaccard(left_concepts, right_concepts)
        shared_evidence = self._jaccard(left_evidence, right_evidence)
        feature_family_similarity = self._jaccard(left_families, right_families)
        confidence_similarity = 1.0 - abs(left.confidence - right.confidence)
        workstream_similarity = 1.0 if self._workstream(left) == self._workstream(right) else 0.0

        score = (
            shared_columns * 0.20
            + shared_concepts * 0.25
            + shared_evidence * 0.15
            + feature_family_similarity * 0.15
            + confidence_similarity * 0.10
            + workstream_similarity * 0.15
        )
        reasons: list[str] = []
        if shared_columns > 0:
            reasons.append("shared columns")
        if shared_concepts > 0:
            reasons.append("shared concepts")
        if shared_evidence > 0:
            reasons.append("shared evidence")
        if feature_family_similarity > 0:
            reasons.append("shared feature families")
        if workstream_similarity > 0:
            reasons.append("shared workstream")
        return score, reasons

    def _build_investigation(
        self,
        hypothesis_group: list[Hypothesis],
        graph: nx.Graph,
        evidence_lookup: dict[str, EvidenceUnit],
        knowledge_lookup: dict[str, KnowledgeObject],
    ) -> Investigation:
        supporting_evidence_ids = sorted({
            evidence_id
            for hypothesis in hypothesis_group
            for evidence_id in hypothesis.supporting_evidence
        })
        supporting_knowledge_ids = sorted({
            knowledge_id
            for hypothesis in hypothesis_group
            for knowledge_id in hypothesis.supporting_knowledge_objects
        })
        supporting_hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in hypothesis_group]
        contradicting_evidence_ids = sorted({
            evidence_id
            for hypothesis in hypothesis_group
            for evidence_id in hypothesis.contradicting_evidence
        })
        unknown_evidence = sorted({
            item
            for hypothesis in hypothesis_group
            for item in hypothesis.unknown_evidence
        })

        supporting = [
            evidence_lookup[evidence_id]
            for evidence_id in supporting_evidence_ids
            if evidence_id in evidence_lookup
        ]
        contradicting = [
            evidence_lookup[evidence_id]
            for evidence_id in contradicting_evidence_ids
            if evidence_id in evidence_lookup
        ]

        title = self._title_generator.generate(hypothesis_group)
        summary = self._investigation_summary(hypothesis_group, supporting)
        hypothesis_statement = " ".join(
            hypothesis.statement for hypothesis in hypothesis_group
        )
        confidence, confidence_breakdown = self._confidence_engine.score(
            supporting,
            contradicting=contradicting,
            unknown_evidence=unknown_evidence,
        )

        evidence_strength = (
            sum(evidence.strength for evidence in supporting) / max(len(supporting), 1)
        )
        average_community_strength = round(
            sum(evidence.community_strength for evidence in supporting) / max(len(supporting), 1),
            4,
        )
        semantic_coverage = min(
            1.0,
            len({
                column for evidence in supporting for column in evidence.affected_columns
            }) / 8.0,
        )
        covered_columns = len({
            column for evidence in supporting for column in evidence.affected_columns
        })
        severity_score = max(
            self._severity_weight(str(evidence.metadata.get("max_severity", "low")))
            for evidence in supporting
        ) if supporting else 0.2
        agreement = min(
            1.0,
            len({
                module
                for evidence in supporting
                for module in evidence.provenance.get("investigators", [])
            }) / 2.0,
        )
        contradiction_penalty = (
            sum(evidence.strength for evidence in contradicting) / max(len(contradicting), 1)
        ) / 100.0 if contradicting else 0.0
        hypothesis_agreement = min(1.0, len(hypothesis_group) / 3.0)
        cohesion = self._scope_refiner.group_cohesion(hypothesis_group, graph)

        priority = round(max(0.0, min(
            100.0,
            evidence_strength * 0.30
            + confidence * 20.0
            + semantic_coverage * 15.0
            + severity_score * 15.0
            + agreement * 10.0
            + hypothesis_agreement * 10.0
            + cohesion * 10.0
            - contradiction_penalty * 20.0,
        )), 1)

        explanation = [
            f"Evidence strength averaged {evidence_strength:.1f}.",
            f"{len(hypothesis_group)} related hypothesis/hypotheses were merged into one "
            "work item.",
            f"Semantic coverage spans {covered_columns} column(s).",
            f"Independent investigator agreement is {agreement:.0%}.",
            f"Semantic cohesion scored {cohesion:.2f}.",
        ]
        if contradicting:
            explanation.append(
                f"{len(contradicting)} contradicting evidence unit(s) reduced priority."
            )
        else:
            explanation.append("No contradicting evidence weakened this work item.")
        if unknown_evidence:
            explanation.append(
                f"{len(unknown_evidence)} unknown evidence gap(s) remain unresolved."
            )

        finding_ids = sorted({
            finding_id
            for evidence in supporting
            for finding_id in evidence.supporting_findings
        })
        affected_columns = sorted({
            column for evidence in supporting for column in evidence.affected_columns
        })
        possible_causes = self._merge_unique_lists(
            hypothesis.plausible_causes for hypothesis in hypothesis_group
        )
        next_steps = self._merge_unique_lists(
            hypothesis.validation_steps for hypothesis in hypothesis_group
        )
        evidence_strength_details = self._build_evidence_strength(
            supporting=supporting,
            contradicting=contradicting,
            unknown_evidence=unknown_evidence,
            cohesion=cohesion,
            average_community_strength=average_community_strength,
        )
        likely_causes = self._build_likely_causes(
            supporting=supporting,
            contradicting=contradicting,
            knowledge_lookup=knowledge_lookup,
            supporting_knowledge_ids=supporting_knowledge_ids,
            unknown_evidence=unknown_evidence,
            investigation_confidence=confidence,
        )
        if likely_causes:
            possible_causes = [cause.cause for cause in likely_causes]

        investigation_id = stable_id(
            "investigation",
            {
                "title": title,
                "hypotheses": supporting_hypothesis_ids,
                "evidence": supporting_evidence_ids,
            },
        )
        return Investigation(
            investigation_id=investigation_id,
            title=title,
            summary=summary,
            hypothesis=hypothesis_statement,
            supporting_findings=finding_ids,
            supporting_evidence=supporting_evidence_ids,
            supporting_hypotheses=supporting_hypothesis_ids,
            supporting_knowledge_objects=supporting_knowledge_ids,
            contradicting_evidence=contradicting_evidence_ids,
            unknown_evidence=unknown_evidence,
            evidence_score=round(evidence_strength, 1),
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            priority=priority,
            possible_causes=possible_causes,
            likely_causes=likely_causes,
            recommended_next_steps=next_steps,
            evidence_strength_details=evidence_strength_details,
            priority_explanation=explanation,
            affected_columns=affected_columns,
            provenance={
                "hypothesis_ids": supporting_hypothesis_ids,
                "knowledge_ids": supporting_knowledge_ids,
                "evidence_ids": supporting_evidence_ids,
                "finding_ids": finding_ids,
                "investigators": sorted({
                    module
                    for evidence in supporting
                    for module in evidence.provenance.get("investigators", [])
                }),
                "investigation_graph_density": (
                    round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0
                ),
                "semantic_cohesion": cohesion,
            },
            metadata={
                "reasoning_layer": "decision_layer",
                "knowledge_concepts": [
                    knowledge_lookup[knowledge_id].concept
                    for knowledge_id in supporting_knowledge_ids
                    if knowledge_id in knowledge_lookup
                ],
                "dominant_columns": affected_columns[:3],
                "workstream": self._workstream(hypothesis_group[0]),
                "semantic_coverage": semantic_coverage,
                "agreement": agreement,
                "hypothesis_agreement": hypothesis_agreement,
                "semantic_cohesion": cohesion,
                "contradiction_penalty": contradiction_penalty,
                "evidence_strength_rating": evidence_strength_details.rating,
                "creation_explanation": (
                    f"Created by merging {len(hypothesis_group)} related hypothesis/hypotheses "
                    "into one analyst work item."
                ),
            },
            created_at=stable_datetime(
                "investigation",
                {
                    "investigation_id": investigation_id,
                    "supporting_hypotheses": supporting_hypothesis_ids,
                },
            ),
        )

    def _investigation_summary(
        self,
        hypotheses: list[Hypothesis],
        supporting: list[EvidenceUnit],
    ) -> str:
        statements = [hypothesis.statement for hypothesis in hypotheses]
        concepts = sorted({
            concept
            for hypothesis in hypotheses
            for concept in hypothesis.metadata.get("knowledge_concepts", [])
        })
        columns = sorted({
            column
            for hypothesis in hypotheses
            for column in hypothesis.metadata.get("affected_columns", [])
        })
        lead_concepts = ", ".join(concepts[:3]) if concepts else "the observed reasoning pattern"
        lead_columns = ", ".join(columns[:4]) if columns else "the affected features"
        category_counts: dict[str, int] = {}
        for evidence in supporting:
            category_counts[evidence.category] = category_counts.get(evidence.category, 0) + 1
        category_summary = ", ".join(
            f"{count} {category.lower()}"
            for category, count in sorted(
                category_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        )
        return (
            f"{len(supporting)} evidence units ({category_summary or 'mixed evidence'}) connect "
            f"{lead_concepts} to a recurring pattern across {lead_columns}. "
            + " ".join(statements)
        )

    def _build_evidence_strength(
        self,
        *,
        supporting: list[EvidenceUnit],
        contradicting: list[EvidenceUnit],
        unknown_evidence: list[str],
        cohesion: float,
        average_community_strength: float,
    ) -> EvidenceStrengthAssessment:
        supporting_finding_ids = sorted({
            finding_id
            for evidence in supporting
            for finding_id in evidence.supporting_findings
        })
        investigator_count = len({
            module
            for evidence in supporting
            for module in evidence.provenance.get("investigators", [])
        })
        mean_strength = sum(evidence.strength for evidence in supporting) / max(len(supporting), 1)
        if mean_strength >= 75.0 and len(supporting) >= 2 and cohesion >= 0.70:
            rating = "Strong"
        elif mean_strength >= 50.0 and len(supporting) >= 1:
            rating = "Moderate"
        else:
            rating = "Limited"
        if investigator_count >= 3:
            agreement = "High"
        elif investigator_count == 2:
            agreement = "Moderate"
        else:
            agreement = "Low"
        contradiction_level = "None" if not contradicting else "Moderate" if len(contradicting) <= 2 else "High"
        unknown_level = "Low" if len(unknown_evidence) <= 1 else "Moderate" if len(unknown_evidence) <= 3 else "High"
        return EvidenceStrengthAssessment(
            rating=rating,
            supporting_evidence_units=[evidence.evidence_id for evidence in supporting],
            supporting_findings=supporting_finding_ids,
            cross_investigator_agreement=agreement,
            contradictions=contradiction_level,
            unknown_evidence=unknown_level,
            graph_cohesion=round(cohesion, 4),
            community_strength=average_community_strength,
        )

    def _build_likely_causes(
        self,
        *,
        supporting: list[EvidenceUnit],
        contradicting: list[EvidenceUnit],
        knowledge_lookup: dict[str, KnowledgeObject],
        supporting_knowledge_ids: list[str],
        unknown_evidence: list[str],
        investigation_confidence: float,
    ) -> list[LikelyCause]:
        concepts = {
            knowledge_lookup[knowledge_id].concept
            for knowledge_id in supporting_knowledge_ids
            if knowledge_id in knowledge_lookup
        }
        likely_causes: list[LikelyCause] = []
        cause_specs: list[tuple[str, set[str], tuple[str, ...]]] = []
        if {
            "Neighbor Cell Measurement Subsystem",
            "Measurement Availability",
        }.intersection(concepts):
            cause_specs.extend(
                [
                    (
                        "Neighbor-cell telemetry may be collected only under conditional or degraded operating states.",
                        {"Neighbor Cell Measurement Subsystem", "Measurement Availability"},
                        ("missing", "neighbor", "availability"),
                    ),
                    (
                        "The collection or ingestion pipeline may be truncating neighbor-cell reporting fields.",
                        {"Neighbor Cell Measurement Subsystem"},
                        ("missing", "duplicate", "neighbor", "collection"),
                    ),
                ]
            )
        if {"Configuration Parameters", "Feature Engineering Signals"}.intersection(concepts):
            cause_specs.extend(
                [
                    (
                        "Configuration parameters may be exported with little operational variation.",
                        {"Configuration Parameters"},
                        ("constant", "cardinality", "config", "parameter"),
                    ),
                    (
                        "Feature-engineering defaults may be flattening degradation indicators into near-constant values.",
                        {"Feature Engineering Signals"},
                        ("constant", "duplicate", "variance", "information"),
                    ),
                ]
            )
        if "Identifier Integrity" in concepts:
            cause_specs.append(
                (
                    "Primary-key generation, joining, or deduplication logic may be unstable.",
                    {"Identifier Integrity"},
                    ("identifier", "duplicate", "record", "entity"),
                )
            )
        if "Radio Signal Quality" in concepts or any(
            evidence.category == "Correlation Community" for evidence in supporting
        ):
            cause_specs.append(
                (
                    "Multiple telemetry fields may be expressing one latent subsystem rather than independent signals.",
                    {"Radio Signal Quality"},
                    ("correlation", "signal", "radio", "community"),
                )
            )
        if not cause_specs:
            cause_specs.append(
                (
                    "The affected columns likely reflect one coherent subsystem or pipeline behavior.",
                    concepts,
                    tuple(),
                )
            )

        for cause_text, required_concepts, keywords in cause_specs:
            supported_evidence = [
                evidence
                for evidence in supporting
                if (
                    not keywords
                    or any(keyword in evidence.title.lower() for keyword in keywords)
                    or any(keyword in evidence.category.lower() for keyword in keywords)
                    or any(
                        keyword in column.lower()
                        for column in evidence.affected_columns
                        for keyword in keywords
                    )
                )
            ]
            matched_knowledge_ids = [
                knowledge_id
                for knowledge_id in supporting_knowledge_ids
                if knowledge_id in knowledge_lookup
                and (
                    not required_concepts
                    or knowledge_lookup[knowledge_id].concept in required_concepts
                )
            ]
            if not supported_evidence:
                continue
            supporting_columns = {
                column for evidence in supported_evidence for column in evidence.affected_columns
            }
            contradicting_ids = [
                evidence.evidence_id
                for evidence in contradicting
                if supporting_columns.intersection(evidence.affected_columns)
            ]
            supporting_findings = sorted({
                finding_id
                for evidence in supported_evidence
                for finding_id in evidence.supporting_findings
            })
            confidence = round(
                max(
                    0.35,
                    min(
                        0.95,
                        investigation_confidence * 0.7
                        + len(supported_evidence) * 0.08
                        + len(matched_knowledge_ids) * 0.04
                        - len(contradicting_ids) * 0.05,
                    ),
                ),
                2,
            )
            likely_causes.append(
                LikelyCause(
                    cause=cause_text,
                    confidence=confidence,
                    supporting_evidence=[evidence.evidence_id for evidence in supported_evidence],
                    supporting_findings=supporting_findings,
                    knowledge_objects=matched_knowledge_ids,
                    contradicting_evidence=sorted(contradicting_ids),
                    unknown_evidence=unknown_evidence[:2],
                )
            )
        likely_causes.sort(key=lambda item: (-item.confidence, item.cause))
        return likely_causes

    def _workstream(self, hypothesis: Hypothesis) -> str:
        return str(hypothesis.metadata.get("workstream", "measurement_pipeline"))

    @staticmethod
    def _merge_unique_lists(lists: object) -> list[str]:
        merged: list[str] = []
        for values in lists:
            for value in values:
                if value not in merged:
                    merged.append(value)
        return merged

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _severity_weight(self, severity: str) -> float:
        mapping = {
            Severity.CRITICAL.value: 1.0,
            Severity.HIGH.value: 0.8,
            Severity.MEDIUM.value: 0.5,
            Severity.LOW.value: 0.2,
            Severity.INFO.value: 0.0,
        }
        return mapping.get(severity, 0.2)
