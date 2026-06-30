"""Multi-evidence hypothesis generation engine."""

from __future__ import annotations

from collections import Counter

import networkx as nx
from loguru import logger
from investigation_engine.reasoning.confidence import ConfidencePropagationEngine

from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker
from investigation_engine.utils.deterministic import stable_datetime, stable_id


class HypothesisGenerationEngine:
    """Generates multi-concept, evidence-backed hypotheses."""

    def __init__(self) -> None:
        self._confidence_engine = ConfidencePropagationEngine()

    def generate(
        self,
        knowledge_objects: list[KnowledgeObject],
        evidence_units: list[EvidenceUnit],
    ) -> list[Hypothesis]:
        if not knowledge_objects:
            return []

        evidence_lookup = {evidence.evidence_id: evidence for evidence in evidence_units}
        knowledge_lookup = {knowledge.knowledge_id: knowledge for knowledge in knowledge_objects}
        graph = self._build_hypothesis_graph(knowledge_objects)
        communities = [
            sorted(component)
            for component in nx.connected_components(graph)
        ]
        communities.sort(key=lambda component: (component[0], len(component)))

        hypotheses = [
            self._build_hypothesis(
                hypothesis_group_id=f"hypothesis_group_{index + 1:04d}",
                knowledge_group=[knowledge_lookup[knowledge_id] for knowledge_id in community],
                graph=graph.subgraph(community).copy(),
                evidence_lookup=evidence_lookup,
            )
            for index, community in enumerate(communities)
        ]
        logger.info(
            "HypothesisGenerationEngine: generated {} hypotheses from {} knowledge objects",
            len(hypotheses),
            len(knowledge_objects),
        )
        return sorted(hypotheses, key=lambda item: (-item.confidence, item.title))

    def _build_hypothesis_graph(self, knowledge_objects: list[KnowledgeObject]) -> nx.Graph:
        graph = nx.Graph()
        for knowledge in knowledge_objects:
            graph.add_node(knowledge.knowledge_id, knowledge=knowledge)

        for index, left in enumerate(knowledge_objects):
            for right in knowledge_objects[index + 1:]:
                score, reasons = self._knowledge_link_score(left, right)
                if score < 0.32:
                    continue
                graph.add_edge(left.knowledge_id, right.knowledge_id, weight=score, reasons=reasons)

        return graph

    def _knowledge_link_score(
        self,
        left: KnowledgeObject,
        right: KnowledgeObject,
    ) -> tuple[float, list[str]]:
        left_columns = set(left.metadata.get("affected_columns", []))
        right_columns = set(right.metadata.get("affected_columns", []))
        left_terms = set(left.metadata.get("semantic_terms", []))
        right_terms = set(right.metadata.get("semantic_terms", []))
        left_categories = set(left.metadata.get("categories", []))
        right_categories = set(right.metadata.get("categories", []))

        shared_columns = self._jaccard(left_columns, right_columns)
        shared_terms = self._jaccard(left_terms, right_terms)
        shared_categories = self._jaccard(left_categories, right_categories)
        related_concept_bonus = 1.0 if right.concept in left.related_concepts else 0.0

        score = (
            shared_columns * 0.25
            + shared_terms * 0.35
            + shared_categories * 0.25
            + related_concept_bonus * 0.15
        )
        reasons: list[str] = []
        if shared_columns > 0:
            reasons.append("shared columns")
        if shared_terms > 0:
            reasons.append("shared semantic terms")
        if shared_categories > 0:
            reasons.append("shared categories")
        if related_concept_bonus > 0:
            reasons.append("related concepts")
        return score, reasons

    def _build_hypothesis(
        self,
        hypothesis_group_id: str,
        knowledge_group: list[KnowledgeObject],
        graph: nx.Graph,
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> Hypothesis:
        supporting_evidence_ids = sorted({
            evidence_id
            for knowledge in knowledge_group
            for evidence_id in knowledge.supporting_evidence
        })
        supporting = [
            evidence_lookup[evidence_id]
            for evidence_id in supporting_evidence_ids
            if evidence_id in evidence_lookup
        ]
        supporting_knowledge_ids = [knowledge.knowledge_id for knowledge in knowledge_group]
        contradicting = self._find_contradictions(
            knowledge_group,
            supporting_evidence_ids,
            evidence_lookup,
        )
        contradicting_knowledge_ids = self._find_contradicting_knowledge_objects(
            knowledge_group,
            contradicting,
        )
        title, statement = self._compose_statement(knowledge_group, supporting)
        unknown_evidence = self._unknown_evidence(knowledge_group, supporting)
        feature_families = self._feature_families(supporting)
        workstream = self._workstream(knowledge_group, supporting)
        confidence, confidence_breakdown = self._confidence_engine.score(
            supporting,
            contradicting=contradicting,
            unknown_evidence=unknown_evidence,
        )

        lead_knowledge = max(
            knowledge_group,
            key=lambda knowledge: (knowledge.abstraction_level, knowledge.confidence),
        )
        hypothesis_id = stable_id(
            "hypothesis",
            {
                "knowledge_ids": supporting_knowledge_ids,
                "evidence_ids": supporting_evidence_ids,
                "statement": statement,
            },
        )
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            title=title,
            statement=statement,
            supporting_evidence=supporting_evidence_ids,
            supporting_knowledge_objects=supporting_knowledge_ids,
            contradicting_evidence=[evidence.evidence_id for evidence in contradicting],
            contradicting_knowledge_objects=contradicting_knowledge_ids,
            unknown_evidence=unknown_evidence,
            unknowns=list(unknown_evidence),
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
            plausible_causes=self._causes_for(knowledge_group, supporting),
            validation_steps=self._validation_steps_for(knowledge_group, supporting),
            provenance={},
            metadata={
                "dominant_category": self._dominant_category(supporting),
                "affected_columns": sorted({
                    column
                    for knowledge in knowledge_group
                    for column in knowledge.metadata.get("affected_columns", [])
                }),
                "knowledge_concepts": [knowledge.concept for knowledge in knowledge_group],
                "hypothesis_group_id": hypothesis_group_id,
                "feature_families": feature_families,
                "workstream": workstream,
                "creation_explanation": (
                    f"Created by merging {len(knowledge_group)} knowledge object(s) connected "
                    "in the "
                    "hypothesis graph."
                ),
                "hypothesis_graph_density": (
                    round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0
                ),
            },
            created_at=stable_datetime(
                "hypothesis",
                {
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_group_id": hypothesis_group_id,
                },
            ),
        )
        hypothesis.provenance = {
            **ProvenanceTracker.for_hypothesis(hypothesis, lead_knowledge, evidence_lookup),
            "knowledge_ids": supporting_knowledge_ids,
            "hypothesis_group_id": hypothesis_group_id,
        }
        return hypothesis

    def _compose_statement(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting: list[EvidenceUnit],
    ) -> tuple[str, str]:
        concepts = {knowledge.concept for knowledge in knowledge_group}
        categories = {
            category
            for knowledge in knowledge_group
            for category in knowledge.metadata.get("categories", [])
        }
        evidence_count = len(supporting)
        columns = sorted({
            column
            for knowledge in knowledge_group
            for column in knowledge.metadata.get("affected_columns", [])
        })
        category_summary = self._category_summary(supporting)
        representative_columns = ", ".join(columns[:4]) if columns else "the affected telemetry fields"

        if {
            "Neighbor Cell Measurement Subsystem",
            "Measurement Availability",
        }.issubset(concepts) or {"Neighbor Cell Availability", "Neighbor Cell Telemetry"}.intersection(concepts):
            return (
                "Neighbor-cell measurements are systematically unavailable.",
                f"{evidence_count} evidence units ({category_summary}) consistently indicate "
                f"systematic missingness and strong dependence across {representative_columns}, "
                "suggesting a shared acquisition or telemetry constraint.",
            )
        if {
            "Radio Signal Quality",
            "Measurement Availability",
        }.issubset(concepts):
            return (
                "Radio-signal measurements are incompletely collected.",
                f"{evidence_count} evidence units ({category_summary}) show that "
                f"{representative_columns} are incomplete in a pattern consistent with "
                "collection-boundary or ingestion limitations.",
            )
        if {
            "Configuration Parameters",
            "Feature Engineering Signals",
        }.issubset(concepts):
            return (
                "Configuration and engineered features contain low-information structure.",
                f"{evidence_count} evidence units ({category_summary}) show that "
                f"{representative_columns} remain constant, redundant, or weakly informative, "
                "suggesting a configuration or feature-engineering artifact.",
            )
        if "Identifier Integrity" in concepts:
            return (
                "Record-linkage integrity is at risk.",
                f"{evidence_count} evidence units ({category_summary}) show identifier-related "
                "instability that could create duplicate joins, null-key propagation, or "
                "incorrect aggregation.",
            )
        if "Correlation Community" in categories or "Radio Signal Quality" in concepts:
            return (
                "Signal-quality features behave as one tightly coupled subsystem.",
                f"{evidence_count} evidence units ({category_summary}) show that "
                f"{representative_columns} move together strongly enough to indicate a shared "
                "subsystem rather than independent variables.",
            )
        if "Neighbor Cell Availability" in concepts:
            return (
                "Neighbor-cell telemetry is conditionally collected.",
                f"{evidence_count} evidence units ({category_summary}) show that "
                f"{representative_columns} are absent or incomplete in a repeatable pattern, "
                "suggesting collection that depends on network state or telemetry policy.",
            )
        if "Neighbor Cell Telemetry" in concepts:
            return (
                "Neighbor-cell telemetry is degraded by one shared reporting failure.",
                f"{evidence_count} evidence units ({category_summary}) connect "
                f"{representative_columns} through missingness, redundancy, or statistical coupling, "
                "indicating one reporting or acquisition failure mode.",
            )
        if "Radio Configuration" in concepts:
            return (
                "Radio-configuration parameters show little operational variation.",
                f"{evidence_count} evidence units ({category_summary}) show that "
                f"{representative_columns} remain nearly constant, suggesting a fixed export profile "
                "or a configuration artifact rather than a varying operational signal.",
            )

        lead = max(
            knowledge_group,
            key=lambda knowledge: (knowledge.abstraction_level, knowledge.confidence),
        )
        return (
            f"{lead.concept} shows a consistent, evidence-backed structural pattern.",
            f"{evidence_count} evidence units ({category_summary}) connect {lead.concept.lower()} "
            f"to recurring behavior across {representative_columns}.",
        )

    def _find_contradictions(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting_evidence_ids: list[str],
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> list[EvidenceUnit]:
        covered_columns = {
            column
            for knowledge in knowledge_group
            for column in knowledge.metadata.get("affected_columns", [])
        }
        supporting_categories = {
            category
            for knowledge in knowledge_group
            for category in knowledge.metadata.get("categories", [])
        }
        contradicting: list[EvidenceUnit] = []
        for evidence in evidence_lookup.values():
            if evidence.evidence_id in supporting_evidence_ids:
                continue
            if not set(evidence.affected_columns).intersection(covered_columns):
                continue
            if evidence.category not in supporting_categories:
                contradicting.append(evidence)
        return sorted(contradicting, key=lambda item: (-item.strength, item.title))[:4]

    def _find_contradicting_knowledge_objects(
        self,
        knowledge_group: list[KnowledgeObject],
        contradicting: list[EvidenceUnit],
    ) -> list[str]:
        contradicting_evidence_ids = {evidence.evidence_id for evidence in contradicting}
        contradicting_knowledge_ids: list[str] = []
        for knowledge in knowledge_group:
            if contradicting_evidence_ids.intersection(knowledge.supporting_evidence):
                contradicting_knowledge_ids.append(knowledge.knowledge_id)
        return sorted(set(contradicting_knowledge_ids))


    def _causes_for(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting: list[EvidenceUnit],
    ) -> list[str]:
        concepts = {knowledge.concept for knowledge in knowledge_group}
        if "Neighbor Cell Measurement Subsystem" in concepts:
            return [
                "Neighbor-cell telemetry may only be collected under specific network states.",
                "The collection pipeline may be truncating or omitting neighbor measurements.",
            ]
        if "Configuration Parameters" in concepts:
            return [
                "Feature engineering defaults may be flattening useful variation.",
                "Configuration parameters may have been exported without operational diversity.",
            ]
        if "Identifier Integrity" in concepts:
            return [
                "Primary-key generation or deduplication logic may be unstable.",
                "A join or ingestion fan-out may be duplicating records.",
            ]
        if any(evidence.category == "Correlation Community" for evidence in supporting):
            return [
                "Several features may be expressing the same latent subsystem behavior.",
                "Derived variables may be repeating one source signal under multiple names.",
            ]
        return ["The pattern likely reflects a coherent subsystem or pipeline behavior."]

    def _validation_steps_for(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting: list[EvidenceUnit],
    ) -> list[str]:
        columns = sorted({
            column
            for knowledge in knowledge_group
            for column in knowledge.metadata.get("affected_columns", [])
        })
        concepts = {knowledge.concept for knowledge in knowledge_group}
        focus_columns = columns[:4]
        focus_text = ", ".join(focus_columns) if focus_columns else "the affected columns"
        if {"Neighbor Cell Measurement Subsystem", "Neighbor Cell Availability", "Neighbor Cell Telemetry"}.intersection(concepts):
            return [
                f"Validate whether {focus_text} are intentionally absent for specific radio configurations or operating states.",
                "Compare the missingness pattern against cell_id, timestamp, vendor, software_version, and handover_state.",
                "Check whether neighbor-cell acquisition is gated by serving-cell quality, measurement policy, or collection boundaries.",
            ]
        if {"Configuration Parameters", "Radio Configuration", "Feature Engineering Signals"}.intersection(concepts):
            return [
                f"Verify whether {focus_text} are expected to remain constant for this export or software profile.",
                "Compare these fields across vendor, software_version, batch, and deployment configuration slices.",
                "Inspect feature-engineering defaults to confirm whether low variation was introduced during preprocessing.",
            ]
        if "Identifier Integrity" in concepts:
            return [
                f"Trace {focus_text} back to the source system and confirm key uniqueness before joins.",
                "Compare duplicate or null-key patterns across ingestion batch, source table, and merge stage.",
                "Validate whether fan-out joins or partial-record ingestion introduced identifier corruption.",
            ]
        return [
            f"Inspect the source values and generation logic for {focus_text}.",
            "Compare the pattern across timestamp, source segment, vendor, and operational state.",
            "Confirm whether the observed structure reflects collection behavior, feature engineering, or true domain behavior.",
        ]

    def _category_summary(self, supporting: list[EvidenceUnit]) -> str:
        categories = Counter(evidence.category for evidence in supporting)
        return ", ".join(
            f"{count} {category.lower()}"
            for category, count in categories.most_common(3)
        )

    def _unknown_evidence(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting: list[EvidenceUnit],
    ) -> list[str]:
        unknowns = [
            "Temporal information unavailable for causal sequencing.",
        ]
        if any(evidence.affected_rows is None for evidence in supporting):
            unknowns.append("Row-level event lineage unavailable for direct trace comparison.")
        if len(knowledge_group) > 1:
            unknowns.append(
                "External system metadata unavailable for confirming subsystem boundaries."
            )
        return sorted(set(unknowns))

    def _dominant_category(self, supporting: list[EvidenceUnit]) -> str:
        categories = Counter(evidence.category for evidence in supporting)
        return categories.most_common(1)[0][0] if categories else "unknown"

    def _feature_families(self, supporting: list[EvidenceUnit]) -> list[str]:
        families = {
            column.split("_")[0].lower()
            for evidence in supporting
            for column in (evidence.representative_columns or evidence.affected_columns)
            if "_" in column
        }
        return sorted(families)

    def _workstream(
        self,
        knowledge_group: list[KnowledgeObject],
        supporting: list[EvidenceUnit],
    ) -> str:
        concepts = {knowledge.concept for knowledge in knowledge_group}
        dominant = self._dominant_category(supporting)
        if "Identifier Integrity" in concepts:
            return "record_linkage"
        if {"Neighbor Cell Measurement Subsystem", "Neighbor Cell Availability", "Neighbor Cell Telemetry"}.intersection(concepts):
            return "neighbor_collection"
        if {"Configuration Parameters", "Radio Configuration"}.intersection(concepts):
            return "configuration_pipeline"
        if {"Feature Engineering Signals", "Feature Engineering Artifact"}.intersection(concepts):
            return "feature_engineering"
        if "Radio Signal Quality" in concepts or dominant == "Correlation Community":
            return "signal_quality"
        return "measurement_pipeline"

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
