"""Semantic knowledge construction engine."""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx
from loguru import logger

from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.knowledge.models import KnowledgeObject
from investigation_engine.reasoning.provenance.tracker import ProvenanceTracker
from investigation_engine.utils.deterministic import stable_datetime, stable_id

_SEMANTIC_LABEL_RULES: tuple[tuple[str, set[str], str], ...] = (
    (
        "Neighbor Cell Measurement Subsystem",
        {"neighbor", "pci", "earfcn", "cell", "serving", "rsrp", "rsrq", "sinr"},
        "neighbor_cell_measurement",
    ),
    (
        "Radio Signal Quality",
        {"rsrp", "rsrq", "sinr", "rssi", "cqi", "signal", "radio"},
        "radio_signal_quality",
    ),
    (
        "Measurement Availability",
        {"missing", "availability", "null", "dropout", "unavailable"},
        "measurement_availability",
    ),
    (
        "Configuration Parameters",
        {"window", "size", "config", "parameter", "threshold", "setting", "batch"},
        "configuration_parameters",
    ),
    (
        "Spatial Context",
        {"latitude", "longitude", "lat", "lon", "geo", "spatial", "location"},
        "spatial_context",
    ),
    (
        "Identifier Integrity",
        {"id", "identifier", "record", "account", "user", "entity"},
        "identifier_integrity",
    ),
    (
        "Feature Engineering Signals",
        {"redundant", "duplicate", "constant", "variance", "information"},
        "feature_engineering_signals",
    ),
)
_SEMANTIC_STOPWORDS = {
    "into",
    "context",
    "available",
    "evidence",
    "indicates",
    "indicate",
    "analytical",
    "pattern",
    "patterns",
    "coherent",
    "form",
    "from",
    "were",
    "this",
    "that",
    "with",
    "dataset",
    "wide",
    "feature",
    "features",
    "measurements",
    "measurement",
}


class KnowledgeConstructionEngine:
    """Transforms evidence units into higher-level semantic knowledge objects."""

    def construct(self, evidence_units: list[EvidenceUnit]) -> list[KnowledgeObject]:
        if not evidence_units:
            return []

        graph = self._build_knowledge_graph(evidence_units)
        communities = [
            sorted(component)
            for component in nx.connected_components(graph)
        ]
        communities.sort(key=lambda component: (component[0], len(component)))

        evidence_lookup = {evidence.evidence_id: evidence for evidence in evidence_units}
        knowledge_objects = [
            self._build_knowledge_object(
                community_id=f"knowledge_{index + 1:04d}",
                evidence_units=[evidence_lookup[evidence_id] for evidence_id in community],
                graph=graph.subgraph(community).copy(),
            )
            for index, community in enumerate(communities)
        ]
        knowledge_objects = self._merge_duplicate_concepts(knowledge_objects, evidence_lookup)
        self._attach_related_concepts(knowledge_objects)
        logger.info(
            "KnowledgeConstructionEngine: constructed {} knowledge objects from {} evidence units",
            len(knowledge_objects),
            len(evidence_units),
        )
        return sorted(
            knowledge_objects,
            key=lambda item: (-item.abstraction_level, -item.confidence, item.concept),
        )

    def _build_knowledge_graph(self, evidence_units: list[EvidenceUnit]) -> nx.Graph:
        graph = nx.Graph()
        for evidence in evidence_units:
            graph.add_node(
                evidence.evidence_id,
                evidence=evidence,
                categories=evidence.metadata.get("source_categories", [evidence.category]),
                columns=evidence.affected_columns,
            )

        for index, left in enumerate(evidence_units):
            for right in evidence_units[index + 1:]:
                similarity, reasons = self._knowledge_similarity(left, right)
                if similarity < 0.34:
                    continue
                graph.add_edge(
                    left.evidence_id,
                    right.evidence_id,
                    weight=round(similarity, 4),
                    reasons=reasons,
                )

        return graph

    def _knowledge_similarity(
        self,
        left: EvidenceUnit,
        right: EvidenceUnit,
    ) -> tuple[float, list[str]]:
        left_terms = self._semantic_terms(left)
        right_terms = self._semantic_terms(right)
        shared_terms = left_terms & right_terms

        left_columns = set(left.affected_columns)
        right_columns = set(right.affected_columns)
        shared_columns = left_columns & right_columns

        left_categories = set(left.metadata.get("source_categories", [left.category]))
        right_categories = set(right.metadata.get("source_categories", [right.category]))
        shared_categories = left_categories & right_categories
        left_concept_key = self._evidence_concept_key(left)
        right_concept_key = self._evidence_concept_key(right)

        term_similarity = self._jaccard(left_terms, right_terms)
        column_similarity = self._jaccard(left_columns, right_columns)
        category_similarity = self._jaccard(left_categories, right_categories)
        confidence_similarity = 1.0 - abs(left.confidence - right.confidence)
        community_similarity = 1.0 - abs(left.community_strength - right.community_strength)
        concept_bonus = 1.0 if left_concept_key == right_concept_key else 0.0

        score = (
            term_similarity * 0.30
            + column_similarity * 0.20
            + category_similarity * 0.20
            + confidence_similarity * 0.10
            + community_similarity * 0.10
            + concept_bonus * 0.10
        )
        if left_concept_key != right_concept_key and not shared_columns:
            score *= 0.20

        reasons: list[str] = []
        if shared_terms:
            reasons.append(f"shared semantic terms: {sorted(shared_terms)[:4]}")
        if shared_columns:
            reasons.append(f"shared columns: {sorted(shared_columns)[:4]}")
        if shared_categories:
            reasons.append(f"shared evidence categories: {sorted(shared_categories)}")
        if concept_bonus:
            reasons.append("same semantic concept family")

        return score, reasons

    def _build_knowledge_object(
        self,
        community_id: str,
        evidence_units: list[EvidenceUnit],
        graph: nx.Graph,
    ) -> KnowledgeObject:
        concept, concept_key = self._discover_concept(evidence_units)
        confidence = round(
            sum(evidence.confidence for evidence in evidence_units) / len(evidence_units),
            2,
        )
        categories = sorted({
            category
            for evidence in evidence_units
            for category in evidence.metadata.get("source_categories", [evidence.category])
        })
        columns = sorted({
            column for evidence in evidence_units for column in evidence.affected_columns
        })
        abstraction_level = round(
            min(
                1.0,
                0.35
                + 0.15 * max(len(evidence_units) - 1, 0)
                + 0.1 * max(len(categories) - 1, 0),
            ),
            2,
        )

        provenance = ProvenanceTracker.for_evidence(
            evidence_units,
            {
                finding_id: None  # type: ignore[dict-item]
                for evidence in evidence_units
                for finding_id in evidence.supporting_findings
            },
        )
        provenance["knowledge_graph_id"] = community_id

        knowledge_id = stable_id(
            "knowledge",
            {
                "concept": concept,
                "evidence": [evidence.evidence_id for evidence in evidence_units],
                "columns": columns,
            },
        )
        return KnowledgeObject(
            knowledge_id=knowledge_id,
            concept=concept,
            description=(
                f"{concept} abstracts {len(evidence_units)} evidence unit(s) spanning "
                f"{categories} into one reusable semantic concept."
            ),
            supporting_evidence=[evidence.evidence_id for evidence in evidence_units],
            knowledge_graph_id=community_id,
            provenance=provenance,
            confidence=confidence,
            abstraction_level=abstraction_level,
            metadata={
                "concept_key": concept_key,
                "categories": categories,
                "affected_columns": columns,
                "knowledge_graph_density": (
                    round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0
                ),
                "semantic_terms": sorted({
                    term for evidence in evidence_units for term in self._semantic_terms(evidence)
                }),
                "creation_explanation": (
                    f"Created because {len(evidence_units)} evidence unit(s) shared "
                    "semantic terms, "
                    "categories, or column structure."
                ),
            },
            created_at=stable_datetime(
                "knowledge",
                {
                    "knowledge_id": knowledge_id,
                    "community_id": community_id,
                },
            ),
        )

    def _discover_concept(self, evidence_units: list[EvidenceUnit]) -> tuple[str, str]:
        tokens = Counter(
            term
            for evidence in evidence_units
            for term in self._semantic_terms(evidence)
            if term not in _SEMANTIC_STOPWORDS
        )
        categories = {
            category
            for evidence in evidence_units
            for category in evidence.metadata.get("source_categories", [evidence.category])
        }
        columns = sorted({
            column for evidence in evidence_units for column in evidence.affected_columns
        })
        concept = self._match_concept(set(tokens), categories, columns)
        if concept is not None:
            return concept
        return self._fallback_concept(columns, categories, set(tokens))

    def _evidence_concept_key(self, evidence: EvidenceUnit) -> str:
        terms = self._semantic_terms(evidence)
        categories = set(evidence.metadata.get("source_categories", [evidence.category]))
        concept = self._match_concept(terms, categories, evidence.affected_columns)
        if concept is not None:
            return concept[1]
        return str(evidence.metadata.get("concept_key", evidence.category)).lower()

    def _match_concept(
        self,
        terms: set[str],
        categories: set[str],
        columns: list[str],
    ) -> tuple[str, str] | None:
        lowered_columns = [column.lower() for column in columns]
        column_tokens = {
            token
            for column in columns
            for token in re.split(r"[^a-zA-Z0-9]+", column.lower())
            if len(token) >= 3
        }
        combined_terms = terms | column_tokens
        has_neighbor_columns = any(column.startswith("neighbor") for column in lowered_columns)
        has_serving_columns = any(column.startswith("serving") for column in lowered_columns)
        if has_neighbor_columns and {"missing_values", "missingness_relationships", "Systematic Missingness"}.intersection(categories):
            return "Neighbor Cell Availability", "neighbor_cell_availability"
        if has_neighbor_columns and {"duplicate_features", "redundant_features", "Correlation Community", "pearson_correlation"}.intersection(categories | combined_terms):
            return "Neighbor Cell Telemetry", "neighbor_cell_telemetry"
        if {"neighbor", "pci"}.intersection(combined_terms) and {"missing", "availability", "null", "unavailable"}.intersection(combined_terms):
            return "Neighbor Cell Availability", "neighbor_cell_availability"
        if {"neighbor", "rsrp", "rsrq", "sinr"}.intersection(combined_terms) and {"correlation", "signal", "radio"}.intersection(combined_terms):
            return "Neighbor Cell Telemetry", "neighbor_cell_telemetry"
        if has_serving_columns or {"serving", "rsrp", "rsrq", "sinr"}.intersection(combined_terms):
            return "Serving Cell Signal Quality", "serving_cell_signal_quality"
        if {"config", "parameter", "threshold", "window", "batch", "setting"}.intersection(combined_terms):
            return "Radio Configuration", "radio_configuration"
        if {"handover", "mobility"}.intersection(combined_terms):
            return "Mobility Events", "mobility_events"
        if {"interference", "noise"}.intersection(combined_terms):
            return "Interference Pattern", "interference_pattern"
        if {"telemetry", "pipeline", "collection", "ingestion"}.intersection(combined_terms):
            return "Measurement Pipeline", "measurement_pipeline"
        if {"constant_features", "near_constant_features", "cardinality", "duplicate_features"}.intersection(categories):
            return "Feature Engineering Artifact", "feature_engineering_artifact"
        if {"missing_values", "missingness_relationships", "Systematic Missingness"}.intersection(categories):
            return "Missingness Cluster", "missingness_cluster"
        for label, required_terms, key in _SEMANTIC_LABEL_RULES:
            if len(required_terms & combined_terms) >= max(1, min(2, len(required_terms))):
                return label, key

        if "Systematic Missingness" in categories or "missing_values" in categories:
            return "Measurement Availability", "measurement_availability"
        if "Correlation Community" in categories:
            return "Signal Relationship Structure", "signal_relationship_structure"
        if "Redundant Feature Group" in categories:
            return "Feature Engineering Signals", "feature_engineering_signals"
        if "Identifier Integrity Failure" in categories:
            return "Identifier Integrity", "identifier_integrity"
        return None

    def _fallback_concept(
        self,
        columns: list[str],
        categories: set[str],
        terms: set[str],
    ) -> tuple[str, str]:
        family = self._column_family(columns, terms)
        family_key = family.lower().replace(" ", "_")
        if {"missing_values", "missingness_relationships", "Systematic Missingness"}.intersection(categories):
            return f"{family} Missingness", f"{family_key}_missingness"
        if {"duplicate_features", "redundant_features", "Redundant Feature Group"}.intersection(categories):
            return f"{family} Feature Duplication", f"{family_key}_feature_duplication"
        if {"constant_features", "near_constant_features", "cardinality", "Low Information Feature Group"}.intersection(categories):
            return f"{family} Feature Stability", f"{family_key}_feature_stability"
        if "Identifier Integrity Failure" in categories:
            return "Identifier Integrity", "identifier_integrity"
        if "Correlation Community" in categories:
            return f"{family} Dependency Pattern", f"{family_key}_dependency_pattern"
        return f"{family} Telemetry Subsystem", f"{family_key}_telemetry_subsystem"

    def _column_family(self, columns: list[str], terms: set[str]) -> str:
        lowered_columns = [column.lower() for column in columns]
        joined = " ".join(lowered_columns)
        if any(column.startswith("neighbor") for column in lowered_columns) or "neighbor" in terms:
            return "Neighbor Cell"
        if any(column.startswith("serving") for column in lowered_columns) or "serving" in terms:
            return "Serving Cell"
        if any(token in joined for token in ("vendor", "software", "version", "config", "window", "threshold")):
            return "Radio Configuration"
        if any(token in joined for token in ("handover", "mobility", "cell_id")):
            return "Mobility"
        if columns:
            return " ".join(part.capitalize() for part in columns[0].split("_")[:2])
        return "Measurement"

    def _semantic_terms(self, evidence: EvidenceUnit) -> set[str]:
        raw_parts = [
            evidence.category,
            evidence.title,
            evidence.summary,
            *evidence.representative_columns,
            *evidence.affected_columns,
        ]
        terms: set[str] = set()
        for part in raw_parts:
            for token in re.split(r"[^a-zA-Z0-9]+", part.lower()):
                if len(token) < 3:
                    continue
                if token in _SEMANTIC_STOPWORDS:
                    continue
                terms.add(token)
        return terms

    def _merge_duplicate_concepts(
        self,
        knowledge_objects: list[KnowledgeObject],
        evidence_lookup: dict[str, EvidenceUnit],
    ) -> list[KnowledgeObject]:
        grouped: dict[tuple[str, str], list[KnowledgeObject]] = {}
        for knowledge in knowledge_objects:
            concept_key = str(knowledge.metadata.get("concept_key", knowledge.concept)).lower()
            grouped.setdefault((knowledge.concept, concept_key), []).append(knowledge)

        merged: list[KnowledgeObject] = []
        for (concept, concept_key), group in sorted(grouped.items(), key=lambda item: item[0]):
            if len(group) == 1:
                merged.append(group[0])
                continue
            supporting_evidence = sorted({
                evidence_id
                for knowledge in group
                for evidence_id in knowledge.supporting_evidence
            })
            evidence_units = [
                evidence_lookup[evidence_id]
                for evidence_id in supporting_evidence
                if evidence_id in evidence_lookup
            ]
            categories = sorted({
                category
                for evidence in evidence_units
                for category in evidence.metadata.get("source_categories", [evidence.category])
            })
            columns = sorted({
                column for evidence in evidence_units for column in evidence.affected_columns
            })
            merged_id = stable_id(
                "knowledge",
                {
                    "concept": concept,
                    "evidence": supporting_evidence,
                    "columns": columns,
                },
            )
            merged.append(
                KnowledgeObject(
                    knowledge_id=merged_id,
                    concept=concept,
                    description=(
                        f"{concept} consolidates {len(evidence_units)} evidence unit(s) spanning "
                        f"{categories} into one analyst-facing semantic concept."
                    ),
                    supporting_evidence=supporting_evidence,
                    knowledge_graph_id=group[0].knowledge_graph_id,
                    related_concepts=[],
                    provenance={
                        "merged_knowledge_ids": [knowledge.knowledge_id for knowledge in group],
                        "knowledge_graph_ids": [
                            knowledge.knowledge_graph_id
                            for knowledge in group
                            if knowledge.knowledge_graph_id
                        ],
                        "evidence_ids": supporting_evidence,
                    },
                    confidence=round(
                        sum(knowledge.confidence for knowledge in group) / len(group),
                        2,
                    ),
                    abstraction_level=max(knowledge.abstraction_level for knowledge in group),
                    metadata={
                        "concept_key": concept_key,
                        "categories": categories,
                        "affected_columns": columns,
                        "knowledge_graph_density": round(
                            sum(
                                float(knowledge.metadata.get("knowledge_graph_density", 0.0))
                                for knowledge in group
                            ) / len(group),
                            4,
                        ),
                        "semantic_terms": sorted({
                            term
                            for knowledge in group
                            for term in knowledge.metadata.get("semantic_terms", [])
                        }),
                        "creation_explanation": (
                            f"Merged {len(group)} semantically redundant knowledge objects into one "
                            "analyst-facing concept."
                        ),
                    },
                    created_at=stable_datetime(
                        "knowledge",
                        {
                            "knowledge_id": merged_id,
                            "supporting_evidence": supporting_evidence,
                        },
                    ),
                )
            )
        return sorted(
            merged,
            key=lambda item: (-item.abstraction_level, -item.confidence, item.concept),
        )

    def _attach_related_concepts(self, knowledge_objects: list[KnowledgeObject]) -> None:
        for knowledge in knowledge_objects:
            related: list[str] = []
            columns = set(knowledge.metadata.get("affected_columns", []))
            terms = set(knowledge.metadata.get("semantic_terms", []))
            for other in knowledge_objects:
                if other.knowledge_id == knowledge.knowledge_id:
                    continue
                other_columns = set(other.metadata.get("affected_columns", []))
                other_terms = set(other.metadata.get("semantic_terms", []))
                if columns.intersection(other_columns) or terms.intersection(other_terms):
                    related.append(other.concept)
            knowledge.related_concepts = sorted(set(related))

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
