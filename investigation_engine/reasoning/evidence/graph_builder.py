"""Evidence graph construction for graph-based evidence compression."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
from loguru import logger

from investigation_engine.config.settings import EvidenceCompressionSettings
from investigation_engine.models.finding import Finding
from investigation_engine.reasoning.evidence.similarity import (
    FindingSimilarityScorer,
    SimilarityResult,
)


class EvidenceGraphBuilder:
    """Builds a sparse, explainable evidence graph from findings."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._config = config
        self._scorer = FindingSimilarityScorer(config)
        self._similarity_cache: dict[tuple[str, str], SimilarityResult] = {}

    @property
    def scorer(self) -> FindingSimilarityScorer:
        return self._scorer

    def build_graph(self, findings: list[Finding]) -> nx.Graph:
        graph = nx.Graph()
        for finding in findings:
            graph.add_node(
                finding.id,
                finding=finding,
                finding_id=finding.id,
                investigator=finding.module,
                severity=finding.severity.value,
                confidence=finding.confidence,
                metadata=finding.metadata,
                affected_columns=list(finding.affected_columns),
                affected_rows=list(finding.affected_rows or []),
                evidence=finding.evidence,
            )

        candidate_pairs = self._candidate_pairs(findings)
        logger.debug(
            "EvidenceGraphBuilder: evaluating {} candidate finding pairs",
            len(candidate_pairs),
        )

        finding_lookup = {finding.id: finding for finding in findings}
        for left_id, right_id in sorted(candidate_pairs):
            left = finding_lookup[left_id]
            right = finding_lookup[right_id]
            result = self._score_pair(left, right)
            if result.total_weight < self._config.edge_weight_threshold:
                continue
            graph.add_edge(
                left_id,
                right_id,
                weight=result.total_weight,
                structural_similarity=result.structural_similarity,
                statistical_similarity=result.statistical_similarity,
                semantic_similarity=result.semantic_similarity,
                reasons=result.reasons,
                signals=result.signals,
            )

        logger.info(
            "EvidenceGraphBuilder: built graph with {} nodes and {} edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def _candidate_pairs(self, findings: list[Finding]) -> set[tuple[str, str]]:
        if len(findings) <= 1:
            return set()

        pair_ids: set[tuple[str, str]] = set()
        indexes: dict[str, dict[str, set[str]]] = {
            "columns": defaultdict(set),
            "families": defaultdict(set),
            "terms": defaultdict(set),
            "categories": defaultdict(set),
        }

        for finding in findings:
            if self._config.candidate_shared_columns:
                for column in finding.affected_columns:
                    indexes["columns"][column.lower()].add(finding.id)

            if self._config.candidate_shared_feature_family:
                family = self._scorer.feature_family(finding)
                if family is not None:
                    indexes["families"][family].add(finding.id)

            if self._config.candidate_shared_terms:
                for token in self._scorer.semantic_tokens(finding):
                    indexes["terms"][token].add(finding.id)

            if self._config.candidate_shared_category:
                category = str(finding.metadata.get("category", "unknown"))
                indexes["categories"][category].add(finding.id)

        for buckets in indexes.values():
            for members in buckets.values():
                if 1 < len(members) <= self._config.max_index_bucket_size:
                    for left_id, right_id in combinations(sorted(members), 2):
                        pair_ids.add((left_id, right_id))

        if not pair_ids and len(findings) <= self._config.max_index_bucket_size:
            for left, right in combinations(sorted(finding.id for finding in findings), 2):
                pair_ids.add((left, right))

        return pair_ids

    def _score_pair(self, left: Finding, right: Finding) -> SimilarityResult:
        key = tuple(sorted((left.id, right.id)))
        cached = self._similarity_cache.get(key)
        if cached is not None:
            return cached
        result = self._scorer.score(left, right)
        self._similarity_cache[key] = result
        return result

