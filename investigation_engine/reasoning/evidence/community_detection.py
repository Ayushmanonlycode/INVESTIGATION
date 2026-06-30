"""Community detection utilities for evidence graphs."""

from __future__ import annotations

import networkx as nx
from loguru import logger
from networkx.algorithms.community import (
    greedy_modularity_communities,
    louvain_communities,
    modularity,
)

from investigation_engine.config.settings import EvidenceCompressionSettings


class EvidenceCommunityDetector:
    """Detects evidence communities using a configurable graph algorithm."""

    def __init__(self, config: EvidenceCompressionSettings) -> None:
        self._config = config

    def detect(self, graph: nx.Graph) -> list[list[str]]:
        if graph.number_of_nodes() == 0:
            return []

        if graph.number_of_edges() == 0:
            communities = [{node_id} for node_id in graph.nodes]
        elif self._config.algorithm == "connected_components":
            communities = list(nx.connected_components(graph))
        elif self._config.algorithm == "greedy_modularity":
            communities = list(greedy_modularity_communities(
                graph,
                weight="weight",
                resolution=self._config.greedy_resolution,
            ))
        else:
            communities = list(louvain_communities(
                graph,
                weight="weight",
                resolution=self._config.louvain_resolution,
                seed=self._config.random_seed,
            ))

        normalized = [sorted(component) for component in communities]
        normalized.sort(key=lambda component: (component[0], len(component)))
        logger.info(
            "EvidenceCommunityDetector: detected {} communities using {}",
            len(normalized),
            self._config.algorithm,
        )
        return normalized

    def modularity_score(self, graph: nx.Graph, communities: list[list[str]]) -> float:
        if graph.number_of_edges() == 0 or len(communities) <= 1:
            return 0.0
        return float(
            modularity(
                graph,
                [set(community) for community in communities],
                weight="weight",
            )
        )
