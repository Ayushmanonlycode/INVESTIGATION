"""Evidence Graph — network representation of dataset findings.

This represents the internal graph of findings, where nodes are Findings and
edges represent relationships (overlapping columns, correlated issues, shared
feature families, etc.).
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
from loguru import logger

from investigation_engine.models.finding import Finding


class EvidenceGraph:
    """Internal Evidence Graph mapping relationships between Findings.

    Allows graph-based reasoning (e.g. finding connected components or clusters
    representing related issues) before applying fusion rules.
    """

    def __init__(self) -> None:
        self._graph = nx.Graph()

    @property
    def graph(self) -> nx.Graph:
        """Access the underlying NetworkX graph."""
        return self._graph

    def build_from_findings(self, findings: list[Finding]) -> None:
        """Populate nodes and build edges based on relationships.

        Args:
            findings: List of Findings to graph.
        """
        self._graph.clear()

        if not findings:
            return

        # 1. Add all findings as nodes with metadata
        for finding in findings:
            self._graph.add_node(
                finding.id,
                finding=finding,
                module=finding.module,
                category=finding.metadata.get("category", "unknown"),
            )

        logger.debug("EvidenceGraph: added {} nodes", len(findings))

        # 2. Build edges based on shared attributes
        for i, f1 in enumerate(findings):
            for f2 in findings[i + 1:]:
                weight = 0.0
                reasons: list[str] = []

                # Column overlap
                cols1 = set(f1.affected_columns)
                cols2 = set(f2.affected_columns)
                overlap = cols1.intersection(cols2)
                if overlap:
                    # Weight based on overlap ratio
                    overlap_ratio = len(overlap) / max(len(cols1.union(cols2)), 1)
                    weight += 3.0 * overlap_ratio
                    reasons.append("column_overlap")

                # Shared category
                cat1 = f1.metadata.get("category")
                cat2 = f2.metadata.get("category")
                if cat1 and cat2 and cat1 == cat2:
                    weight += 1.5
                    reasons.append("shared_category")

                # Shared module
                if f1.module == f2.module:
                    weight += 0.5
                    reasons.append("shared_module")

                # Shared feature family (heuristic name-based matching, e.g. neighborX)
                family1 = self._get_feature_family(cols1)
                family2 = self._get_feature_family(cols2)
                if family1 and family2 and family1 == family2:
                    weight += 2.5
                    reasons.append("shared_feature_family")

                # Missingness correlation relationship link
                if self._are_correlated_missingness(f1, f2):
                    weight += 4.0
                    reasons.append("missingness_correlation_relationship")

                # Same severity
                if f1.severity == f2.severity:
                    weight += 0.2
                    reasons.append("same_severity")

                # If relationship strength exceeds threshold, add edge
                if weight >= 1.5:
                    self._graph.add_edge(
                        f1.id,
                        f2.id,
                        weight=weight,
                        reasons=reasons,
                    )

        logger.debug(
            "EvidenceGraph: built {} edges between findings",
            self._graph.number_of_edges(),
        )

    def get_connected_subgraphs(self) -> list[list[Finding]]:
        """Find clusters of related findings (connected components).

        Returns:
            List of lists of Findings, each representing a connected cluster.
        """
        subgraphs: list[list[Finding]] = []
        components = list(nx.connected_components(self._graph))

        for component in components:
            findings_in_comp = [
                self._graph.nodes[node_id]["finding"]
                for node_id in component
            ]
            subgraphs.append(findings_in_comp)

        return subgraphs

    @staticmethod
    def _get_feature_family(columns: set[str]) -> str | None:
        """Identify a common prefix/suffix family across columns.

        Example:
            {'neighbor1_pci', 'neighbor1_brsrp_dbm'} -> 'neighbor1'
        """
        if not columns:
            return None

        # Take first column as reference
        ref = list(columns)[0]

        # Look for prefix families like "neighborX" or "serving"
        match = re.match(r"^(neighbor\d+|serving|target)", ref, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Split on underscores and look at prefix
        parts = ref.split("_")
        if len(parts) > 1 and len(parts[0]) > 3:
            return parts[0].lower()

        return None

    @staticmethod
    def _are_correlated_missingness(f1: Finding, f2: Finding) -> bool:
        """Check if one finding specifically points to correlated missingness between columns in another finding."""
        cat1 = f1.metadata.get("category")
        cat2 = f2.metadata.get("category")

        if cat1 == "missingness_relationships" and cat2 == "missing_values":
            # If f1 is a correlation finding, does it involve the column in f2?
            col2 = f2.affected_columns[0] if f2.affected_columns else ""
            if col2 in f1.affected_columns:
                return True

        if cat2 == "missingness_relationships" and cat1 == "missing_values":
            col1 = f1.affected_columns[0] if f1.affected_columns else ""
            if col1 in f2.affected_columns:
                return True

        return False
