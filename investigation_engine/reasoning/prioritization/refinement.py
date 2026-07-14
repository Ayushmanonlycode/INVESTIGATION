"""Deterministic investigation scope refinement and title generation."""

from __future__ import annotations

from collections import Counter
from statistics import mean

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

from investigation_engine.reasoning.hypothesis.models import Hypothesis


class InvestigationScopeRefiner:
    """Splits low-cohesion hypothesis groups into coherent analyst tasks."""

    def __init__(self, cohesion_threshold: float) -> None:
        self._cohesion_threshold = cohesion_threshold

    def split(
        self,
        hypotheses: list[Hypothesis],
        graph: nx.Graph,
        *,
        total_evidence_count: int | None = None,
        max_evidence_share: float = 1.0,
    ) -> list[list[Hypothesis]]:
        if len(hypotheses) <= 1:
            return [sorted(hypotheses, key=lambda item: item.hypothesis_id)]

        split_graph = nx.Graph()
        for hypothesis in hypotheses:
            split_graph.add_node(hypothesis.hypothesis_id, hypothesis=hypothesis)

        for index, left in enumerate(hypotheses):
            for right in hypotheses[index + 1:]:
                cohesion = self._pair_cohesion(left, right, graph)
                if cohesion >= self._cohesion_threshold:
                    split_graph.add_edge(left.hypothesis_id, right.hypothesis_id, weight=cohesion)

        if split_graph.number_of_edges() == 0:
            return [[hypothesis] for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id)]
        groups = self._connected_groups(split_graph)
        evidence_share = self._evidence_share(hypotheses, total_evidence_count)
        edge_density = self.group_edge_density(graph)
        modularity_score = self.group_modularity(split_graph)

        if (
            len(groups) == 1
            and len(hypotheses) > 1
            and evidence_share > max_evidence_share
            and (
                self.group_cohesion(hypotheses, graph) < max(self._cohesion_threshold + 0.12, 0.70)
                or edge_density < 0.85
                or modularity_score > 0.0
            )
        ):
            groups = self._modularity_groups(split_graph)

        if len(groups) == 1 and evidence_share > max_evidence_share and len(hypotheses) > 1:
            return [[hypothesis] for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id)]

        return groups

    def _connected_groups(self, split_graph: nx.Graph) -> list[list[Hypothesis]]:
        groups = [
            sorted(
                [split_graph.nodes[node_id]["hypothesis"] for node_id in component],
                key=lambda item: item.hypothesis_id,
            )
            for component in nx.connected_components(split_graph)
        ]
        groups.sort(key=lambda group: (group[0].hypothesis_id, len(group)))
        return groups

    def group_cohesion(self, hypotheses: list[Hypothesis], graph: nx.Graph) -> float:
        if len(hypotheses) <= 1:
            return 1.0
        scores = [
            self._pair_cohesion(left, right, graph)
            for index, left in enumerate(hypotheses)
            for right in hypotheses[index + 1 :]
        ]
        return round(mean(scores), 4) if scores else 1.0

    @staticmethod
    def group_edge_density(graph: nx.Graph) -> float:
        return round(float(nx.density(graph)) if graph.number_of_nodes() > 1 else 1.0, 4)

    def group_modularity(self, graph: nx.Graph) -> float:
        if graph.number_of_nodes() <= 2 or graph.number_of_edges() == 0:
            return 0.0
        communities = list(greedy_modularity_communities(graph, weight="weight"))
        if len(communities) <= 1:
            return 0.0
        return round(float(modularity(graph, communities, weight="weight")), 4)

    def _modularity_groups(self, graph: nx.Graph) -> list[list[Hypothesis]]:
        if graph.number_of_nodes() <= 2 or graph.number_of_edges() == 0:
            return self._connected_groups(graph)
        communities = list(greedy_modularity_communities(graph, weight="weight"))
        if len(communities) <= 1:
            return self._connected_groups(graph)
        groups = [
            sorted(
                [graph.nodes[node_id]["hypothesis"] for node_id in community],
                key=lambda item: item.hypothesis_id,
            )
            for community in communities
        ]
        groups.sort(key=lambda group: (group[0].hypothesis_id, len(group)))
        return groups

    @staticmethod
    def _evidence_share(hypotheses: list[Hypothesis], total_evidence_count: int | None) -> float:
        if total_evidence_count is None or total_evidence_count <= 0:
            return 0.0
        supporting_evidence = {
            evidence_id for hypothesis in hypotheses for evidence_id in hypothesis.supporting_evidence
        }
        return len(supporting_evidence) / total_evidence_count

    def _pair_cohesion(
        self,
        left: Hypothesis,
        right: Hypothesis,
        graph: nx.Graph,
    ) -> float:
        left_concepts = set(left.metadata.get("knowledge_concepts", []))
        right_concepts = set(right.metadata.get("knowledge_concepts", []))
        left_evidence = set(left.supporting_evidence)
        right_evidence = set(right.supporting_evidence)
        left_families = set(left.metadata.get("feature_families", []))
        right_families = set(right.metadata.get("feature_families", []))
        concept_similarity = self._jaccard(left_concepts, right_concepts)
        evidence_similarity = self._jaccard(left_evidence, right_evidence)
        feature_family_similarity = self._jaccard(left_families, right_families)
        workstream_similarity = (
            1.0 if left.metadata.get("workstream") == right.metadata.get("workstream") else 0.0
        )
        graph_connectivity = 0.0
        edge_density = self.group_edge_density(graph)
        if graph.has_edge(left.hypothesis_id, right.hypothesis_id):
            graph_connectivity = float(graph[left.hypothesis_id][right.hypothesis_id].get("weight", 0.0))

        return round(
            concept_similarity * 0.25
            + evidence_similarity * 0.15
            + graph_connectivity * 0.20
            + feature_family_similarity * 0.15
            + workstream_similarity * 0.15
            + edge_density * 0.10,
            4,
        )

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


class InvestigationTitleGenerator:
    """Generates deterministic analyst-oriented investigation titles."""

    def generate(self, hypotheses: list[Hypothesis]) -> str:
        lead_hypothesis = max(
            hypotheses,
            key=lambda item: (item.confidence, item.title, item.hypothesis_id),
        )
        concepts = Counter(
            concept
            for hypothesis in hypotheses
            for concept in hypothesis.metadata.get("knowledge_concepts", [])
        )
        categories = Counter(
            str(hypothesis.metadata.get("dominant_category", "unknown"))
            for hypothesis in hypotheses
        )
        workstreams = Counter(
            str(hypothesis.metadata.get("workstream", "unknown")) for hypothesis in hypotheses
        )
        affected_columns = Counter(
            column
            for hypothesis in hypotheses
            for column in hypothesis.metadata.get("affected_columns", [])
        )
        dominant_concept = concepts.most_common(1)[0][0] if concepts else "Measurement Availability"
        dominant_category = categories.most_common(1)[0][0] if categories else "unknown"
        workstream = workstreams.most_common(1)[0][0] if workstreams else "measurement_pipeline"
        dominant_column = affected_columns.most_common(1)[0][0] if affected_columns else ""
        direct_title = self._issue_focused_title(lead_hypothesis.title)
        if direct_title is not None:
            return direct_title

        if workstream == "neighbor_collection":
            if "Correlation Community" in categories or "Radio Signal Quality" in concepts:
                return "Correlated Neighbor Cell Reporting Failure"
            return "Systematic Neighbor Cell Measurement Loss"
        if workstream == "configuration_pipeline":
            if any(keyword in dominant_column.lower() for keyword in ("window", "config", "threshold", "setting", "batch")):
                return "Constant Configuration Parameter"
            return "Low-Variation Configuration Parameter"
        if workstream == "feature_engineering":
            if any(keyword in dominant_column.lower() for keyword in ("sample", "degradation", "indicator")):
                return "Near-Constant Sample Degradation Indicator"
            return "Low-Information Feature Engineering Signal"
        if workstream == "signal_quality":
            if "Measurement Availability" in concepts:
                return "Incomplete Radio Signal Collection"
            return "Coupled Signal Quality Feature Behavior"
        if dominant_category == "Systematic Missingness" or "Measurement Availability" in dominant_concept:
            return "Systematic Measurement Availability Loss"
        if workstream == "record_linkage":
            return "Record Linkage Integrity Risk"
        if dominant_column:
            return f"{self._column_phrase(dominant_column)} Consistency Risk"
        return f"{self._concept_prefix(dominant_concept)} Consistency Risk"

    @staticmethod
    def _concept_prefix(concept: str) -> str:
        if "Signal" in concept:
            return "Signal Quality"
        if "Configuration" in concept:
            return "Configuration Pipeline"
        if "Feature Engineering" in concept:
            return "Feature Engineering"
        if "Neighbor Cell" in concept:
            return "Neighbor Cell Collection"
        if "Identifier" in concept:
            return "Record Linkage Integrity"
        return "Measurement Pipeline"

    @staticmethod
    def _column_phrase(column_name: str) -> str:
        return " ".join(
            part.upper() if part.isupper() else part.capitalize()
            for part in column_name.split("_")
        )

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        cleaned = subject.replace("-", " ")
        cleaned = cleaned.replace("measurements", "measurement").replace("Measurements", "Measurement")
        return " ".join(word.capitalize() for word in cleaned.split())

    @staticmethod
    def _issue_focused_title(title: str) -> str | None:
        normalized = title.strip().rstrip(".")
        lowered = normalized.lower()
        if lowered.endswith("appear systematically unavailable"):
            subject = normalized[: -len("appear systematically unavailable")].strip()
            subject = InvestigationTitleGenerator._normalize_subject(subject)
            return f"Systematic {subject} Loss"
        if lowered.endswith("are systematically unavailable"):
            subject = normalized[: -len("are systematically unavailable")].strip()
            subject = InvestigationTitleGenerator._normalize_subject(subject)
            return f"Systematic {subject} Loss"
        if lowered.endswith("appear incompletely collected"):
            subject = normalized[: -len("appear incompletely collected")].strip()
            subject = InvestigationTitleGenerator._normalize_subject(subject)
            return f"Incomplete {subject} Collection"
        if lowered.endswith("are incompletely collected"):
            subject = normalized[: -len("are incompletely collected")].strip()
            subject = InvestigationTitleGenerator._normalize_subject(subject)
            return f"Incomplete {subject} Collection"
        if lowered.endswith("is conditionally collected") or lowered.endswith("are conditionally collected"):
            suffix = "is conditionally collected" if lowered.endswith("is conditionally collected") else "are conditionally collected"
            subject = normalized[: -len(suffix)].strip()
            subject = InvestigationTitleGenerator._normalize_subject(subject)
            return f"Conditional {subject} Collection"
        if lowered.endswith("behave as a tightly coupled subsystem"):
            subject = normalized[: -len("behave as a tightly coupled subsystem")].strip()
            return f"Tightly Coupled {subject} Behavior"
        if lowered.endswith("may be compromised"):
            subject = normalized[: -len("may be compromised")].strip()
            return f"{subject} Risk"
        if lowered.endswith("may be limiting signal quality"):
            return "Low-Information Configuration Feature Structure"
        if lowered.endswith("deserve investigation"):
            subject = normalized[: -len("deserve investigation")].strip()
            return f"{subject} Consistency Risk"
        return None
