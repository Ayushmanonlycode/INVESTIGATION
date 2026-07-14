"""Deterministic reasoning graph export utilities."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path
from typing import Literal
import json

import networkx as nx

from investigation_engine.models.investigation import InvestigationResult
from investigation_engine.utils.deterministic import stable_payload


GraphKind = Literal["evidence", "knowledge", "hypothesis", "investigation"]


class ReasoningGraphExporter:
    """Builds and exports deterministic reasoning graphs."""

    def build_graph(self, result: InvestigationResult, kind: GraphKind) -> nx.Graph:
        if kind == "evidence":
            return self._build_evidence_graph(result)
        if kind == "knowledge":
            return self._build_knowledge_graph(result)
        if kind == "hypothesis":
            return self._build_hypothesis_graph(result)
        if kind == "investigation":
            return self._build_investigation_graph(result)
        raise ValueError(f"Unsupported graph kind: {kind}")

    def export(self, graph: nx.Graph, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".graphml":
            nx.write_graphml(graph, output_path)
        elif suffix == ".gexf":
            nx.write_gexf(graph, output_path)
        elif suffix == ".json":
            payload = nx.node_link_data(graph)
            output_path.write_text(json.dumps(payload, indent=2, default=str))
        elif suffix == ".mmd":
            output_path.write_text(self.to_mermaid(graph))
        elif suffix == ".svg":
            output_path.write_text(self.to_svg(graph))
        elif suffix == ".png":
            self._write_png(graph, output_path)
        else:
            raise ValueError(f"Unsupported graph export extension: {output_path.suffix}")
        return output_path

    @staticmethod
    def to_mermaid(graph: nx.Graph) -> str:
        lines = ["flowchart LR"]
        for node_id, attributes in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
            title = str(attributes.get("title", node_id)).replace('"', "'")
            lines.append(f'  {node_id}["{title}"]')
        for left, right in sorted(graph.edges()):
            lines.append(f"  {left} --> {right}")
        return "\n".join(lines)

    @staticmethod
    def to_svg(graph: nx.Graph) -> str:
        nodes = sorted(graph.nodes())
        if not nodes:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120"></svg>'
        radius = 120
        center_x = 160
        center_y = 160
        positions = {}
        for index, node in enumerate(nodes):
            angle = 2 * pi * (index / len(nodes))
            positions[node] = (
                center_x + radius * cos(angle),
                center_y + radius * sin(angle),
            )
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320">']
        for left, right in sorted(graph.edges()):
            x1, y1 = positions[left]
            x2, y2 = positions[right]
            parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#94a3b8" stroke-width="1.5" />')
        for node in nodes:
            x, y = positions[node]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="18" fill="#1d4ed8" />')
            label = str(graph.nodes[node].get("title", node))
            parts.append(f'<text x="{x:.2f}" y="{y + 32:.2f}" font-size="10" text-anchor="middle">{label}</text>')
        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def _write_png(graph: nx.Graph, output_path: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise RuntimeError("PNG export requires matplotlib to be installed.") from error
        layout = nx.circular_layout(graph)
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(graph, pos=layout, with_labels=True, node_size=800, font_size=8)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, format="png")
        plt.close()

    @staticmethod
    def _build_evidence_graph(result: InvestigationResult) -> nx.Graph:
        graph = nx.Graph()
        for evidence in result.evidence_units:
            graph.add_node(
                evidence.evidence_id,
                title=evidence.title,
                confidence=evidence.confidence,
                provenance=stable_payload(evidence.provenance),
            )
        for index, left in enumerate(result.evidence_units):
            for right in result.evidence_units[index + 1:]:
                if set(left.affected_columns).intersection(right.affected_columns):
                    graph.add_edge(left.evidence_id, right.evidence_id)
        return graph

    @staticmethod
    def _build_knowledge_graph(result: InvestigationResult) -> nx.Graph:
        graph = nx.Graph()
        for knowledge in result.knowledge_objects:
            graph.add_node(
                knowledge.knowledge_id,
                title=knowledge.concept,
                confidence=knowledge.confidence,
                provenance=stable_payload(knowledge.provenance),
            )
        for index, left in enumerate(result.knowledge_objects):
            for right in result.knowledge_objects[index + 1:]:
                if set(left.supporting_evidence).intersection(right.supporting_evidence) or set(left.related_concepts).intersection({right.concept}):
                    graph.add_edge(left.knowledge_id, right.knowledge_id)
        return graph

    @staticmethod
    def _build_hypothesis_graph(result: InvestigationResult) -> nx.Graph:
        graph = nx.Graph()
        for hypothesis in result.hypotheses:
            graph.add_node(
                hypothesis.hypothesis_id,
                title=hypothesis.title,
                confidence=hypothesis.confidence,
                provenance=stable_payload(hypothesis.provenance),
            )
        for index, left in enumerate(result.hypotheses):
            for right in result.hypotheses[index + 1:]:
                if set(left.supporting_evidence).intersection(right.supporting_evidence) or set(left.supporting_knowledge_objects).intersection(right.supporting_knowledge_objects):
                    graph.add_edge(left.hypothesis_id, right.hypothesis_id)
        return graph

    @staticmethod
    def _build_investigation_graph(result: InvestigationResult) -> nx.Graph:
        graph = nx.Graph()
        for investigation in result.investigations:
            graph.add_node(
                investigation.investigation_id,
                title=investigation.title,
                confidence=investigation.confidence,
                provenance=stable_payload(investigation.provenance),
            )
        for index, left in enumerate(result.investigations):
            for right in result.investigations[index + 1:]:
                if set(left.supporting_hypotheses).intersection(right.supporting_hypotheses) or set(left.supporting_evidence).intersection(right.supporting_evidence):
                    graph.add_edge(left.investigation_id, right.investigation_id)
        return graph
