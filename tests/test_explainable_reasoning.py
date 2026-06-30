"""Tests for Phase 2.3 explainable investigative reasoning."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from typer.testing import CliRunner

from investigation_engine.cli.main import _summarize_findings, app
from investigation_engine.config.settings import EvidenceCompressionSettings, Settings
from investigation_engine.core.engine import InvestigationEngine
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.confidence import ConfidencePropagationEngine
from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
from investigation_engine.reasoning.evidence.graph_builder import EvidenceGraphBuilder
from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine
from investigation_engine.reasoning.evidence.models import EvidenceUnit
from investigation_engine.reasoning.hypothesis.models import Hypothesis
from investigation_engine.reasoning.prioritization.refinement import (
    InvestigationScopeRefiner,
    InvestigationTitleGenerator,
)
from investigation_engine.utils.deterministic import stable_payload


def _finding(
    finding_id: str,
    category: str,
    columns: list[str],
    *,
    module: str = "relationship_investigator",
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.8,
) -> Finding:
    return Finding(
        id=finding_id,
        module=module,
        title=f"{category} finding {finding_id}",
        description=f"Detailed description for {finding_id}.",
        evidence={
            "correlation_coefficient": 0.91 if "correlation" in category else None,
            "missing_ratio": 0.42 if "missing" in category else None,
            "thresholds": {"synthetic": 1.0},
        },
        severity=severity,
        confidence=confidence,
        recommendation="Investigate this pattern.",
        affected_columns=columns,
        metadata={"category": category},
        created_at=datetime.now(UTC),
    )


def _evidence(
    evidence_id: str,
    title: str,
    findings: list[str],
    columns: list[str],
    *,
    category: str = "Correlation Community",
    confidence: float = 0.85,
    strength: float = 72.0,
    community_strength: float = 0.7,
    investigators: list[str] | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        category=category,
        title=title,
        summary=f"Summary for {title}.",
        supporting_findings=findings,
        provenance={"investigators": investigators or ["relationship_investigator"]},
        confidence=confidence,
        strength=strength,
        community_strength=community_strength,
        structural_similarity=0.7,
        statistical_similarity=0.6,
        semantic_similarity=0.8,
        merge_explanation=["Merged because of strong semantic similarity."],
        affected_columns=columns,
        representative_columns=columns[:1],
    )


def _hypothesis(
    hypothesis_id: str,
    title: str,
    evidence_ids: list[str],
    *,
    concepts: list[str],
    workstream: str,
    feature_families: list[str],
    affected_columns: list[str],
    confidence: float = 0.75,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=title,
        statement=f"Statement for {title}.",
        supporting_evidence=evidence_ids,
        supporting_knowledge_objects=[f"knowledge_{hypothesis_id}"],
        confidence=confidence,
        confidence_breakdown={"evidence_strength": 0.3},
        plausible_causes=["Cause"],
        validation_steps=["Step"],
        metadata={
            "knowledge_concepts": concepts,
            "dominant_category": "Systematic Missingness",
            "workstream": workstream,
            "feature_families": feature_families,
            "affected_columns": affected_columns,
        },
    )


class TestConfidencePropagation:
    def test_confidence_breakdown_is_deterministic_and_explainable(self) -> None:
        engine = ConfidencePropagationEngine()
        supporting = [
            _evidence(
                "e1",
                "Neighbor availability",
                ["f1"],
                ["neighbor1_rsrp"],
                investigators=["integrity_investigator", "relationship_investigator"],
            ),
            _evidence("e2", "Neighbor relation", ["f2"], ["neighbor1_rsrq"]),
        ]
        contradicting = [_evidence("e3", "Counter signal", ["f3"], ["neighbor1_sinr"], strength=30.0)]

        confidence, breakdown = engine.score(
            supporting,
            contradicting=contradicting,
            unknown_evidence=["Temporal context unavailable."],
        )

        assert confidence > 0.0
        assert set(breakdown) == {
            "evidence_strength",
            "community_strength",
            "cross_investigator_agreement",
            "contradicting_evidence",
            "unknown_evidence",
        }
        assert round(sum(breakdown.values()), 2) == confidence


class TestInvestigationRefinement:
    def test_title_generator_prefers_issue_focused_titles(self) -> None:
        generator = InvestigationTitleGenerator()
        title = generator.generate([
            _hypothesis(
                "h1",
                "Neighbor issue",
                ["e1"],
                concepts=["Neighbor Cell Measurement Subsystem", "Measurement Availability"],
                workstream="neighbor_collection",
                feature_families=["neighbor1"],
                affected_columns=["neighbor1_rsrp"],
            )
        ])
        assert title == "Systematic Neighbor Cell Measurement Loss"

    def test_scope_refiner_splits_low_cohesion_groups(self) -> None:
        refiner = InvestigationScopeRefiner(0.46)
        left = _hypothesis(
            "h1",
            "Neighbor availability",
            ["e1"],
            concepts=["Measurement Availability"],
            workstream="measurement_pipeline",
            feature_families=["neighbor1"],
            affected_columns=["shared_signal", "neighbor1_rsrp"],
        )
        right = _hypothesis(
            "h2",
            "Signal inconsistency",
            ["e2"],
            concepts=["Radio Signal Quality"],
            workstream="measurement_pipeline",
            feature_families=["serving"],
            affected_columns=["shared_signal", "serving_rsrp"],
        )

        import networkx as nx

        graph = nx.Graph()
        graph.add_node(left.hypothesis_id, hypothesis=left)
        graph.add_node(right.hypothesis_id, hypothesis=right)
        graph.add_edge(left.hypothesis_id, right.hypothesis_id, weight=0.35)

        groups = refiner.split([left, right], graph)

        assert len(groups) == 2

    def test_scope_refiner_enforces_evidence_share_guardrail(self) -> None:
        refiner = InvestigationScopeRefiner(0.46)
        hypotheses = [
            _hypothesis(
                f"h{index}",
                f"Hypothesis {index}",
                [f"e{index}"],
                concepts=["Measurement Availability"],
                workstream="measurement_pipeline",
                feature_families=[f"family{index}"],
                affected_columns=[f"shared_signal_{index}"],
            )
            for index in range(1, 4)
        ]

        import networkx as nx

        graph = nx.Graph()
        for hypothesis in hypotheses:
            graph.add_node(hypothesis.hypothesis_id, hypothesis=hypothesis)
        graph.add_edge("h1", "h2", weight=0.48)
        graph.add_edge("h2", "h3", weight=0.48)

        groups = refiner.split(
            hypotheses,
            graph,
            total_evidence_count=4,
            max_evidence_share=0.60,
        )

        assert len(groups) == 3


class TestExplainabilityArtifacts:
    def test_evidence_units_expose_similarity_scores(self) -> None:
        compressor = GraphBasedEvidenceCompressionEngine(
            EvidenceCompressionSettings(algorithm="connected_components", edge_weight_threshold=0.10)
        )
        evidence = compressor.compress([
            _finding("f1", "missing_values", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_sinr"]),
        ])[0]

        assert evidence.structural_similarity >= 0.0
        assert evidence.statistical_similarity >= 0.0
        assert evidence.semantic_similarity >= 0.0
        assert evidence.merge_explanation

    def test_engine_attaches_provenance_trees_and_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "neighbor1_rsrp": [1.0, None, None, 4.0],
                "neighbor1_rsrq": [1.0, None, None, 4.0],
                "neighbor1_sinr": [2.0, 2.0, 2.0, 2.0],
            }
        )
        result = InvestigationEngine(Settings()).investigate_dataframe(df, name="phase23_explain")

        assert result.provenance_trees
        assert result.reasoning_metrics is not None
        root = result.provenance_trees[0]
        assert root["level"] == "investigation"
        assert root["children"]
        assert result.reasoning_metrics["reasoning"]["runtime_per_layer"]
        assert root["confidence_breakdown"]
        assert root["likely_causes"]
        assert root["evidence_strength"]["rating"] in {"Strong", "Moderate", "Limited"}
        evidence_node = root["children"][0]["children"][0]["children"][0]
        assert evidence_node["evidence_explainability"]["merge_explanation"]
        assert evidence_node["category"]
        assert evidence_node["affected_columns"]


class TestBenchmarkAndDeterminism:
    def test_benchmark_exports_markdown_and_json(self) -> None:
        config = EvidenceCompressionSettings(
            algorithm="connected_components",
            edge_weight_threshold=0.10,
            benchmark_repeat_runs=1,
        )
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_rsrq"]),
        ]
        benchmark = EvidenceCompressionBenchmark(config)
        reports = benchmark.benchmark(
            findings,
            {
                "graph_based": GraphBasedEvidenceCompressionEngine(config),
                "rule_based": RuleBasedEvidenceCompressionEngine(),
            },
        )

        markdown = EvidenceCompressionBenchmark.to_markdown(reports)
        json_payload = EvidenceCompressionBenchmark.to_json(reports)

        assert "Compression Ratio" in markdown
        assert "Determinism" in markdown
        assert json_payload["graph_based"]["deterministic"] is True

    def test_reasoning_outputs_are_deterministic_for_identical_findings(self) -> None:
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_sinr"]),
            _finding("f3", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
        ]
        engine = InvestigationEngine(Settings())
        first = engine._fusion_engine.build_reasoning_artifacts(findings)  # noqa: SLF001
        second = engine._fusion_engine.build_reasoning_artifacts(findings)  # noqa: SLF001

        first_signature = [
            (
                item.investigation_id,
                item.title,
                tuple(item.supporting_hypotheses),
                item.confidence,
            )
            for item in first.investigation_queue.investigations
        ]
        second_signature = [
            (
                item.investigation_id,
                item.title,
                tuple(item.supporting_hypotheses),
                item.confidence,
            )
            for item in second.investigation_queue.investigations
        ]
        assert first_signature == second_signature

    def test_reasoning_serialization_is_stable_across_repeated_runs(self) -> None:
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_sinr"]),
            _finding("f3", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
        ]
        engine = InvestigationEngine(Settings())
        first = engine._fusion_engine.build_reasoning_artifacts(findings)  # noqa: SLF001
        second = engine._fusion_engine.build_reasoning_artifacts(findings)  # noqa: SLF001
        first_payload = stable_payload([
            investigation.model_dump(mode="json")
            for investigation in first.investigation_queue.investigations
        ])
        second_payload = stable_payload([
            investigation.model_dump(mode="json")
            for investigation in second.investigation_queue.investigations
        ])
        assert first_payload == second_payload


class TestCliFlags:
    def test_finding_summary_groups_categories_instead_of_listing_ids(self) -> None:
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp"]),
            _finding("f2", "missing_values", ["neighbor1_rsrq"]),
            _finding("f3", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
        ]

        summary = _summarize_findings(
            [finding.id for finding in findings],
            {finding.id: finding for finding in findings},
        )

        assert "2 missing values finding(s)" in summary
        assert "1 pearson correlation finding(s)" in summary
        assert "f1" not in summary
        assert "f2" not in summary
        assert "f3" not in summary
    def test_cli_renders_explain_metrics_and_benchmark(self, tmp_path) -> None:
        dataset_path = tmp_path / "dataset.csv"
        pd.DataFrame(
            {
                "neighbor1_rsrp": [1.0, None, None, 4.0, 5.0],
                "neighbor1_rsrq": [1.0, None, None, 4.0, 5.0],
                "serving_rsrp": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        ).to_csv(dataset_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "investigate",
                str(dataset_path),
                "--log-level",
                "ERROR",
                "--explain",
                "--metrics",
                "--benchmark",
            ],
        )

        assert result.exit_code == 0
        assert "Reasoning Lineage" in result.output
        assert "Knowledge Objects" in result.output
        assert "Structured Evidence Units" in result.output
        assert "Evidence Strength" in result.output
        assert "Reasoning Flow" in result.output
        assert "Counts" in result.output
        assert "Benchmark" in result.output
        assert "Method Comparison" in result.output
        assert "Confidence breakdown" in result.output
        assert "Evidence explainability" in result.output
        assert "Executive Summary" in result.output


class TestDuplicateGraphPrevention:
    def test_engine_builds_evidence_graph_once_in_main_reasoning_path(self, monkeypatch) -> None:
        calls = {"count": 0}
        original = EvidenceGraphBuilder.build_graph

        def counting_build_graph(self, findings):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return original(self, findings)

        monkeypatch.setattr(EvidenceGraphBuilder, "build_graph", counting_build_graph)

        df = pd.DataFrame(
            {
                "neighbor1_rsrp": [1.0, None, None, 4.0],
                "neighbor1_rsrq": [1.0, None, None, 4.0],
                "serving_rsrp": [4.0, 3.0, 2.0, 1.0],
            }
        )
        InvestigationEngine(Settings()).investigate_dataframe(df, name="single_graph_build")

        assert calls["count"] == 1
