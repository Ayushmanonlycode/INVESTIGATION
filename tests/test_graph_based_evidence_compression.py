"""Tests for Graph-Based Evidence Compression."""

from __future__ import annotations

from datetime import UTC, datetime

from investigation_engine.config.settings import EvidenceCompressionSettings, Settings
from investigation_engine.models.finding import Finding
from investigation_engine.models.severity import Severity
from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
from investigation_engine.reasoning.evidence.community_detection import (
    EvidenceCommunityDetector,
)
from investigation_engine.reasoning.evidence.compression import (
    EvidenceCompressionEngine,
    RuleBasedEvidenceCompressionEngine,
)
from investigation_engine.reasoning.evidence.graph_builder import EvidenceGraphBuilder
from investigation_engine.reasoning.evidence.graph_compressor import (
    GraphBasedEvidenceCompressionEngine,
)


def _finding(
    finding_id: str,
    category: str,
    columns: list[str],
    *,
    module: str = "relationship_investigator",
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.8,
    evidence: dict[str, float | dict[str, float] | None] | None = None,
) -> Finding:
    return Finding(
        id=finding_id,
        module=module,
        title=f"{category} finding {finding_id}",
        description=f"Synthetic {category} finding {finding_id}",
        evidence=evidence or {
            "correlation_coefficient": 0.91 if "correlation" in category else None,
            "p_value": 0.001 if "correlation" in category else None,
            "mutual_information_score": 0.27 if category == "mutual_information" else None,
            "vif": 12.0 if category == "multicollinearity" else None,
            "missing_ratio": 0.42 if category == "missing_values" else None,
            "thresholds": {"synthetic": 1.0},
        },
        severity=severity,
        confidence=confidence,
        recommendation="Investigate this pattern.",
        affected_columns=columns,
        metadata={"category": category},
        created_at=datetime.now(UTC),
    )


def _settings(**overrides: object) -> EvidenceCompressionSettings:
    base = EvidenceCompressionSettings()
    return base.model_copy(update=overrides)


class TestEvidenceGraphBuilder:
    def test_builds_explainable_weighted_edges(self):
        config = _settings(edge_weight_threshold=0.10)
        builder = EvidenceGraphBuilder(config)
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding(
                "f2",
                "missingness_relationships",
                ["neighbor1_rsrp", "neighbor1_sinr"],
                evidence={
                    "missingness_correlation": 0.88,
                    "thresholds": {"synthetic": 1.0},
                },
            ),
            _finding("f3", "duplicate_features", ["customer_id"], module="integrity_investigator"),
        ]

        graph = builder.build_graph(findings)

        assert graph.number_of_nodes() == 3
        assert graph.has_edge("f1", "f2")
        edge = graph["f1"]["f2"]
        assert 0.0 <= edge["weight"] <= 1.0
        assert edge["reasons"]
        assert "shared columns" in " ".join(edge["reasons"])

    def test_avoids_connecting_unrelated_findings(self):
        config = _settings(edge_weight_threshold=0.30)
        builder = EvidenceGraphBuilder(config)
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp"]),
            _finding("f2", "identifiers", ["account_id"], module="integrity_investigator"),
        ]

        graph = builder.build_graph(findings)
        assert graph.number_of_edges() == 0


class TestCommunityDetection:
    def test_detects_stable_communities(self):
        config = _settings(algorithm="connected_components", edge_weight_threshold=0.10)
        builder = EvidenceGraphBuilder(config)
        findings = [
            _finding("a1", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("a2", "spearman_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
            _finding("b1", "identifiers", ["user_id"], module="integrity_investigator"),
        ]
        graph = builder.build_graph(findings)

        detector = EvidenceCommunityDetector(config)
        communities = detector.detect(graph)

        assert communities == [["a1", "a2"], ["b1"]]

    def test_modularity_algorithms_are_deterministic(self):
        findings = [
            _finding("c1", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("c2", "spearman_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
            _finding("c3", "mutual_information", ["neighbor2_rsrp", "neighbor2_rsrq"]),
            _finding("c4", "mutual_information", ["neighbor2_rsrp", "neighbor2_sinr"]),
        ]
        config = _settings(algorithm="louvain", edge_weight_threshold=0.10, random_seed=7)
        builder = EvidenceGraphBuilder(config)
        graph = builder.build_graph(findings)
        detector = EvidenceCommunityDetector(config)

        first = detector.detect(graph)
        second = detector.detect(graph)

        assert first == second


class TestGraphCompressor:
    def test_compresses_partially_overlapping_communities(self):
        config = _settings(algorithm="connected_components", edge_weight_threshold=0.10)
        compressor = GraphBasedEvidenceCompressionEngine(config)
        findings = [
            _finding("m1", "missing_values", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("m2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_sinr"]),
            _finding("r1", "pearson_correlation", ["serving_rsrp", "serving_rsrq"]),
            _finding("r2", "spearman_correlation", ["serving_rsrp", "serving_sinr"]),
            _finding("u1", "duplicate_features", ["user_id"], module="integrity_investigator"),
        ]

        evidence_units = compressor.compress(findings)

        assert len(evidence_units) == 3
        assert all(unit.community_id for unit in evidence_units)
        assert all(unit.representative_columns for unit in evidence_units if unit.affected_columns)

    def test_preserves_complete_provenance_and_explanation(self):
        config = _settings(algorithm="connected_components", edge_weight_threshold=0.10)
        compressor = GraphBasedEvidenceCompressionEngine(config)
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_rsrq"]),
        ]

        evidence = compressor.compress(findings)[0]

        assert evidence.provenance["finding_ids"] == ["f1", "f2"]
        assert evidence.provenance["investigators"] == ["relationship_investigator"]
        assert evidence.metadata["merge_explanation"]
        assert evidence.community_strength >= 0.0

    def test_public_engine_remains_backward_compatible(self):
        settings = Settings(
            evidence_compression=_settings(
                algorithm="connected_components",
                edge_weight_threshold=0.10,
            )
        )
        engine = EvidenceCompressionEngine(settings)
        findings = [
            _finding("f1", "pearson_correlation", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("f2", "spearman_correlation", ["neighbor1_rsrp", "neighbor1_sinr"]),
        ]

        evidence_units = engine.compress(findings)

        assert len(evidence_units) == 1
        assert evidence_units[0].category == "Correlation Community"


class TestBenchmarking:
    def test_benchmark_reports_required_metrics(self):
        config = _settings(
            algorithm="connected_components",
            edge_weight_threshold=0.10,
            benchmark_repeat_runs=1,
        )
        findings = [
            _finding("f1", "missing_values", ["neighbor1_rsrp"]),
            _finding("f2", "missingness_relationships", ["neighbor1_rsrp", "neighbor1_rsrq"]),
            _finding("f3", "duplicate_features", ["account_id"], module="integrity_investigator"),
        ]
        benchmark = EvidenceCompressionBenchmark(config)
        reports = benchmark.benchmark(
            findings,
            {
                "graph_based": GraphBasedEvidenceCompressionEngine(config),
                "rule_based": RuleBasedEvidenceCompressionEngine(),
            },
        )

        assert set(reports) == {"graph_based", "rule_based"}
        for report in reports.values():
            assert report.evidence_unit_count >= 1
            assert report.compression_ratio > 0.0
            assert report.average_findings_per_unit > 0.0
            assert report.runtime_seconds >= 0.0
            assert isinstance(report.graph_density, float)
            assert report.provenance_preserved is True
