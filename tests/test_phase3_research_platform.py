"""Tests for Phase 3 evaluation, visualization, and publication assets."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from investigation_engine.cli.main import app
from investigation_engine.config.settings import Settings
from investigation_engine.core.engine import InvestigationEngine
from investigation_engine.evaluation import DatasetEvaluationSuite
from investigation_engine.reasoning.ablation import AblationEngine
from investigation_engine.reports.paper_assets import PaperAssetsWriter
from investigation_engine.reports.visualization import ReasoningGraphExporter


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "neighbor1_rsrp": [1.0, None, None, 4.0, 5.0],
            "neighbor1_rsrq": [1.0, None, None, 4.0, 5.0],
            "neighbor1_sinr": [2.0, 2.0, 2.0, 2.0, 2.0],
            "serving_rsrp": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


class TestPhase3MetricsAndAblation:
    def test_engine_exposes_phase3_metric_sections(self) -> None:
        result = InvestigationEngine(Settings()).investigate_dataframe(_dataset(), name="phase3_metrics")

        assert result.reasoning_metrics is not None
        assert "runtime" in result.reasoning_metrics
        assert "memory" in result.reasoning_metrics
        assert "dashboard" in result.reasoning_metrics
        assert "reasoning_flow" in result.reasoning_metrics
        assert "average_community_size" in result.reasoning_metrics["graph"]
        assert "overall_compression_factor" in result.reasoning_metrics["compression"]

    def test_ablation_engine_emits_all_requested_variants(self) -> None:
        result = InvestigationEngine(Settings()).investigate_dataframe(_dataset(), name="phase3_ablation")
        reports = AblationEngine(Settings()).run(result.findings)

        assert {report.variant for report in reports} == {
            "baseline",
            "no_graph_compression",
            "no_knowledge_layer",
            "no_hypothesis_layer",
            "no_semantic_splitting",
            "no_confidence_propagation",
        }
        assert all(report.runtime_seconds >= 0.0 for report in reports)


class TestPhase3VisualizationAndAssets:
    def test_visualization_exporter_writes_mermaid_graphml_and_svg(self, tmp_path) -> None:
        result = InvestigationEngine(Settings()).investigate_dataframe(_dataset(), name="phase3_visualize")
        exporter = ReasoningGraphExporter()
        graph = exporter.build_graph(result, "evidence")

        mermaid_path = tmp_path / "evidence.mmd"
        graphml_path = tmp_path / "evidence.graphml"
        svg_path = tmp_path / "evidence.svg"

        exporter.export(graph, mermaid_path)
        exporter.export(graph, graphml_path)
        exporter.export(graph, svg_path)

        assert "flowchart LR" in mermaid_path.read_text()
        assert "<graphml" in graphml_path.read_text()
        assert "<svg" in svg_path.read_text()

    def test_paper_assets_writer_creates_publication_bundle(self, tmp_path) -> None:
        writer = PaperAssetsWriter(tmp_path / "paper_assets")
        paths = writer.write_table_bundle(
            "benchmark_tables",
            [{"engine": "graph_based", "compression_ratio": 0.25}],
            title="Compression Benchmark",
        )

        assert {path.suffix for path in paths} == {".md", ".csv", ".json", ".tex"}
        assert all(path.exists() for path in paths)


class TestPhase3EvaluationAndCli:
    def test_evaluation_suite_reports_builtins_and_unavailable_optionals(self) -> None:
        reports = DatasetEvaluationSuite(Settings()).evaluate()
        by_name = {report.dataset_name: report for report in reports}

        assert by_name["Iris"].status == "evaluated"
        assert by_name["Wine"].status == "evaluated"
        assert by_name["Breast Cancer"].status == "evaluated"
        assert by_name["Titanic"].status == "unavailable"

    def test_cli_phase3_commands_write_outputs(self, tmp_path) -> None:
        dataset_path = tmp_path / "dataset.csv"
        _dataset().to_csv(dataset_path, index=False)

        runner = CliRunner()
        benchmark_path = tmp_path / "benchmark.md"
        ablation_path = tmp_path / "ablation.csv"
        graph_path = tmp_path / "graph.mmd"
        paper_assets_dir = tmp_path / "paper_assets"

        benchmark_result = runner.invoke(
            app,
            ["benchmark", str(dataset_path), "--log-level", "ERROR", "--output", str(benchmark_path)],
        )
        ablation_result = runner.invoke(
            app,
            ["ablate", str(dataset_path), "--log-level", "ERROR", "--output", str(ablation_path)],
        )
        visualize_result = runner.invoke(
            app,
            ["visualize", str(dataset_path), "--log-level", "ERROR", "--output", str(graph_path)],
        )
        paper_assets_result = runner.invoke(
            app,
            ["paper-assets", str(dataset_path), "--log-level", "ERROR", "--output-dir", str(paper_assets_dir)],
        )

        assert benchmark_result.exit_code == 0
        assert ablation_result.exit_code == 0
        assert visualize_result.exit_code == 0
        assert paper_assets_result.exit_code == 0
        assert "Method Comparison" in benchmark_path.read_text()
        assert "Knowledge Objects" in benchmark_path.read_text()
        assert "Graph Density" in benchmark_path.read_text()
        assert "Compression Ratio" in benchmark_path.read_text()
        assert "variant" in ablation_path.read_text()
        assert "flowchart LR" in graph_path.read_text()
        assert (paper_assets_dir / "benchmark_tables.md").exists()
        assert (paper_assets_dir / "method_comparison.md").exists()
        assert (paper_assets_dir / "reasoning_statistics.csv").exists()

    def test_verbose_cli_shows_analyst_facing_briefs(self, tmp_path) -> None:
        dataset_path = tmp_path / "dataset.csv"
        _dataset().to_csv(dataset_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["investigate", str(dataset_path), "--log-level", "ERROR", "--verbose"],
        )

        assert result.exit_code == 0
        assert "Summary" in result.output
        assert "Likely Causes" in result.output
        assert "Evidence Strength" in result.output
        assert "Provenance Chain" in result.output
        assert "Recommended Validation" in result.output
        assert "Executive Summary" in result.output
