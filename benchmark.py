"""Convenience entrypoint for compression benchmarking."""

from __future__ import annotations

from pathlib import Path

from investigation_engine.cli.main import _build_config, _load_result_for_source
from investigation_engine.reasoning.comparison import ReasoningMethodComparator
from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine
from investigation_engine.reports.exporters import markdown_table, write_report


def run_benchmark(source: str, output_path: Path | None = None) -> dict[str, dict[str, object]]:
    config = _build_config("INFO")
    result = _load_result_for_source(source, config)
    findings = [
        finding
        for finding in result.findings
        if finding.metadata.get("category") != "structural_health_score"
    ]
    benchmark = EvidenceCompressionBenchmark(config.evidence_compression)
    reports = benchmark.benchmark(
        findings,
        {
            "graph_based": GraphBasedEvidenceCompressionEngine(config.evidence_compression),
            "rule_based": RuleBasedEvidenceCompressionEngine(),
        },
    )
    payload = EvidenceCompressionBenchmark.to_json(reports)
    if output_path is not None:
        suffix = output_path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            output_path.write_text(
                "# Method Comparison\n"
                + markdown_table(ReasoningMethodComparator.to_rows(ReasoningMethodComparator(config).compare(result.findings), humanize=True))
                + "\n\n# Compression Benchmark\n"
                + EvidenceCompressionBenchmark.to_markdown(reports)
            )
        elif suffix == ".tex":
            output_path.write_text(EvidenceCompressionBenchmark.to_latex(reports))
        else:
            write_report(output_path, EvidenceCompressionBenchmark.to_rows(reports), title="Compression Benchmark")
    return payload
