"""Convenience entrypoint for dataset evaluation."""

from __future__ import annotations

from pathlib import Path

from investigation_engine.config.settings import Settings
from investigation_engine.evaluation import DatasetEvaluationSuite
from investigation_engine.reports.exporters import write_report


def run_evaluation(output_path: Path | None = None) -> list[dict[str, object]]:
    suite = DatasetEvaluationSuite(Settings())
    reports = suite.evaluate()
    rows = suite.to_rows(reports)
    if output_path is not None:
        write_report(output_path, rows, title="Dataset Evaluation")
    return rows
