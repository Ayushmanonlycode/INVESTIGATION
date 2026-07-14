"""Dataset evaluation suite for publication-oriented comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from investigation_engine.config.settings import Settings
from investigation_engine.core.engine import InvestigationEngine


@dataclass(frozen=True, slots=True)
class DatasetEvaluationReport:
    dataset_name: str
    status: str
    runtime_seconds: float
    findings: int
    evidence_units: int
    knowledge_objects: int
    hypotheses: int
    investigations: int
    compression_ratio: float
    average_confidence: float
    graph_density: float
    community_count: int
    modularity: float
    note: str


class DatasetEvaluationSuite:
    """Evaluates the engine across built-in and locally available datasets."""

    OPTIONAL_DATASETS: tuple[str, ...] = ("Titanic", "Adult Income", "UNSW-NB15", "CICIDS2017", "V2X")

    def __init__(self, config: Settings | None = None) -> None:
        self._config = config or Settings()
        self._engine = InvestigationEngine(self._config)

    def evaluate(self, search_paths: list[Path] | None = None) -> list[DatasetEvaluationReport]:
        reports = [
            self._evaluate_frame("Iris", self._frame_from_sklearn(load_iris(as_frame=True))),
            self._evaluate_frame("Wine", self._frame_from_sklearn(load_wine(as_frame=True))),
            self._evaluate_frame("Breast Cancer", self._frame_from_sklearn(load_breast_cancer(as_frame=True))),
        ]
        for dataset_name in self.OPTIONAL_DATASETS:
            dataset_path = self._discover_local_dataset(dataset_name, search_paths or [Path.cwd()])
            if dataset_path is None:
                reports.append(DatasetEvaluationReport(
                    dataset_name=dataset_name,
                    status="unavailable",
                    runtime_seconds=0.0,
                    findings=0,
                    evidence_units=0,
                    knowledge_objects=0,
                    hypotheses=0,
                    investigations=0,
                    compression_ratio=0.0,
                    average_confidence=0.0,
                    graph_density=0.0,
                    community_count=0,
                    modularity=0.0,
                    note="Dataset not found locally; skipped deterministically.",
                ))
                continue
            reports.append(self._evaluate_frame(dataset_name, self._load_local_frame(dataset_path), note=str(dataset_path)))
        return reports

    @staticmethod
    def to_rows(reports: list[DatasetEvaluationReport]) -> list[dict[str, object]]:
        return [asdict(report) for report in reports]

    def _evaluate_frame(self, name: str, frame: pd.DataFrame, *, note: str = "built_in") -> DatasetEvaluationReport:
        started = time.perf_counter()
        result = self._engine.investigate_dataframe(frame, name=name)
        runtime_seconds = round(time.perf_counter() - started, 6)
        metrics = result.reasoning_metrics or {}
        return DatasetEvaluationReport(
            dataset_name=name,
            status="evaluated",
            runtime_seconds=runtime_seconds,
            findings=len(result.findings),
            evidence_units=len(result.evidence_units),
            knowledge_objects=len(result.knowledge_objects),
            hypotheses=len(result.hypotheses),
            investigations=len(result.investigations),
            compression_ratio=float(metrics.get("compression", {}).get("overall_compression_factor", 0.0)),
            average_confidence=float(metrics.get("reasoning", {}).get("average_confidence", 0.0)),
            graph_density=float(metrics.get("graph", {}).get("density", 0.0)),
            community_count=int(metrics.get("graph", {}).get("communities", 0)),
            modularity=float(metrics.get("graph", {}).get("modularity", 0.0)),
            note=note,
        )

    @staticmethod
    def _frame_from_sklearn(bundle: object) -> pd.DataFrame:
        frame = getattr(bundle, "frame", None)
        if isinstance(frame, pd.DataFrame):
            return frame
        data = getattr(bundle, "data")
        target = getattr(bundle, "target")
        if isinstance(data, pd.DataFrame):
            output = data.copy()
        else:
            output = pd.DataFrame(data)
        output["target"] = target
        return output

    @staticmethod
    def _discover_local_dataset(dataset_name: str, search_paths: list[Path]) -> Path | None:
        aliases = {
            "Titanic": ("titanic.csv", "titanic.xlsx"),
            "Adult Income": ("adult.csv", "adult_income.csv"),
            "UNSW-NB15": ("unsw_nb15.csv", "UNSW-NB15.csv"),
            "CICIDS2017": ("cicids2017.csv", "CICIDS2017.csv"),
            "V2X": ("v2x.csv", "V2X.csv"),
        }
        for search_path in search_paths:
            for alias in aliases.get(dataset_name, ()):
                matches = sorted(search_path.rglob(alias))
                if matches:
                    return matches[0]
        return None

    @staticmethod
    def _load_local_frame(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported dataset file: {path}")
