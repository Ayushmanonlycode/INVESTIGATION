"""Helpers for writing publication-ready paper assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from investigation_engine.reports.exporters import write_csv, write_json, write_latex, write_markdown


class PaperAssetsWriter:
    """Writes deterministic publication-ready asset bundles."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def write_table_bundle(self, stem: str, rows: list[dict[str, Any]], *, title: str) -> list[Path]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            write_markdown(self._output_dir / f"{stem}.md", rows, title=title),
            write_csv(self._output_dir / f"{stem}.csv", rows),
            write_json(self._output_dir / f"{stem}.json", rows),
            write_latex(self._output_dir / f"{stem}.tex", rows, caption=title, label=stem),
        ]
        return paths
