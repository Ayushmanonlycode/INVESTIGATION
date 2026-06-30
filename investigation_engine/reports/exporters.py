"""Reusable report export helpers for research assets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from investigation_engine.utils.deterministic import stable_payload


def _normalize_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return stable_payload(value)
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True))
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_value(row.get(key, "")) for key in fieldnames})
    return path


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows available._"
    headers = sorted({key for row in rows for key in row})
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(_normalize_value(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_markdown(path: Path, rows: list[dict[str, Any]], *, title: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = markdown_table(rows)
    if title:
        content = f"# {title}\n{content}"
    path.write_text(content)
    return path


def latex_table(rows: list[dict[str, Any]], *, caption: str | None = None, label: str | None = None) -> str:
    if not rows:
        return "% No rows available."
    headers = sorted({key for row in rows for key in row})
    alignment = "l" * len(headers)
    lines = [r"\begin{table}[ht]", r"\centering", rf"\begin{{tabular}}{{{alignment}}}", r"\hline"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(_normalize_value(row.get(header, "")) for header in headers) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex(path: Path, rows: list[dict[str, Any]], *, caption: str | None = None, label: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_table(rows, caption=caption, label=label))
    return path


def write_report(path: Path, rows: list[dict[str, Any]], *, title: str | None = None) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return write_json(path, rows)
    if suffix == ".csv":
        return write_csv(path, rows)
    if suffix in {".md", ".markdown"}:
        return write_markdown(path, rows, title=title)
    if suffix == ".tex":
        return write_latex(path, rows, caption=title, label=title.lower().replace(" ", "_") if title else None)
    raise ValueError(f"Unsupported report extension: {path.suffix}")


def flatten_nested_dict(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        next_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_nested_dict(next_key, value))
        else:
            flattened[next_key] = value
    return flattened
