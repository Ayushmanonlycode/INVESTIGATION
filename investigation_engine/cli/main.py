"""Investigation Engine CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="investigation-engine",
    help="AI Investigation Intelligence Engine — Autonomous dataset investigation.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()


def _parse_modules(modules_str: str | None) -> list[str] | None:
    if modules_str is None:
        return None
    return [item.strip() for item in modules_str.split(",") if item.strip()]


def _build_config(log_level: str) -> "Settings":  # type: ignore[name-defined]  # noqa: F821
    from investigation_engine.config.settings import LoggingSettings, Settings

    return Settings(logging=LoggingSettings(log_level=log_level.upper()))


def _load_result_for_source(
    source: str,
    config: "Settings",  # type: ignore[name-defined]  # noqa: F821
    *,
    modules: list[str] | None = None,
    table_name: str | None = None,
    query: str | None = None,
) -> "InvestigationResult":  # type: ignore[name-defined]  # noqa: F821
    from investigation_engine.core.engine import InvestigationEngine

    engine = InvestigationEngine(config=config)
    return engine.investigate(
        source,
        modules=modules,
        table_name=table_name,
        query=query,
    )


@app.command()
def investigate(
    source: Annotated[
        str,
        typer.Argument(help="Path to CSV/XLSX file or PostgreSQL connection string."),
    ],
    modules: Annotated[
        Optional[str],
        typer.Option("--modules", "-m", help="Comma-separated list of module names to run."),
    ] = None,
    table_name: Annotated[
        Optional[str],
        typer.Option("--table", "-t", help="Table name for PostgreSQL sources."),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option("--query", "-q", help="SQL query for PostgreSQL sources."),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write exported output."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed findings in console.")] = False,
    explain: Annotated[bool, typer.Option("--explain", help="Render complete reasoning provenance trees.")] = False,
    metrics: Annotated[bool, typer.Option("--metrics", help="Display reasoning and compression metrics.")] = False,
    benchmark: Annotated[bool, typer.Option("--benchmark", help="Compare rule-based and graph-based compression.")] = False,
) -> None:
    from investigation_engine.reasoning.comparison import ReasoningMethodComparator
    from investigation_engine.reasoning.provenance.renderer import ProvenanceTreeRenderer
    from investigation_engine.reports.exporters import flatten_nested_dict, markdown_table, write_json, write_report

    config = _build_config(log_level)

    console.print(
        Panel(
            f"[bold]Investigating:[/bold] {source}",
            title="[bold blue]Investigation Engine[/bold blue]",
            border_style="blue",
        )
    )

    try:
        result = _load_result_for_source(
            source,
            config,
            modules=_parse_modules(modules),
            table_name=table_name,
            query=query,
        )
    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1)

    _display_summary(result, verbose)
    if result.reasoning_metrics:
        _display_reasoning_flow(result.reasoning_metrics)
    if explain or verbose:
        _display_investigation_dashboards(result)
    if explain and result.provenance_trees:
        _display_reasoning_objects(result)
        _display_explain(result.provenance_trees)
    if verbose:
        _display_confidence_breakdowns(result.investigations)
    if metrics and result.reasoning_metrics:
        _display_metrics(result.reasoning_metrics)
    if benchmark:
        comparison_reports = ReasoningMethodComparator(config).compare(result.findings)
        _display_method_comparison(comparison_reports)
        _display_benchmark(config, result)

    if output_file:
        suffix = output_file.suffix.lower()
        if benchmark and suffix in {".json", ".csv", ".md", ".markdown", ".tex"}:
            if suffix in {".md", ".markdown"}:
                output_file.write_text(
                    "# Method Comparison\n"
                    + markdown_table(ReasoningMethodComparator.to_rows(comparison_reports, humanize=True))
                    + "\n\n# Compression Benchmark\n"
                    + _display_benchmark(config, result)
                )
            elif suffix == ".tex":
                from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
                output_file.write_text(
                    EvidenceCompressionBenchmark.to_latex(_benchmark_reports_from_result(result))
                )
            else:
                write_report(output_file, list((result.benchmark_reports or {}).values()), title="Compression Benchmark")
        elif metrics and suffix in {".json", ".csv", ".md", ".markdown", ".tex"}:
            rows = [
                {"section": section, **flatten_nested_dict("", payload)}
                for section, payload in (result.reasoning_metrics or {}).items()
                if isinstance(payload, dict)
            ]
            write_report(output_file, rows, title="Reasoning Metrics")
        elif explain and suffix in {".json", ".md", ".markdown"}:
            if suffix == ".json":
                write_json(output_file, result.provenance_trees or [])
            else:
                output_file.write_text(ProvenanceTreeRenderer().render_terminal(result.provenance_trees or []))
        else:
            write_json(output_file, result.model_dump(mode="json"))
        console.print(f"\n[green]Results written to:[/green] {output_file}")


def _display_summary(result: "InvestigationResult", verbose: bool = False) -> None:  # type: ignore[name-defined]  # noqa: F821
    info = result.dataset_info
    console.print(f"\n[bold]Dataset:[/bold] {info.name}")
    console.print(f"[dim]Rows: {info.row_count:,} | Columns: {info.column_count} | Memory: {info.memory_usage_bytes / (1024 * 1024):.1f} MB[/dim]")
    console.print(
        f"\n[bold]Modules:[/bold] "
        f"[green]{len(result.modules_executed)} executed[/green] | "
        f"[red]{len(result.modules_failed)} failed[/red] | "
        f"[yellow]{len(result.modules_skipped)} skipped[/yellow]"
    )
    console.print(f"[bold]Duration:[/bold] {result.duration_seconds:.3f}s")

    severity_summary = result.get_severity_summary()
    severity_colors = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "blue",
        "info": "dim",
    }
    severity_parts = [
        f"[{severity_colors.get(name, 'white')}]{name.upper()}: {count}[/{severity_colors.get(name, 'white')}]"
        for name, count in severity_summary.items()
        if count > 0
    ]
    if severity_parts:
        console.print(f"\n[bold]Findings ({result.total_findings}):[/bold] " + " | ".join(severity_parts))
    else:
        console.print("\n[dim]No findings produced.[/dim]")

    console.print(
        f"[bold]Evidence Units:[/bold] [green]{len(result.evidence_units)}[/green] | "
        f"[bold]Knowledge Objects:[/bold] [green]{len(result.knowledge_objects)}[/green] | "
        f"[bold]Hypotheses:[/bold] [green]{len(result.hypotheses)}[/green]"
    )
    console.print(
        f"[bold]Investigation Queue:[/bold] "
        f"[green]{len(result.investigations)} prioritized investigation(s) synthesized[/green]"
    )
    if result.investigations:
        _display_executive_summary(result)

    dashboard = result.reasoning_metrics.get("dashboard", {}) if result.reasoning_metrics else {}
    if isinstance(dashboard, dict) and dashboard:
        table = Table(title="Reasoning Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Findings", str(len(result.findings)))
        table.add_row("Evidence Units", str(len(result.evidence_units)))
        table.add_row("Knowledge Objects", str(len(result.knowledge_objects)))
        table.add_row("Hypotheses", str(len(result.hypotheses)))
        table.add_row("Investigations", str(len(result.investigations)))
        for key, value in dashboard.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        console.print()
        console.print(table)

    if result.investigations and verbose:
        table = Table(
            title="Synthesized Investigations (Evidentiary Reasoning)",
            show_lines=True,
            title_style="bold green",
        )
        table.add_column("Priority", style="bold magenta", justify="right", width=10)
        table.add_column("Title", style="bold white", width=40)
        table.add_column("Confidence", justify="right", width=12)
        table.add_column("Evidence Units", justify="right", width=15)
        table.add_column("Columns", width=25)
        for investigation in result.investigations:
            if investigation.metadata.get("original_category") == "structural_health_score":
                continue
            table.add_row(
                f"{investigation.priority:.1f}",
                investigation.title,
                f"{investigation.confidence:.0%}",
                f"{len(investigation.supporting_evidence)} unit(s)",
                ", ".join(investigation.affected_columns[:3]) + ("..." if len(investigation.affected_columns) > 3 else ""),
            )
        console.print()
        console.print(table)


def _display_explain(provenance_trees: list[dict[str, object]]) -> None:
    from investigation_engine.reasoning.provenance.renderer import ProvenanceTreeRenderer

    console.print()
    console.print(Panel(
        ProvenanceTreeRenderer().render_terminal(provenance_trees),
        title="[bold blue]Reasoning Lineage[/bold blue]",
        border_style="blue",
    ))


def _display_reasoning_objects(result: "InvestigationResult") -> None:  # type: ignore[name-defined]  # noqa: F821
    if result.evidence_units:
        console.print()
        console.print("[bold blue]Structured Evidence Units[/bold blue]")
        for evidence in result.evidence_units:
            body = (
                f"[bold]Category[/bold]\n{evidence.category}\n\n"
                f"[bold]Confidence[/bold]\n{evidence.confidence:.0%}\n\n"
                f"[bold]Supporting Findings[/bold]\n{len(evidence.supporting_findings)}\n\n"
                f"[bold]Affected Columns[/bold]\n{_format_list(evidence.affected_columns)}\n\n"
                f"[bold]Community[/bold]\n{evidence.community_id or '-'}\n\n"
                f"[bold]Strength[/bold]\n{evidence.strength:.1f}\n\n"
                f"[bold]Provenance[/bold]\n{_format_mapping(evidence.provenance)}"
            )
            console.print(
                Panel(
                    body,
                    title=f"[bold green]Evidence Unit {evidence.evidence_id}[/bold green]",
                    border_style="green",
                )
            )
    if result.knowledge_objects:
        console.print()
        console.print("[bold blue]Knowledge Objects[/bold blue]")
        for knowledge in result.knowledge_objects:
            body = (
                f"[bold]Concept[/bold]\n{knowledge.concept}\n\n"
                f"[bold]Summary[/bold]\n{knowledge.description}\n\n"
                f"[bold]Confidence[/bold]\n{knowledge.confidence:.0%}\n\n"
                f"[bold]Supporting Evidence Units[/bold]\n{_format_list(knowledge.supporting_evidence)}\n\n"
                f"[bold]Related Concepts[/bold]\n{_format_list(knowledge.related_concepts)}\n\n"
                f"[bold]Affected Columns[/bold]\n{_format_list(knowledge.metadata.get('affected_columns', []))}\n\n"
                f"[bold]Provenance[/bold]\n{_format_mapping(knowledge.provenance)}"
            )
            console.print(
                Panel(
                    body,
                    title=f"[bold cyan]Knowledge Object {knowledge.knowledge_id}[/bold cyan]",
                    border_style="cyan",
                )
            )
    if result.hypotheses:
        table = Table(title="Hypotheses", show_lines=True)
        table.add_column("ID", style="white")
        table.add_column("Title", style="bold white")
        table.add_column("Confidence", justify="right")
        table.add_column("Knowledge Objects", justify="right")
        table.add_column("Unknown Evidence", justify="right")
        for item in result.hypotheses:
            table.add_row(
                item.hypothesis_id,
                item.title,
                f"{item.confidence:.0%}",
                str(len(item.supporting_knowledge_objects)),
                str(len(item.unknown_evidence)),
            )
        console.print()
        console.print(table)


def _display_reasoning_flow(reasoning_metrics: dict[str, object]) -> None:
    flow = reasoning_metrics.get("reasoning_flow", {})
    if not isinstance(flow, dict):
        return
    stages = flow.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return
    lines: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        lines.append(f"[bold]{stage.get('count', 0)} {stage.get('label', 'Stage')}[/bold]")
        if index < len(stages) - 1:
            lines.append(
                "[dim]"
                f"↓ {stage.get('reduction_percent', 0.0)}% reduction"
                f" | {stage.get('compression_factor', 0.0)}× compression"
                "[/dim]"
            )
    lines.append("")
    lines.append(
        f"[bold]Overall[/bold] {flow.get('overall_reduction_percent', 0.0)}% reduction"
        f" | {flow.get('overall_compression_factor', 0.0)}× compression"
    )
    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold blue]Reasoning Flow[/bold blue]",
            border_style="blue",
        )
    )


def _display_metrics(reasoning_metrics: dict[str, object]) -> None:
    console.print()
    for section_name in ("counts", "compression", "graph", "reasoning", "runtime", "memory", "dashboard", "reasoning_flow"):
        section = reasoning_metrics.get(section_name, {})
        if not isinstance(section, dict):
            continue
        table = Table(title=section_name.replace("_", " ").title(), show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for key, value in section.items():
            table.add_row(key.replace("_", " ").title(), json.dumps(value) if isinstance(value, dict) else str(value))
        console.print(table)
        console.print()


def _display_confidence_breakdowns(investigations: list["Investigation"]) -> None:  # type: ignore[name-defined]  # noqa: F821
    actionable = [
        investigation
        for investigation in investigations
        if investigation.metadata.get("original_category") != "structural_health_score"
    ]
    if not actionable:
        return
    table = Table(title="Investigation Confidence Breakdown", show_lines=True)
    table.add_column("Title", style="bold white", width=36)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Breakdown", style="white", width=70)
    for investigation in actionable:
        breakdown = " | ".join(
            f"{key.replace('_', ' ')}={value:+.2f}"
            for key, value in investigation.confidence_breakdown.items()
        )
        table.add_row(
            investigation.title,
            f"{investigation.confidence:.0%}",
            breakdown,
        )
    console.print()
    console.print(table)


def _display_executive_summary(result: "InvestigationResult") -> None:  # type: ignore[name-defined]  # noqa: F821
    actionable = [
        investigation
        for investigation in result.investigations
        if investigation.metadata.get("original_category") != "structural_health_score"
    ]
    if not actionable:
        return
    high_priority = sum(1 for investigation in actionable if investigation.priority >= 75.0)
    medium_priority = sum(1 for investigation in actionable if 50.0 <= investigation.priority < 75.0)
    lead = actionable[0]
    estimated_cause = (
        lead.likely_causes[0].cause
        if lead.likely_causes
        else (lead.possible_causes[0] if lead.possible_causes else "Unavailable")
    )
    first_action = lead.recommended_next_steps[0] if lead.recommended_next_steps else "Unavailable"
    body = (
        "[bold]Overall Investigation Summary[/bold]\n\n"
        f"Investigations Generated: {len(actionable)}\n"
        f"High Priority: {high_priority}\n"
        f"Medium Priority: {medium_priority}\n"
        f"Primary Issue: {lead.title}\n"
        f"Estimated Cause: {estimated_cause}\n"
        f"Affected Features: {len(lead.affected_columns)}\n"
        f"Confidence: {lead.confidence:.0%}\n"
        f"Recommended First Action: {first_action}"
    )
    console.print()
    console.print(Panel(body, title="[bold magenta]Executive Summary[/bold magenta]", border_style="magenta"))


def _format_list(values: object) -> str:
    if not values:
        return "None"
    if isinstance(values, list):
        return "\n".join(f"- {value}" for value in values) if values else "None"
    return str(values)


def _format_mapping(values: object) -> str:
    if not isinstance(values, dict) or not values:
        return "None"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def _format_reasoning_tree(tree: object) -> str:
    if not isinstance(tree, dict):
        return "Unavailable"
    lines = [f"Investigation {tree.get('id', '-')}: {tree.get('title', '-')}"]
    for hypothesis in tree.get("children", []):
        if not isinstance(hypothesis, dict):
            continue
        lines.append(f"  -> Hypothesis {hypothesis.get('id', '-')}: {hypothesis.get('title', '-')}")
        for knowledge in hypothesis.get("children", []):
            if not isinstance(knowledge, dict):
                continue
            lines.append(f"     -> Knowledge {knowledge.get('id', '-')}: {knowledge.get('title', '-')}")
            for evidence in knowledge.get("children", [])[:3]:
                if not isinstance(evidence, dict):
                    continue
                lines.append(f"        -> Evidence {evidence.get('id', '-')}: {evidence.get('title', '-')}")
    return "\n".join(lines)


def _summarize_findings(
    finding_ids: list[str],
    finding_lookup: dict[str, object],
) -> str:
    category_counts: dict[str, int] = {}
    for finding_id in finding_ids:
        finding = finding_lookup.get(finding_id)
        if finding is None:
            continue
        category = str(getattr(finding, "metadata", {}).get("category", "finding"))
        category_counts[category] = category_counts.get(category, 0) + 1
    if not category_counts:
        return "None"
    return "\n".join(
        f"- {count} {category.replace('_', ' ')} finding(s)"
        for category, count in sorted(
            category_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _display_investigation_dashboards(result: "InvestigationResult") -> None:  # type: ignore[name-defined]  # noqa: F821
    actionable = [
        investigation
        for investigation in result.investigations
        if investigation.metadata.get("original_category") != "structural_health_score"
    ]
    if not actionable:
        return
    console.print()
    evidence_lookup = {evidence.evidence_id: evidence for evidence in result.evidence_units}
    knowledge_lookup = {knowledge.knowledge_id: knowledge for knowledge in result.knowledge_objects}
    hypothesis_lookup = {hypothesis.hypothesis_id: hypothesis for hypothesis in result.hypotheses}
    finding_lookup = {finding.id: finding for finding in result.findings}
    tree_lookup = {
        str(tree.get("id")): tree
        for tree in (result.provenance_trees or [])
        if isinstance(tree, dict)
    }
    for investigation in actionable:
        evidence_lines: list[str] = []
        for evidence_id in investigation.supporting_evidence:
            evidence = evidence_lookup.get(evidence_id)
            if evidence is None:
                continue
            evidence_lines.append(
                f"{evidence.evidence_id}: {evidence.category} | conf={evidence.confidence:.0%} | "
                f"findings={len(evidence.supporting_findings)} | "
                f"columns={', '.join(evidence.affected_columns[:2]) or '-'} | "
                f"community={evidence.community_id or '-'} | strength={evidence.strength:.1f}"
            )
        knowledge_lines = [
            f"{knowledge_id}: {knowledge_lookup[knowledge_id].concept} ({knowledge_lookup[knowledge_id].confidence:.0%})"
            for knowledge_id in investigation.supporting_knowledge_objects
            if knowledge_id in knowledge_lookup
        ]
        cause_lines = []
        for cause in investigation.likely_causes:
            cause_lines.append(
                "\n".join(
                    [
                        f"Cause: {cause.cause}",
                        f"Confidence: {cause.confidence:.0%}",
                        f"Supported By: {_format_list(cause.supporting_evidence)}",
                        f"Supporting Findings: {_summarize_findings(cause.supporting_findings, finding_lookup)}",
                        f"Knowledge Objects: {_format_list(cause.knowledge_objects)}",
                        f"Contradicted By: {_format_list(cause.contradicting_evidence)}",
                        f"Unknown: {_format_list(cause.unknown_evidence)}",
                    ]
                )
            )
        likely_cause = "\n\n".join(cause_lines) or _format_list(investigation.possible_causes) or "No likely cause synthesized."
        validation = "; ".join(investigation.recommended_next_steps[:2]) or "No validation steps synthesized."
        impact = (
            f"Affects {len(investigation.affected_columns)} column(s), "
            f"{len(investigation.supporting_evidence)} evidence unit(s), "
            f"and {len(investigation.supporting_hypotheses)} hypothesis/hypotheses."
        )
        evidence_strength = investigation.evidence_strength_details
        evidence_strength_text = "Unavailable"
        if evidence_strength is not None:
            investigator_count = len(investigation.provenance.get("investigators", []))
            evidence_strength_text = (
                f"Supporting Findings: {len(evidence_strength.supporting_findings)}\n"
                f"Supporting Evidence Units: {len(evidence_strength.supporting_evidence_units)}\n"
                f"Investigators: {investigator_count}\n"
                f"Knowledge Objects: {len(investigation.supporting_knowledge_objects)}\n"
                f"Unknown Evidence: {evidence_strength.unknown_evidence}\n"
                f"Graph Cohesion: {evidence_strength.graph_cohesion:.2f}\n"
                f"Community Strength: {evidence_strength.community_strength:.2f}\n"
                f"Overall: {evidence_strength.rating}"
            )
        confidence_breakdown = "\n".join(
            f"{key.replace('_', ' ').title()}: {value:+.2f}"
            for key, value in investigation.confidence_breakdown.items()
        ) or "Unavailable"
        provenance_chain = (
            f"Investigation {investigation.investigation_id}"
            f" -> Hypotheses {len(investigation.supporting_hypotheses)}"
            f" -> Knowledge Objects {len(investigation.supporting_knowledge_objects)}"
            f" -> Evidence Units {len(investigation.supporting_evidence)}"
            f" -> Findings {len(investigation.supporting_findings)}"
            f" -> Investigators {len(investigation.provenance.get('investigators', []))}"
        )
        reasoning_tree = _format_reasoning_tree(tree_lookup.get(investigation.investigation_id))
        lead_hypotheses = [
            hypothesis_lookup[hypothesis_id].title
            for hypothesis_id in investigation.supporting_hypotheses
            if hypothesis_id in hypothesis_lookup
        ]
        body = (
            f"[bold]Summary[/bold]\n{investigation.summary}\n\n"
            f"[bold]Evidence Strength[/bold]\n{evidence_strength_text}\n\n"
            f"[bold]Confidence[/bold]\n{investigation.confidence:.0%}\n\n"
            f"[bold]Supporting Evidence[/bold]\n{_format_list(evidence_lines)}\n\n"
            f"[bold]Knowledge Objects[/bold]\n{_format_list(knowledge_lines)}\n\n"
            f"[bold]Hypothesis[/bold]\n{investigation.hypothesis}\n\n"
            f"[bold]Hypothesis Titles[/bold]\n{_format_list(lead_hypotheses)}\n\n"
            f"[bold]Likely Causes[/bold]\n{likely_cause}\n\n"
            f"[bold]Impact[/bold]\n{impact}\n\n"
            f"[bold]Recommended Validation[/bold]\n{validation}\n\n"
            f"[bold]Provenance Chain[/bold]\n{provenance_chain}\n\n"
            f"[bold]Confidence Breakdown[/bold]\n{confidence_breakdown}\n\n"
            f"[bold]Reasoning Tree[/bold]\n{reasoning_tree}"
        )
        console.print(Panel(body, title=f"[bold green]{investigation.title}[/bold green]", border_style="green"))


def _display_benchmark(config: "Settings", result: "InvestigationResult") -> str:  # type: ignore[name-defined]  # noqa: F821
    from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
    from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
    from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine

    findings = [
        finding
        for finding in result.findings
        if finding.metadata.get("category") != "structural_health_score"
    ]
    benchmark = EvidenceCompressionBenchmark(config.evidence_compression)
    graph_metrics = result.reasoning_metrics.get("graph", {}) if isinstance(result.reasoning_metrics, dict) else {}
    reports = benchmark.benchmark(
        findings,
        {
            "graph_based": GraphBasedEvidenceCompressionEngine(config.evidence_compression),
            "rule_based": RuleBasedEvidenceCompressionEngine(),
        },
        graph_density=float(graph_metrics.get("density", 0.0)) if graph_metrics else None,
        community_modularity=float(graph_metrics.get("modularity", 0.0)) if graph_metrics else None,
    )
    result.benchmark_reports = benchmark.to_json(reports)
    markdown = benchmark.to_markdown(reports)
    console.print()
    console.print(Panel(markdown, title="[bold blue]Benchmark[/bold blue]", border_style="blue"))
    return markdown


def _display_method_comparison(reports: list["MethodComparisonReport"]) -> None:  # type: ignore[name-defined]  # noqa: F821
    table = Table(title="Method Comparison", show_lines=True)
    table.add_column("Method", style="bold white")
    table.add_column("Findings", justify="right")
    table.add_column("Evidence", justify="right")
    table.add_column("Knowledge", justify="right")
    table.add_column("Hypotheses", justify="right")
    table.add_column("Investigations", justify="right")
    table.add_column("Compression", justify="right")
    table.add_column("Avg Evidence/Hyp.", justify="right")
    table.add_column("Avg Knowledge/Hyp.", justify="right")
    table.add_column("Avg Confidence", justify="right")
    table.add_column("Avg Community", justify="right")
    table.add_column("Largest Community", justify="right")
    table.add_column("Graph Density", justify="right")
    table.add_column("Depth", justify="right")
    for report in reports:
        table.add_row(
            report.method,
            str(report.findings),
            str(report.evidence_units),
            str(report.knowledge_objects),
            str(report.hypotheses),
            str(report.investigations),
            f"{report.compression_factor:.1f}×",
            f"{report.average_evidence_per_hypothesis:.2f}",
            f"{report.average_knowledge_per_hypothesis:.2f}",
            f"{report.average_confidence:.2f}",
            f"{report.average_community_size:.2f}",
            str(report.largest_community),
            f"{report.graph_density:.3f}",
            str(report.reasoning_depth),
        )
    console.print()
    console.print(table)


def _benchmark_reports_from_result(
    result: "InvestigationResult",  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, "CompressionBenchmarkReport"]:  # type: ignore[name-defined]  # noqa: F821
    from investigation_engine.reasoning.evidence.benchmark import CompressionBenchmarkReport

    return {
        name: CompressionBenchmarkReport(**payload)
        for name, payload in (result.benchmark_reports or {}).items()
    }


@app.command("benchmark")
def benchmark_command(
    source: Annotated[str, typer.Argument(help="Path to the dataset to benchmark.")],
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write benchmark output."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
) -> None:
    from investigation_engine.reasoning.comparison import ReasoningMethodComparator
    from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
    from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
    from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine
    from investigation_engine.reports.exporters import markdown_table, write_report

    config = _build_config(log_level)
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
    comparison_reports = ReasoningMethodComparator(config).compare(result.findings)
    _display_method_comparison(comparison_reports)
    console.print(Panel(benchmark.to_markdown(reports), title="[bold blue]Benchmark[/bold blue]", border_style="blue"))
    if output_file:
        suffix = output_file.suffix.lower()
        if suffix in {".md", ".markdown"}:
            output_file.write_text(
                "# Method Comparison\n"
                + markdown_table(ReasoningMethodComparator.to_rows(comparison_reports, humanize=True))
                + "\n\n# Compression Benchmark\n"
                + benchmark.to_markdown(reports)
            )
        elif suffix == ".tex":
            output_file.write_text(benchmark.to_latex(reports))
        else:
            write_report(output_file, benchmark.to_rows(reports), title="Compression Benchmark")
        console.print(f"\n[green]Benchmark written to:[/green] {output_file}")


@app.command("ablate")
def ablate_command(
    source: Annotated[str, typer.Argument(help="Path to the dataset to ablate.")],
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write ablation results."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
) -> None:
    from investigation_engine.reasoning.ablation import AblationEngine
    from investigation_engine.reports.exporters import markdown_table, write_report

    config = _build_config(log_level)
    result = _load_result_for_source(source, config)
    reports = AblationEngine(config).run(result.findings)
    rows = AblationEngine.to_rows(reports)
    console.print(Panel(markdown_table(rows), title="[bold blue]Ablation[/bold blue]", border_style="blue"))
    if output_file:
        write_report(output_file, rows, title="Ablation Results")
        console.print(f"\n[green]Ablation results written to:[/green] {output_file}")


@app.command("evaluate")
def evaluate_command(
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write dataset evaluation results."),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
) -> None:
    from investigation_engine.evaluation import DatasetEvaluationSuite
    from investigation_engine.reports.exporters import markdown_table, write_report

    config = _build_config(log_level)
    suite = DatasetEvaluationSuite(config)
    reports = suite.evaluate()
    rows = suite.to_rows(reports)
    console.print(Panel(markdown_table(rows), title="[bold blue]Dataset Evaluation[/bold blue]", border_style="blue"))
    if output_file:
        write_report(output_file, rows, title="Dataset Evaluation")
        console.print(f"\n[green]Evaluation results written to:[/green] {output_file}")


@app.command("visualize")
def visualize_command(
    source: Annotated[str, typer.Argument(help="Path to the dataset to visualize.")],
    graph_type: Annotated[
        str,
        typer.Option("--graph-type", help="Graph type: evidence, knowledge, hypothesis, investigation."),
    ] = "evidence",
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write the graph export."),
    ] = Path("reasoning_graph.mmd"),
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
) -> None:
    from investigation_engine.reports.visualization import ReasoningGraphExporter

    config = _build_config(log_level)
    result = _load_result_for_source(source, config)
    exporter = ReasoningGraphExporter()
    graph = exporter.build_graph(result, graph_type)  # type: ignore[arg-type]
    exporter.export(graph, output_file)
    console.print(f"[green]Visualization written to:[/green] {output_file}")


@app.command("paper-assets")
def paper_assets_command(
    source: Annotated[str, typer.Argument(help="Path to the dataset used to generate paper assets.")],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for publication-ready paper assets."),
    ] = Path("paper_assets"),
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Logging level.")] = "INFO",
) -> None:
    from investigation_engine.evaluation import DatasetEvaluationSuite
    from investigation_engine.reasoning.ablation import AblationEngine
    from investigation_engine.reasoning.comparison import ReasoningMethodComparator
    from investigation_engine.reasoning.evidence.benchmark import EvidenceCompressionBenchmark
    from investigation_engine.reasoning.evidence.compression import RuleBasedEvidenceCompressionEngine
    from investigation_engine.reasoning.evidence.graph_compressor import GraphBasedEvidenceCompressionEngine
    from investigation_engine.reports.exporters import flatten_nested_dict, write_csv
    from investigation_engine.reports.paper_assets import PaperAssetsWriter

    config = _build_config(log_level)
    result = _load_result_for_source(source, config)
    findings = [
        finding
        for finding in result.findings
        if finding.metadata.get("category") != "structural_health_score"
    ]
    benchmark = EvidenceCompressionBenchmark(config.evidence_compression)
    benchmark_reports = benchmark.benchmark(
        findings,
        {
            "graph_based": GraphBasedEvidenceCompressionEngine(config.evidence_compression),
            "rule_based": RuleBasedEvidenceCompressionEngine(),
        },
    )
    comparison_reports = ReasoningMethodComparator(config).compare(result.findings)
    ablation_reports = AblationEngine(config).run(result.findings)
    evaluation_suite = DatasetEvaluationSuite(config)
    evaluation_reports = evaluation_suite.evaluate()

    writer = PaperAssetsWriter(output_dir)
    writer.write_table_bundle("benchmark_tables", benchmark.to_rows(benchmark_reports, humanize=True), title="Compression Benchmark")
    writer.write_table_bundle("method_comparison", ReasoningMethodComparator.to_rows(comparison_reports, humanize=True), title="Method Comparison")
    writer.write_table_bundle("dataset_summary", evaluation_suite.to_rows(evaluation_reports), title="Dataset Evaluation")
    writer.write_table_bundle("ablation_results", AblationEngine.to_rows(ablation_reports), title="Ablation Results")
    reasoning_rows = [
        {"section": section, **flatten_nested_dict("", payload)}
        for section, payload in (result.reasoning_metrics or {}).items()
        if isinstance(payload, dict)
    ]
    writer.write_table_bundle("reasoning_statistics", reasoning_rows, title="Reasoning Statistics")
    write_csv(output_dir / "compression_statistics.csv", benchmark.to_rows(benchmark_reports))
    write_csv(
        output_dir / "graph_metrics.csv",
        [flatten_nested_dict("graph", result.reasoning_metrics.get("graph", {}))] if result.reasoning_metrics else [],
    )
    console.print(f"[green]Paper assets written to:[/green] {output_dir}")


@app.command()
def info() -> None:
    from investigation_engine.core.plugin import discover_modules, get_registered_modules

    discover_modules()
    registry = get_registered_modules()

    console.print(
        Panel(
            "[bold]AI Investigation Intelligence Engine[/bold]\n"
            "Autonomous dataset investigation with structured evidence output.",
            title="[bold blue]investigation-engine v0.1.0[/bold blue]",
            border_style="blue",
        )
    )

    if not registry:
        console.print("\n[yellow]No investigation modules registered.[/yellow]")
        console.print("[dim]Add modules to investigation_engine/modules/ and decorate with @register_module.[/dim]")
        return

    table = Table(title="Registered Investigation Modules", show_lines=True)
    table.add_column("Name", style="cyan", width=25)
    table.add_column("Version", width=10)
    table.add_column("Description", width=50)
    table.add_column("Tags", width=20)
    for name, module_cls in sorted(registry.items()):
        table.add_row(
            name,
            getattr(module_cls, "version", "?"),
            getattr(module_cls, "description", ""),
            ", ".join(getattr(module_cls, "tags", [])),
        )
    console.print()
    console.print(table)
    console.print(f"\n[bold]Total modules:[/bold] {len(registry)}")


@app.command()
def schema() -> None:
    from investigation_engine.models.finding import Finding

    console.print_json(json.dumps(Finding.model_json_schema(), indent=2))


if __name__ == "__main__":
    app()
