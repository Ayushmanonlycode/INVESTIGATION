"""Investigation Engine CLI.

Provides a command-line interface for running investigations on datasets.
Built with Typer for a rich, user-friendly terminal experience.

Usage:
    investigation-engine investigate data.csv
    investigation-engine investigate data.xlsx --modules outlier,correlation
    investigation-engine investigate "postgresql://..." --table users
    investigation-engine info
"""

from __future__ import annotations

import json
import sys
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
    """Parse comma-separated module names into a list."""
    if modules_str is None:
        return None
    return [m.strip() for m in modules_str.split(",") if m.strip()]


@app.command()
def investigate(
    source: Annotated[
        str,
        typer.Argument(help="Path to CSV/XLSX file or PostgreSQL connection string."),
    ],
    modules: Annotated[
        Optional[str],
        typer.Option(
            "--modules",
            "-m",
            help="Comma-separated list of module names to run.",
        ),
    ] = None,
    table_name: Annotated[
        Optional[str],
        typer.Option(
            "--table",
            "-t",
            help="Table name for PostgreSQL sources.",
        ),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option(
            "--query",
            "-q",
            help="SQL query for PostgreSQL sources.",
        ),
    ] = None,
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to write JSON output.",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Logging level.",
        ),
    ] = "INFO",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed findings in console.",
        ),
    ] = False,
) -> None:
    """Run an investigation on a dataset.

    Loads the dataset, executes all applicable investigation modules,
    and outputs structured findings.
    """
    from investigation_engine.config.settings import Settings, LoggingSettings, EngineSettings
    from investigation_engine.core.engine import InvestigationEngine

    # Build configuration
    config = Settings(
        logging=LoggingSettings(log_level=log_level.upper()),
    )

    engine = InvestigationEngine(config=config)

    console.print(
        Panel(
            f"[bold]Investigating:[/bold] {source}",
            title="[bold blue]Investigation Engine[/bold blue]",
            border_style="blue",
        )
    )

    try:
        result = engine.investigate(
            source,
            modules=_parse_modules(modules),
            table_name=table_name,
            query=query,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    # Display summary
    _display_summary(result, verbose)

    # Write output if requested
    if output_file:
        output_data = result.model_dump(mode="json")
        output_file.write_text(json.dumps(output_data, indent=2, default=str))
        console.print(f"\n[green]Results written to:[/green] {output_file}")


def _display_summary(result: "InvestigationResult", verbose: bool = False) -> None:  # type: ignore[name-defined]  # noqa: F821
    """Display investigation results in the terminal."""
    from investigation_engine.models.severity import Severity

    # Dataset info
    info = result.dataset_info
    console.print(f"\n[bold]Dataset:[/bold] {info.name}")
    console.print(f"[dim]Rows: {info.row_count:,} | Columns: {info.column_count} | Memory: {info.memory_usage_bytes / (1024*1024):.1f} MB[/dim]")

    # Execution summary
    console.print(
        f"\n[bold]Modules:[/bold] "
        f"[green]{len(result.modules_executed)} executed[/green] | "
        f"[red]{len(result.modules_failed)} failed[/red] | "
        f"[yellow]{len(result.modules_skipped)} skipped[/yellow]"
    )
    console.print(f"[bold]Duration:[/bold] {result.duration_seconds:.3f}s")

    # Severity breakdown
    severity_summary = result.get_severity_summary()
    severity_colors = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "blue",
        "info": "dim",
    }

    severity_parts = []
    for sev_name, count in severity_summary.items():
        if count > 0:
            color = severity_colors.get(sev_name, "white")
            severity_parts.append(f"[{color}]{sev_name.upper()}: {count}[/{color}]")

    if severity_parts:
        console.print(f"\n[bold]Findings ({result.total_findings}):[/bold] " + " | ".join(severity_parts))
    else:
        console.print("\n[dim]No findings produced.[/dim]")

    # Fused Investigations summary
    console.print(f"[bold]Fused Investigations:[/bold] [green]{len(result.investigations)} unified hypothesis/hypotheses synthesized[/green]")

    # Investigations table
    if result.investigations and verbose:
        table = Table(
            title="Synthesized Investigations (Evidence Fusion)",
            show_lines=True,
            title_style="bold green",
        )
        table.add_column("Priority", style="bold magenta", justify="right", width=10)
        table.add_column("Title", style="bold white", width=40)
        table.add_column("Confidence", justify="right", width=12)
        table.add_column("Evidence Count", justify="right", width=15)
        table.add_column("Columns", width=25)

        for inv in result.investigations:
            # Skip summary findings from high-priority list to keep it focused
            if inv.metadata.get("original_category") == "structural_health_score":
                continue
            table.add_row(
                f"{inv.priority:.1f}",
                inv.title,
                f"{inv.confidence:.0%}",
                f"{len(inv.supporting_findings)} finding(s)",
                ", ".join(inv.affected_columns[:3])
                + ("..." if len(inv.affected_columns) > 3 else ""),
            )

        console.print()
        console.print(table)

        # Print the health score summary finding specifically
        for inv in result.investigations:
            if inv.metadata.get("original_category") == "structural_health_score":
                console.print(Panel(
                    f"[bold]{inv.title}[/bold]\n{inv.summary}",
                    title="[bold green]Structural Health Summary[/bold green]",
                    border_style="green",
                ))


@app.command()
def info() -> None:
    """Show registered modules and engine information."""
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
        tags = ", ".join(getattr(module_cls, "tags", []))
        table.add_row(
            name,
            getattr(module_cls, "version", "?"),
            getattr(module_cls, "description", ""),
            tags,
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold]Total modules:[/bold] {len(registry)}")


@app.command()
def schema() -> None:
    """Output the Finding JSON schema."""
    from investigation_engine.models.finding import Finding

    schema_json = json.dumps(Finding.model_json_schema(), indent=2)
    console.print_json(schema_json)


if __name__ == "__main__":
    app()
