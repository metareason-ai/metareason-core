"""Main CLI entry point for MetaReason."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.traceback import install

from .config import config_group
from .templates import template_group

# Install rich tracebacks for better error reporting
install(show_locals=True)

console = Console()


@click.group(name="metareason")
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing configuration files",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_dir: Optional[Path]) -> None:
    """MetaReason CLI - A framework for LLM sampling and oracle evaluation.

    MetaReason provides tools for systematic evaluation of language models
    through declarative YAML configurations, statistical sampling, and
    comprehensive oracle-based assessment.
    """
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_dir"] = config_dir

    if verbose:
        console.print("[dim]MetaReason CLI initialized[/dim]")


# Register command groups
cli.add_command(config_group)
cli.add_command(template_group)


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format",
)
@click.pass_context
def info(ctx: click.Context, output_format: str) -> None:
    """Show system information and configuration."""
    import json
    from datetime import datetime

    import yaml as pyyaml

    from metareason import __version__
    from metareason.config.cache import get_global_cache, is_caching_enabled
    from metareason.config.environment import get_environment_info

    info_data: Dict[str, Any] = {
        "metareason": {"version": __version__, "timestamp": datetime.now().isoformat()},
        "system": {
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "platform": sys.platform,
            "executable": sys.executable,
        },
        "cache": {
            "enabled": is_caching_enabled(),
            "stats": get_global_cache().get_stats() if is_caching_enabled() else None,
        },
        "environment": get_environment_info(),
    }

    if output_format == "json":
        console.print_json(json.dumps(info_data, indent=2, default=str))
    elif output_format == "yaml":
        console.print(pyyaml.dump(info_data, default_flow_style=False))
    else:
        # Text format
        console.print(f"🔧 [bold blue]MetaReason CLI[/bold blue] v{__version__}")
        console.print(
            f"🐍 Python {info_data['system']['python_version']} "
            f"on {info_data['system']['platform']}"
        )

        if info_data["cache"]["enabled"]:
            stats = info_data["cache"]["stats"]
            console.print(
                f"💾 Cache: {stats['active_entries']} active, "
                f"{stats['expired_entries']} expired"
            )
        else:
            console.print("💾 Cache: disabled")

        env_info = info_data["environment"]
        mr_vars = len(env_info["metareason_vars"])
        console.print(f"🌍 Environment: {mr_vars} MetaReason variables found")


@cli.command()
@click.option(
    "--spec-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the specification file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file for results",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for results and dashboard",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "parquet", "html", "dashboard"]),
    default="json",
    help="Output format",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without executing"
)
@click.option(
    "--max-concurrent",
    default=10,
    help="Maximum concurrent requests",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def run(
    ctx: click.Context,
    spec_file: Path,
    output: Optional[Path],
    output_dir: Optional[Path],
    output_format: str,
    dry_run: bool,
    max_concurrent: int,
    verbose: bool,
) -> None:
    """Run an evaluation using the specified specification file."""
    import asyncio
    from metareason.config import load_yaml_config
    from metareason.pipeline import PipelineRunner
    from metareason.results import ResultExporter, ResultFormatter
    from metareason.visualization import ResultVisualizer

    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    try:
        # Load configuration
        config = load_yaml_config(spec_file)
        console.print(f"📄 [bold]Loaded specification:[/bold] {config.spec_id}")

        # Create pipeline runner
        runner = PipelineRunner(config, max_concurrent=max_concurrent)

        if dry_run:
            # Create and display execution plan
            console.print("🗓️  [bold blue]Creating execution plan...[/bold blue]")
            plan = asyncio.run(runner.create_execution_plan())

            formatter = ResultFormatter(console)
            plan_text = formatter.format_execution_plan(plan)
            console.print(plan_text)
            return

        # Run the evaluation
        console.print("🚀 [bold green]Starting evaluation pipeline...[/bold green]")

        with console.status("[bold green]Running evaluation..."):
            result = asyncio.run(runner.run())

        # Display summary
        console.print("\n" + "=" * 60)
        formatter = ResultFormatter(console)
        summary_text = formatter.format_summary(result)
        console.print(summary_text)

        # Display detailed tables
        if verbose:
            console.print(formatter.create_summary_table(result))
            console.print(formatter.create_step_results_table(result))
            if result.oracle_results:
                console.print(formatter.create_oracle_results_table(result))

        # Export results
        if output or output_dir:
            console.print("\n💾 [bold blue]Exporting results...[/bold blue]")
            exporter = ResultExporter()

            if output_dir:
                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)

                if output_format == "dashboard":
                    # Generate complete dashboard
                    visualizer = ResultVisualizer()
                    dashboard_path = visualizer.generate_dashboard(result, output_dir)
                    console.print(f"📊 Dashboard saved to: {dashboard_path}")

                    # Also save JSON for data access
                    json_path = output_dir / "results.json"
                    exporter.export_json(result, json_path)
                    console.print(f"📄 Raw data saved to: {json_path}")
                else:
                    # Export to specified format in directory
                    file_name = f"results.{output_format}"
                    result_path = output_dir / file_name
                    _export_result(exporter, result, result_path, output_format)
                    console.print(f"💾 Results saved to: {result_path}")

            elif output:
                # Export to specific file
                _export_result(exporter, result, output, output_format)
                console.print(f"💾 Results saved to: {output}")

        console.print(
            "\n✅ [bold green]Evaluation completed successfully![/bold green]"
        )
        console.print(
            f"📊 Overall success rate: [bold]{result.overall_success_rate:.1%}[/bold]"
        )

    except Exception as e:
        console.print(f"\n❌ [red]Error during evaluation:[/red] {e}")
        if verbose:
            import traceback

            console.print("[red]" + traceback.format_exc() + "[/red]")
            
        sys.exit(1)


def _export_result(exporter, result, path: Path, format: str) -> None:
    """Export result to specified format."""
    if format == "json":
        exporter.export_json(result, path)
    elif format == "csv":
        exporter.export_csv(result, path)
    elif format == "parquet":
        exporter.export_parquet(result, path)
    elif format == "html":
        exporter.export_html_report(result, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    cli()
