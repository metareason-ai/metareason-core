"""Main CLI entry point for MetaReason."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.traceback import install

from .config import config_group

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
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file for results",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without executing"
)
@click.pass_context
def run(
    ctx: click.Context, config_file: Path, output: Optional[Path], dry_run: bool
) -> None:
    """Run an evaluation using the specified configuration file."""
    from metareason.config import load_yaml_config

    try:
        config = load_yaml_config(config_file)

        if dry_run:
            console.print("[dim]Would run evaluation with configuration:[/dim]")
            console.print(f"📄 Config: {config_file}")
            console.print(f"🎯 Prompt: {config.prompt_id}")
            console.print(f"🔢 Variants: {config.n_variants}")
            console.print(
                f"📊 Sampling: {config.sampling.method if config.sampling else 'None'}"
            )
            oracles = [
                config.oracles.accuracy,
                config.oracles.explainability,
                config.oracles.confidence_calibration,
            ]
            console.print(f"🔍 Oracles: {len([o for o in oracles if o])}")
            if output:
                console.print(f"💾 Output: {output}")
        else:
            console.print("🚀 [bold green]Running evaluation...[/bold green]")
            # TODO: Implement actual evaluation logic
            console.print(
                "⚠️  [yellow]Evaluation execution not yet implemented[/yellow]"
            )

    except Exception as e:
        console.print(f"❌ [red]Error loading configuration:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
