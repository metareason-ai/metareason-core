import asyncio
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.syntax import Syntax

from ..config import SpecConfig
from ..pipeline import load_spec, runner

console = Console()


@click.group()
def metareason():
    """MetaReason Core - Quantitative Measurement of LLM systems."""
    pass


@metareason.command()
@click.argument("spec")
def run(spec):
    """Run an evaluation based on a specification file."""
    try:
        spec_path = Path(spec)
        console.log(f"Running evaluation with spec: {spec_path}")
        responses = asyncio.run(runner.run(spec_path))

        for response in responses:
            console.log(response)

    except Exception as e:
        console.log(e)


@metareason.command()
@click.argument("spec")
def validate(spec):
    """Validate a specification file."""
    try:
        spec_path = Path(spec)
        spec_config: SpecConfig = load_spec(spec_path)
        spec_content = yaml.dump(spec_config.model_dump(), default_flow_style=False)
        syntax = Syntax(spec_content, "yaml", theme="monokai", line_numbers=True)
        console.print(f"[green] {spec} is valid![/green]")
        console.print(syntax)
    except Exception as e:
        console.log(f"Failed to load spec {spec}: {e}")


if __name__ == "__main__":
    metareason()
