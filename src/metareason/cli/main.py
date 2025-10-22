import asyncio
from datetime import datetime
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
@click.option("--output", "-o", help="Output file for results (JSON format)")
def run(spec, output):
    """Run an evaluation based on a specification file."""
    try:
        spec_path = Path(spec)
        console.print(
            f"[bold blue]Running evaluation with spec:[/bold blue] {spec_path}"
        )
        responses = asyncio.run(runner.run(spec_path))

        console.print(
            f"\n[bold green]✓ Completed {len(responses)} evaluations[/bold green]\n"
        )

        # Display results
        for i, response in enumerate(responses, 1):
            console.rule(f"[bold]Sample {i}/{len(responses)}[/bold]")

            # Show sample parameters
            console.print("[bold cyan]Parameters:[/bold cyan]")
            for key, value in response.sample_params.items():
                if isinstance(value, float):
                    console.print(f"  • {key}: {value:.3f}")
                else:
                    console.print(f"  • {key}: {value}")

            # Show response preview
            console.print("\n[bold cyan]Response Preview:[/bold cyan]")
            preview = (
                response.final_response[:200] + "..."
                if len(response.final_response) > 200
                else response.final_response
            )
            console.print(f"  {preview}")

            # Show evaluation scores
            console.print("\n[bold cyan]Evaluation Scores:[/bold cyan]")
            for oracle_name, eval_result in response.evaluations.items():
                console.print(
                    f"  • {oracle_name}: [bold yellow]{eval_result.score:.2f}/5.0[/bold yellow]"
                )
                if eval_result.explanation:
                    explanation_preview = (
                        eval_result.explanation[:150] + "..."
                        if len(eval_result.explanation) > 150
                        else eval_result.explanation
                    )
                    console.print(f"    → {explanation_preview}")
            console.print()

        # Save results to JSON file
        import json

        if output:
            # User specified output path
            output_path = Path(output)
        else:
            # Default: save to reports directory with timestamp
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            spec_name = Path(spec).stem
            output_path = reports_dir / f"{spec_name}_{timestamp}.json"

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        results_data = [r.model_dump() for r in responses]
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        console.print(f"\n[green]✓ Results saved to {output_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


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
