"""CLI commands for template operations."""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..config import load_yaml_config
from ..sampling import create_sampler
from ..templates import (
    TemplateEngine,
    TemplateValidator,
    ValidationLevel,
    generate_prompts_from_config,
)

console = Console()


@click.group(name="template")
def template_group() -> None:
    """Template validation and rendering commands."""
    pass


@template_group.command("validate")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--level",
    type=click.Choice(["permissive", "standard", "strict"]),
    default="standard",
    help="Validation strictness level",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def validate_template(config_path: Path, level: str, output_format: str) -> None:
    """Validate template syntax and security."""
    try:
        # Load configuration
        config = load_yaml_config(config_path)

        # Create validator
        validation_level = ValidationLevel(level)
        validator = TemplateValidator(level=validation_level)

        # Validate template
        result = validator.validate(
            config.prompt_template,
            expected_variables=set(config.axes.keys()),
            max_length=10000,
        )

        if output_format == "json":
            output = {
                "is_valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "variables": list(result.variables),
                "metadata": result.metadata,
            }
            console.print_json(json.dumps(output, indent=2))
        else:
            # Text format
            if result.is_valid:
                console.print("✅ [green]Template is valid[/green]")
            else:
                console.print("❌ [red]Template validation failed[/red]")

            if result.errors:
                console.print("\n[red]Errors:[/red]")
                for error in result.errors:
                    console.print(f"  • {error}")

            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  • {warning}")

            # Show template info
            console.print(
                f"\n[dim]Template variables:[/dim] {', '.join(sorted(result.variables))}"
            )
            console.print(
                f"[dim]Length:[/dim] {result.metadata.get('length', 0)} characters"
            )
            console.print(f"[dim]Lines:[/dim] {result.metadata.get('lines', 0)}")

    except Exception as e:
        console.print(f"❌ [red]Error loading configuration:[/red] {e}")
        raise click.Abort()


@template_group.command("render")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--samples", type=int, default=10, help="Number of sample prompts to render"
)
@click.option(
    "--output", type=click.Path(path_type=Path), help="Output file for rendered prompts"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "jsonl"]),
    default="text",
    help="Output format",
)
@click.option(
    "--show-context", is_flag=True, help="Show variable context for each prompt"
)
@click.option("--validate-outputs", is_flag=True, help="Validate rendered outputs")
def render_samples(
    config_path: Path,
    samples: int,
    output: Optional[Path],
    output_format: str,
    show_context: bool,
    validate_outputs: bool,
) -> None:
    """Render sample prompts from configuration."""
    try:
        # Load configuration
        config = load_yaml_config(config_path)

        # Override n_variants for sampling
        config.n_variants = samples

        # Create sampler
        sampler = create_sampler(config.axes, config.sampling, samples)

        console.print(f"🎲 Generating {samples} sample prompts...")

        # Generate prompts
        with console.status("Rendering templates..."):
            result = generate_prompts_from_config(
                config, sampler, validate_outputs=validate_outputs
            )

        if not result.is_successful:
            console.print("❌ [red]Prompt generation failed[/red]")
            if result.validation_result.errors:
                for error in result.validation_result.errors:
                    console.print(f"  • {error}")
            return

        console.print(
            f"✅ [green]Generated {result.render_result.success_count} prompts[/green]"
        )

        # Show statistics
        stats_table = Table(title="Generation Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Prompts", str(len(result.prompts)))
        stats_table.add_row("Success Rate", f"{result.render_result.success_rate:.1f}%")
        stats_table.add_row("Render Time", f"{result.render_result.render_time:.3f}s")
        stats_table.add_row(
            "Avg Time/Prompt",
            f"{result.render_result.render_time / samples * 1000:.1f}ms",
        )

        console.print(stats_table)

        # Handle output
        if output_format == "json":
            output_data = {
                "config_path": str(config_path),
                "prompts": [
                    {"prompt": prompt, "context": context if show_context else None}
                    for prompt, context in zip(result.prompts, result.contexts)
                ],
                "statistics": {
                    "total": len(result.prompts),
                    "success_rate": result.render_result.success_rate,
                    "render_time": result.render_result.render_time,
                },
            }

            if output:
                with open(output, "w") as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"📄 Output saved to {output}")
            else:
                console.print_json(json.dumps(output_data, indent=2))

        elif output_format == "jsonl":
            lines = []
            for i, (prompt, context) in enumerate(zip(result.prompts, result.contexts)):
                line = {"id": i, "prompt": prompt}
                if show_context:
                    line["context"] = context
                lines.append(json.dumps(line))

            if output:
                with open(output, "w") as f:
                    f.write("\n".join(lines))
                console.print(f"📄 Output saved to {output}")
            else:
                console.print("\n".join(lines))

        else:
            # Text format
            if output:
                with open(output, "w") as f:
                    for i, (prompt, context) in enumerate(
                        zip(result.prompts[:10], result.contexts[:10])
                    ):
                        f.write(f"=== Prompt {i + 1} ===\n")
                        if show_context:
                            f.write(f"Context: {context}\n\n")
                        f.write(f"{prompt}\n\n")
                console.print(f"📄 Output saved to {output}")
            else:
                # Show first few samples
                for i, (prompt, context) in enumerate(
                    zip(result.prompts[:5], result.contexts[:5])
                ):
                    panel_title = f"Sample Prompt {i + 1}"
                    if show_context:
                        panel_title += f" | Context: {context}"

                    console.print(
                        Panel(
                            Syntax(prompt, "text", theme="monokai"),
                            title=panel_title,
                            expand=False,
                        )
                    )

                if samples > 5:
                    console.print(f"\n[dim]... and {samples - 5} more prompts[/dim]")

        # Show any errors
        if result.render_result.errors:
            console.print("\n[yellow]Rendering errors:[/yellow]")
            for idx, error in result.render_result.errors[:5]:
                console.print(f"  • Sample {idx + 1}: {error}")

    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        raise click.Abort()


@template_group.command("test")
@click.argument("template_string", type=str)
@click.option("--context", type=str, help="JSON string with template context variables")
@click.option(
    "--variables", type=str, help="Comma-separated list of expected variable names"
)
def test_template(
    template_string: str, context: Optional[str], variables: Optional[str]
) -> None:
    """Test a template string directly."""
    try:
        # Parse context
        if context:
            import json

            ctx = json.loads(context)
        else:
            ctx = {}

        # Parse expected variables
        expected_vars = set()
        if variables:
            expected_vars = set(v.strip() for v in variables.split(","))

        # Validate template
        validator = TemplateValidator()
        validation_result = validator.validate(
            template_string, expected_variables=expected_vars if expected_vars else None
        )

        console.print("🧪 [bold]Template Test Results[/bold]")
        console.print()

        # Show template
        console.print(
            Panel(
                Syntax(template_string, "jinja2", theme="monokai"),
                title="Template",
                expand=False,
            )
        )

        # Show validation
        if validation_result.is_valid:
            console.print("✅ [green]Template is valid[/green]")
        else:
            console.print("❌ [red]Template validation failed[/red]")
            for error in validation_result.errors:
                console.print(f"  • {error}")

        if validation_result.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in validation_result.warnings:
                console.print(f"  • {warning}")

        # Try to render if context provided
        if ctx and validation_result.is_valid:
            try:
                engine = TemplateEngine()
                rendered = engine.render(template_string, ctx)
                console.print()
                console.print(
                    Panel(
                        Syntax(rendered, "text", theme="monokai"),
                        title="Rendered Output",
                        expand=False,
                    )
                )
            except Exception as e:
                console.print(f"\n❌ [red]Rendering failed:[/red] {e}")

        # Show variables
        if validation_result.variables:
            console.print(
                f"\n[dim]Variables found:[/dim] {', '.join(sorted(validation_result.variables))}"
            )

    except Exception as e:
        console.print(f"❌ [red]Error:[/red] {e}")
        raise click.Abort()


@template_group.command("filters")
def list_filters() -> None:
    """List available custom template filters."""
    console.print("🔧 [bold]Available Custom Filters[/bold]")
    console.print()

    filters_table = Table()
    filters_table.add_column("Filter", style="cyan")
    filters_table.add_column("Description", style="white")
    filters_table.add_column("Example", style="green")

    filters_info = [
        (
            "format_continuous",
            "Format numerical values",
            "{{ 0.123 | format_continuous(2, 'percent') }} → 12.30%",
        ),
        ("fmt_num", "Alias for format_continuous", "{{ 3.14159 | fmt_num(2) }} → 3.14"),
        (
            "format_list",
            "Format lists with conjunctions",
            "{{ ['A', 'B', 'C'] | format_list }} → A, B, and C",
        ),
        (
            "fmt_list",
            "Alias for format_list",
            "{{ items | fmt_list('; ', 'or') }} → A; B; or C",
        ),
        (
            "conditional_text",
            "Conditional text inclusion",
            "{{ value | conditional_text('high', 'low') }}",
        ),
        ("if_text", "Alias for conditional_text", "{{ flag | if_text('enabled') }}"),
        (
            "round_to_precision",
            "Round to decimal places",
            "{{ 3.14159 | round_to_precision(2) }} → 3.14",
        ),
        (
            "capitalize_first",
            "Capitalize first letter only",
            "{{ 'hello' | capitalize_first }} → Hello",
        ),
        (
            "pluralize",
            "Pluralize based on count",
            "{{ count | pluralize('item', 'items') }}",
        ),
        ("truncate", "Truncate text to length", "{{ text | truncate(50, '...') }}"),
    ]

    for name, desc, example in filters_info:
        filters_table.add_row(name, desc, example)

    console.print(filters_table)

    console.print("\n[dim]Global Functions:[/dim]")
    console.print(
        "  • [cyan]include_if(condition, text)[/cyan] - Include text only if condition is truthy"
    )
    console.print(
        "  • [cyan]select(condition, true_val, false_val)[/cyan] - Select value based on condition"
    )
    console.print(
        "  • [cyan]range(start, stop, step)[/cyan] - Safe range function (max 1000 items)"
    )

    console.print("\n[dim]Example template:[/dim]")
    example_template = """
You are a {{ role }} assistant.
Temperature: {{ temp | format_continuous(1) }}
Skills: {{ skills | format_list }}
{{ include_if(premium, "Premium features enabled.") }}
""".strip()

    console.print(
        Panel(
            Syntax(example_template, "jinja2", theme="monokai"),
            title="Example Template",
            expand=False,
        )
    )
