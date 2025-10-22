import asyncio
import json
from datetime import datetime
from pathlib import Path

import arviz as az
import click
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..analysis.analyzer import BayesianAnalyzer
from ..config import SpecConfig
from ..pipeline import load_spec, runner
from ..pipeline.runner import SampleResult

console = Console()


def display_bayesian_analysis(idata: az.InferenceData, oracle_name: str, results: list):
    """Display Bayesian analysis results in the console.

    Args:
        idata: ArviZ InferenceData from Bayesian analysis.
        oracle_name: Name of the oracle that was analyzed.
        results: List of SampleResult objects.
    """
    console.print(f"\n[bold magenta]Bayesian Analysis: {oracle_name}[/bold magenta]\n")

    # Get summary statistics
    summary = az.summary(idata, var_names=["true_quality", "oracle_noise"])

    # Display oracle noise estimate
    noise_mean = summary.loc["oracle_noise", "mean"]
    noise_hdi_low = summary.loc["oracle_noise", "hdi_3%"]
    noise_hdi_high = summary.loc["oracle_noise", "hdi_97%"]

    console.print(
        f"[cyan]Oracle Measurement Error:[/cyan] "
        f"{noise_mean:.3f} (95% CI: [{noise_hdi_low:.3f}, {noise_hdi_high:.3f}])\n"
    )

    # Create table for variant quality estimates
    table = Table(title="True Quality Estimates (Posterior)")
    table.add_column("Variant", justify="center", style="cyan")
    table.add_column("Observed", justify="center", style="yellow")
    table.add_column("True Quality (Mean)", justify="center", style="green")
    table.add_column("95% Credible Interval", justify="center", style="blue")

    # Add rows for each variant
    for i in range(len(results)):
        var_name = f"true_quality[{i}]"
        if var_name in summary.index:
            mean = summary.loc[var_name, "mean"]
            hdi_low = summary.loc[var_name, "hdi_3%"]
            hdi_high = summary.loc[var_name, "hdi_97%"]
            observed = results[i].evaluations[oracle_name].score

            table.add_row(
                f"#{i+1}",
                f"{observed:.2f}",
                f"{mean:.2f}",
                f"[{hdi_low:.2f}, {hdi_high:.2f}]",
            )

    console.print(table)

    # Display convergence diagnostics
    r_hat_max = summary["r_hat"].max()
    ess_bulk_min = summary["ess_bulk"].min()

    console.print("\n[cyan]Convergence Diagnostics:[/cyan]")
    console.print(f"  Max R̂: {r_hat_max:.4f} {'✓' if r_hat_max < 1.01 else '⚠️'}")
    console.print(
        f"  Min ESS (bulk): {ess_bulk_min:.0f} {'✓' if ess_bulk_min > 400 else '⚠️'}"
    )


@click.group()
def metareason():
    """MetaReason Core - Quantitative Measurement of LLM systems."""
    pass


@metareason.command()
@click.argument("spec")
@click.option("--output", "-o", help="Output file for results (JSON format)")
@click.option(
    "--analyze",
    is_flag=True,
    help="Perform Bayesian analysis on results after evaluation",
)
def run(spec, output, analyze):
    """Run an evaluation based on a specification file."""
    try:
        spec_path = Path(spec)
        console.print(
            f"[bold blue]Running evaluation with spec:[/bold blue] {spec_path}"
        )

        # Load spec to check for analysis config
        spec_config = load_spec(spec_path)

        # Run evaluation
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

        # Perform Bayesian analysis if requested
        if analyze and spec_config.analysis:
            console.print("\n[bold blue]Running Bayesian Analysis...[/bold blue]\n")

            analyzer = BayesianAnalyzer(responses, spec_config)

            # Analyze each oracle
            analysis_results = {}
            for oracle_name in spec_config.oracles.keys():
                try:
                    console.print(
                        f"[cyan]Analyzing oracle:[/cyan] {oracle_name} (MCMC sampling...)"
                    )
                    idata = analyzer.fit_calibration_model(oracle_name)
                    analysis_results[oracle_name] = idata

                    # Display results
                    display_bayesian_analysis(idata, oracle_name, responses)

                except Exception as e:
                    console.print(
                        f"[yellow]⚠️  Analysis failed for {oracle_name}: {e}[/yellow]"
                    )

        elif analyze and not spec_config.analysis:
            console.print(
                "[yellow]⚠️  --analyze flag set but no 'analysis' config in spec. "
                "Using defaults.[/yellow]\n"
            )
            # Could still run with defaults, but let's warn the user
            analyzer = BayesianAnalyzer(responses, spec_config)
            analysis_results = {}
            for oracle_name in spec_config.oracles.keys():
                try:
                    console.print(
                        f"[cyan]Analyzing oracle:[/cyan] {oracle_name} (MCMC sampling...)"
                    )
                    idata = analyzer.fit_calibration_model(oracle_name)
                    analysis_results[oracle_name] = idata
                    display_bayesian_analysis(idata, oracle_name, responses)
                except Exception as e:
                    console.print(
                        f"[yellow]⚠️  Analysis failed for {oracle_name}: {e}[/yellow]"
                    )

        # Save results to JSON file
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

        # Save analysis results if they exist
        if analyze and "analysis_results" in locals():
            for oracle_name, idata in analysis_results.items():
                # Save as NetCDF (ArviZ InferenceData format)
                analysis_path = output_path.with_suffix("").with_suffix(
                    f".{oracle_name}_analysis.nc"
                )
                idata.to_netcdf(analysis_path)
                console.print(
                    f"[green]✓ Analysis for {oracle_name} saved to {analysis_path}[/green]"
                )

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


@metareason.command()
@click.argument("results_json", type=click.Path(exists=True))
@click.option(
    "--spec",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Specification file used to generate the results",
)
@click.option(
    "--oracle",
    "-o",
    help="Specific oracle to analyze (if not specified, analyzes all)",
)
def analyze(results_json, spec, oracle):
    """Perform Bayesian analysis on previously saved evaluation results.

    This command loads evaluation results from a JSON file and runs Bayesian
    analysis to estimate true quality scores and oracle measurement error.

    Example:
        metareason analyze reports/my_eval_20250122_143022.json --spec examples/my_spec.yml
    """
    try:
        results_path = Path(results_json)
        spec_path = Path(spec)

        # Load spec
        console.print(f"[cyan]Loading spec from:[/cyan] {spec_path}")
        spec_config = load_spec(spec_path)

        # Check if spec has analysis config
        if not spec_config.analysis:
            console.print(
                "[yellow]⚠️  No 'analysis' section in spec. Using default parameters.[/yellow]\n"
            )

        # Load results from JSON
        console.print(f"[cyan]Loading results from:[/cyan] {results_path}")
        with open(results_path, "r") as f:
            results_data = json.load(f)

        # Convert JSON back to SampleResult objects
        results = [SampleResult(**r) for r in results_data]
        console.print(f"[green]✓ Loaded {len(results)} evaluation results[/green]\n")

        # Initialize analyzer
        analyzer = BayesianAnalyzer(results, spec_config)

        # Determine which oracles to analyze
        if oracle:
            oracle_names = [oracle] if oracle in spec_config.oracles else []
            if not oracle_names:
                console.print(f"[red]Error: Oracle '{oracle}' not found in spec[/red]")
                return
        else:
            oracle_names = list(spec_config.oracles.keys())

        # Analyze each oracle
        console.print("[bold blue]Running Bayesian Analysis...[/bold blue]\n")
        analysis_results = {}

        for oracle_name in oracle_names:
            try:
                console.print(
                    f"[cyan]Analyzing oracle:[/cyan] {oracle_name} (MCMC sampling...)"
                )
                idata = analyzer.fit_calibration_model(oracle_name)
                analysis_results[oracle_name] = idata

                # Display results
                display_bayesian_analysis(idata, oracle_name, results)

            except KeyError:
                console.print(
                    f"[yellow]⚠️  Oracle '{oracle_name}' not found in results[/yellow]"
                )
            except Exception as e:
                console.print(f"[red]✗ Analysis failed for {oracle_name}: {e}[/red]")

        # Save analysis results
        if analysis_results:
            console.print("\n[bold blue]Saving Analysis Results...[/bold blue]")
            for oracle_name, idata in analysis_results.items():
                # Save in same directory as results JSON
                analysis_path = results_path.with_suffix("").with_suffix(
                    f".{oracle_name}_analysis.nc"
                )
                idata.to_netcdf(analysis_path)
                console.print(
                    f"[green]✓ Analysis for {oracle_name} saved to {analysis_path}[/green]"
                )

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in results file - {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    metareason()
