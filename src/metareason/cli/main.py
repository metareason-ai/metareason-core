import asyncio
import json
from datetime import datetime
from pathlib import Path

import arviz as az
import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from ..analysis.analyzer import BayesianAnalyzer
from ..config import SpecConfig
from ..pipeline import load_spec, runner
from ..pipeline.runner import SampleResult

load_dotenv()

console = Console()


def display_population_quality(result: dict, oracle_name: str):
    """Display population-level quality estimate with HDI.

    Args:
        result: Dictionary from BayesianAnalyzer.estimate_population_quality()
        oracle_name: Name of the oracle being analyzed
    """
    hdi_prob_pct = int(result["hdi_prob"] * 100)

    console.print(f"\n[bold magenta]Population Quality: {oracle_name}[/bold magenta]\n")

    # Main finding - the HDI statement
    console.print(
        f"[bold green]We are {hdi_prob_pct}% confident the true {oracle_name} quality "
        f"is between {result['hdi_lower']:.2f} and {result['hdi_upper']:.2f}[/bold green]\n"
    )

    # Additional statistics
    console.print("[cyan]Population Statistics:[/cyan]")
    console.print(f"  Mean: {result['population_mean']:.3f}")
    console.print(f"  Median: {result['population_median']:.3f}")
    console.print(
        f"  {hdi_prob_pct}% HDI: [{result['hdi_lower']:.3f}, {result['hdi_upper']:.3f}]"
    )

    # Oracle noise estimate
    noise_hdi_lower, noise_hdi_upper = result["oracle_noise_hdi"]
    console.print(
        f"\n[cyan]Oracle Variability:[/cyan] "
        f"{result['oracle_noise_mean']:.3f} "
        f"({hdi_prob_pct}% HDI: [{noise_hdi_lower:.3f}, {noise_hdi_upper:.3f}])"
    )

    console.print(f"[dim]Based on {result['n_samples']} evaluations[/dim]")


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
@click.option(
    "--report",
    is_flag=True,
    help="Generate HTML report after analysis",
)
def run(spec, output, analyze, report):
    """Run an evaluation based on a specification file."""
    try:
        spec_path = Path(spec)

        # Load spec to check for analysis config
        spec_config = load_spec(spec_path)

        console.print(
            f"[bold blue]Running evaluation with spec:[/bold blue] {spec_path}"
        )
        console.print(f"[dim]Generating {spec_config.n_variants} variant(s)...[/dim]\n")

        # Run evaluation with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {spec_config.n_variants} variants with {len(spec_config.oracles)} oracle(s)...",
                total=None,
            )
            responses = asyncio.run(runner.run(spec_path))
            progress.update(task, completed=True)

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
            oracle_names = list(spec_config.oracles.keys())

            # Analyze each oracle with progress tracking
            analysis_results = {}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                analysis_task = progress.add_task(
                    f"[cyan]Bayesian analysis: 0/{len(oracle_names)} oracles completed",
                    total=len(oracle_names),
                )

                for idx, oracle_name in enumerate(oracle_names, 1):
                    try:
                        progress.update(
                            analysis_task,
                            description=f"[cyan]Analyzing {oracle_name} (MCMC sampling...)",
                        )

                        # Estimate population-level quality with HDI
                        hdi_prob = (
                            spec_config.analysis.hdi_probability
                            if spec_config.analysis
                            else 0.94
                        )
                        pop_result = analyzer.estimate_population_quality(
                            oracle_name, hdi_prob=hdi_prob
                        )
                        analysis_results[oracle_name] = pop_result

                        progress.update(
                            analysis_task,
                            advance=1,
                            description=f"[cyan]Bayesian analysis: {idx}/{len(oracle_names)} oracles completed",
                        )

                    except Exception as e:
                        progress.update(analysis_task, advance=1)
                        console.print(
                            f"[yellow]⚠️  Analysis failed for {oracle_name}: {e}[/yellow]"
                        )

            # Display results after progress bar completes
            console.print()
            for oracle_name, pop_result in analysis_results.items():
                display_population_quality(pop_result, oracle_name)

        elif analyze and not spec_config.analysis:
            console.print(
                "[yellow]⚠️  --analyze flag set but no 'analysis' config in spec. "
                "Using defaults.[/yellow]\n"
            )
            # Run with defaults
            analyzer = BayesianAnalyzer(responses, spec_config)
            oracle_names = list(spec_config.oracles.keys())
            analysis_results = {}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                analysis_task = progress.add_task(
                    f"[cyan]Bayesian analysis: 0/{len(oracle_names)} oracles completed",
                    total=len(oracle_names),
                )

                for idx, oracle_name in enumerate(oracle_names, 1):
                    try:
                        progress.update(
                            analysis_task,
                            description=f"[cyan]Analyzing {oracle_name} (MCMC sampling...)",
                        )

                        # Estimate population-level quality with HDI
                        pop_result = analyzer.estimate_population_quality(
                            oracle_name, hdi_prob=0.94
                        )
                        analysis_results[oracle_name] = pop_result

                        progress.update(
                            analysis_task,
                            advance=1,
                            description=f"[cyan]Bayesian analysis: {idx}/{len(oracle_names)} oracles completed",
                        )

                    except Exception as e:
                        progress.update(analysis_task, advance=1)
                        console.print(
                            f"[yellow]⚠️  Analysis failed for {oracle_name}: {e}[/yellow]"
                        )

            # Display results after progress bar completes
            console.print()
            for oracle_name, pop_result in analysis_results.items():
                display_population_quality(pop_result, oracle_name)

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
            for oracle_name, pop_result in analysis_results.items():
                # Save population quality results as JSON
                analysis_path = output_path.with_suffix("").with_suffix(
                    f".{oracle_name}_population_quality.json"
                )
                with open(analysis_path, "w") as f:
                    json.dump(pop_result, f, indent=2)
                console.print(
                    f"[green]✓ Population quality analysis for {oracle_name} saved to {analysis_path}[/green]"
                )

        # Generate HTML report if requested
        if report and analyze and "analysis_results" in locals() and analysis_results:
            from ..reporting import ReportGenerator

            generator = ReportGenerator(responses, spec_config, analysis_results)
            report_path = output_path.with_suffix(".html")
            generator.generate_html(report_path)
            console.print(f"[green]✓ HTML report saved to {report_path}[/green]")

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
@click.option(
    "--report",
    is_flag=True,
    help="Generate HTML report after analysis",
)
def analyze(results_json, spec, oracle, report):
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            analysis_task = progress.add_task(
                f"[cyan]Bayesian analysis: 0/{len(oracle_names)} oracles completed",
                total=len(oracle_names),
            )

            for idx, oracle_name in enumerate(oracle_names, 1):
                try:
                    progress.update(
                        analysis_task,
                        description=f"[cyan]Analyzing {oracle_name} (MCMC sampling...)",
                    )

                    # Estimate population-level quality with HDI
                    hdi_prob = (
                        spec_config.analysis.hdi_probability
                        if spec_config.analysis
                        else 0.94
                    )
                    pop_result = analyzer.estimate_population_quality(
                        oracle_name, hdi_prob=hdi_prob
                    )
                    analysis_results[oracle_name] = pop_result

                    progress.update(
                        analysis_task,
                        advance=1,
                        description=f"[cyan]Bayesian analysis: {idx}/{len(oracle_names)} oracles completed",
                    )

                except KeyError:
                    progress.update(analysis_task, advance=1)
                    console.print(
                        f"[yellow]⚠️  Oracle '{oracle_name}' not found in results[/yellow]"
                    )
                except Exception as e:
                    progress.update(analysis_task, advance=1)
                    console.print(
                        f"[red]✗ Analysis failed for {oracle_name}: {e}[/red]"
                    )

        # Display results after progress bar completes
        console.print()
        for oracle_name, pop_result in analysis_results.items():
            display_population_quality(pop_result, oracle_name)

        # Save analysis results
        if analysis_results:
            console.print("\n[bold blue]Saving Analysis Results...[/bold blue]")
            for oracle_name, pop_result in analysis_results.items():
                # Save population quality results as JSON
                analysis_path = results_path.with_suffix("").with_suffix(
                    f".{oracle_name}_population_quality.json"
                )
                with open(analysis_path, "w") as f:
                    json.dump(pop_result, f, indent=2)
                console.print(
                    f"[green]✓ Population quality analysis for {oracle_name} saved to {analysis_path}[/green]"
                )

        # Generate HTML report if requested
        if report and analysis_results:
            from ..reporting import ReportGenerator

            generator = ReportGenerator(results, spec_config, analysis_results)
            report_path = results_path.with_suffix(".html")
            generator.generate_html(report_path)
            console.print(f"[green]✓ HTML report saved to {report_path}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in results file - {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@metareason.command()
@click.argument("results_json", type=click.Path(exists=True))
@click.option(
    "--spec",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Specification file",
)
@click.option("--output", "-o", help="Output path for HTML report")
def report(results_json, spec, output):
    """Generate an HTML report from evaluation results."""
    try:
        results_path = Path(results_json)
        spec_path = Path(spec)

        # Load spec
        console.print(f"[cyan]Loading spec from:[/cyan] {spec_path}")
        spec_config = load_spec(spec_path)

        # Load results
        console.print(f"[cyan]Loading results from:[/cyan] {results_path}")
        with open(results_path, "r") as f:
            results_data = json.load(f)
        results = [SampleResult(**r) for r in results_data]
        console.print(f"[green]✓ Loaded {len(results)} results[/green]\n")

        # Run Bayesian analysis
        console.print("[bold blue]Running Bayesian Analysis...[/bold blue]\n")
        analyzer = BayesianAnalyzer(results, spec_config)
        hdi_prob = (
            spec_config.analysis.hdi_probability if spec_config.analysis else 0.94
        )

        analysis_results = {}
        oracle_names = list(spec_config.oracles.keys())

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing...", total=len(oracle_names))
            for oracle_name in oracle_names:
                pop_result = analyzer.estimate_population_quality(
                    oracle_name, hdi_prob=hdi_prob
                )
                analysis_results[oracle_name] = pop_result
                progress.update(task, advance=1)

        # Generate report
        from ..reporting import ReportGenerator

        generator = ReportGenerator(results, spec_config, analysis_results)

        if output:
            report_path = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            spec_name = Path(spec).stem
            report_path = Path("reports") / f"{spec_name}_{timestamp}_report.html"

        generator.generate_html(report_path)
        console.print(f"\n[green]✓ HTML report saved to {report_path}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in results file - {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


if __name__ == "__main__":
    metareason()
