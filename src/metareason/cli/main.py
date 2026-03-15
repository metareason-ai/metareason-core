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

from ..analysis.agreement import compute_agreement_summary, extract_scores_by_oracle
from ..analysis.analyzer import BayesianAnalyzer
from ..config import SpecConfig
from ..oracles.llm_judge import LLMJudge
from ..oracles.oracle_base import EvaluationContext
from ..pipeline import load_spec, runner
from ..pipeline.loader import load_calibrate_multi_spec, load_calibrate_spec
from ..pipeline.runner import SampleResult

load_dotenv()

console = Console()


def _run_bayesian_analysis(spec_config, results, con, oracle_names=None):
    """Run Bayesian analysis for all oracles with progress display.

    Args:
        spec_config: The SpecConfig or object with .analysis and .oracles attributes.
        results: List of SampleResult objects.
        con: Rich Console instance for output.
        oracle_names: Optional list of oracle names to analyze.
            If None, analyzes all oracles from spec_config.oracles.

    Returns:
        Dictionary mapping oracle_name -> population quality result dict.
    """
    analyzer = BayesianAnalyzer(results, spec_config)

    if oracle_names is None:
        oracle_names = list(spec_config.oracles.keys())

    hdi_prob = spec_config.analysis.hdi_probability if spec_config.analysis else 0.94

    analysis_results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=con,
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
                con.print(
                    f"[yellow]\u26a0\ufe0f  Analysis failed for {oracle_name}: {e}[/yellow]"
                )

    # Run parameter effects analysis if axes exist
    if hasattr(spec_config, "axes") and spec_config.axes:
        for oracle_name in oracle_names:
            if oracle_name in analysis_results:
                try:
                    effects_result = analyzer.estimate_parameter_effects(
                        oracle_name, spec_config.axes, hdi_prob=hdi_prob
                    )
                    analysis_results[oracle_name]["parameter_effects"] = effects_result
                except Exception as e:
                    con.print(
                        f"[yellow]Parameter effects analysis failed for "
                        f"{oracle_name}: {e}[/yellow]"
                    )

    return analysis_results


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


def display_parameter_effects(effects_result: dict, oracle_name: str):
    """Display parameter effects analysis results."""
    hdi_pct = int(effects_result["hdi_prob"] * 100)

    console.print(f"\n[bold magenta]Parameter Effects: {oracle_name}[/bold magenta]\n")

    table = Table(title="What Matters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Effect", justify="right")
    table.add_column(f"{hdi_pct}% HDI", justify="center")
    table.add_column("P(Direction)", justify="center")

    for e in effects_result["effects"]:
        name = e["parameter"]
        if e["level"]:
            name += f": {e['level']}"

        # Color by direction certainty
        hdi_crosses_zero = e["hdi_lower"] <= 0 <= e["hdi_upper"]
        if hdi_crosses_zero:
            effect_str = f"[dim]{e['effect_mean']:+.3f}[/dim]"
            direction = "[dim]inconclusive[/dim]"
        elif e["effect_mean"] > 0:
            prob = e["prob_positive"]
            effect_str = f"[green]{e['effect_mean']:+.3f}[/green]"
            direction = f"[green]{prob:.0%} positive[/green]"
        else:
            prob = e["prob_negative"]
            effect_str = f"[red]{e['effect_mean']:+.3f}[/red]"
            direction = f"[red]{prob:.0%} negative[/red]"

        hdi_str = f"[{e['hdi_lower']:+.3f}, {e['hdi_upper']:+.3f}]"
        table.add_row(name, effect_str, hdi_str, direction)

    console.print(table)
    console.print(
        f"[dim]Based on {effects_result['n_samples']} samples, "
        f"{effects_result['n_predictors']} predictor(s)[/dim]"
    )


def display_bayesian_analysis(
    idata: az.InferenceData, oracle_name: str, results: list, hdi_prob: float = 0.94
):
    """Display Bayesian analysis results in the console.

    Args:
        idata: ArviZ InferenceData from Bayesian analysis.
        oracle_name: Name of the oracle that was analyzed.
        results: List of SampleResult objects.
        hdi_prob: HDI probability used for the analysis (default 0.94).
    """
    console.print(f"\n[bold magenta]Bayesian Analysis: {oracle_name}[/bold magenta]\n")

    # Compute HDI column names dynamically from the probability
    alpha = (1 - hdi_prob) / 2
    hdi_low_col = f"hdi_{alpha*100:.0f}%"
    hdi_high_col = f"hdi_{(1-alpha)*100:.0f}%"
    hdi_prob_pct = int(hdi_prob * 100)

    # Get summary statistics
    summary = az.summary(
        idata, var_names=["true_quality", "oracle_noise"], hdi_prob=hdi_prob
    )

    # Display oracle noise estimate
    noise_mean = summary.loc["oracle_noise", "mean"]
    noise_hdi_low = summary.loc["oracle_noise", hdi_low_col]
    noise_hdi_high = summary.loc["oracle_noise", hdi_high_col]

    console.print(
        f"[cyan]Oracle Measurement Error:[/cyan] "
        f"{noise_mean:.3f} ({hdi_prob_pct}% CI: [{noise_hdi_low:.3f}, {noise_hdi_high:.3f}])\n"
    )

    # Create table for variant quality estimates
    table = Table(title="True Quality Estimates (Posterior)")
    table.add_column("Variant", justify="center", style="cyan")
    table.add_column("Observed", justify="center", style="yellow")
    table.add_column("True Quality (Mean)", justify="center", style="green")
    table.add_column(
        f"{hdi_prob_pct}% Credible Interval", justify="center", style="blue"
    )

    # Add rows for each variant
    for i in range(len(results)):
        var_name = f"true_quality[{i}]"
        if var_name in summary.index:
            mean = summary.loc[var_name, "mean"]
            hdi_low = summary.loc[var_name, hdi_low_col]
            hdi_high = summary.loc[var_name, hdi_high_col]
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
@click.option(
    "--db",
    type=click.Path(),
    default=None,
    help="SQLite database path to store run data (e.g. runs.db)",
)
def run(spec, output, analyze, report, db):
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

        # Initialize DB storage if requested
        store = None
        run_id = None
        if db:
            from ..storage import RunStore

            store = RunStore(db)
            run_id = store.start_run(
                spec_id=spec_config.spec_id,
                n_variants=len(responses),
                n_oracles=len(spec_config.oracles),
            )
            # Save pipeline stage configs
            stages = [
                {
                    "model": stage.model,
                    "adapter": stage.adapter.name,
                    "temperature": stage.temperature,
                    "top_p": stage.top_p,
                    "max_tokens": stage.max_tokens,
                }
                for stage in spec_config.pipeline
            ]
            store.save_pipeline_stages(run_id, stages)

            # Save every sample + evaluations
            for i, response in enumerate(responses):
                evals = {
                    name: {
                        "score": ev.score,
                        "explanation": ev.explanation,
                    }
                    for name, ev in response.evaluations.items()
                }
                store.save_sample(
                    run_id=run_id,
                    sample_index=i,
                    sample_params=response.sample_params,
                    original_prompt=response.original_prompt,
                    final_response=response.final_response,
                    evaluations=evals,
                )
            console.print(f"[green]✓ Saved {len(responses)} samples to {db}[/green]")

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
        if analyze:
            if not spec_config.analysis:
                console.print(
                    "[yellow]\u26a0\ufe0f  --analyze flag set but no 'analysis' config in spec. "
                    "Using defaults.[/yellow]\n"
                )

            console.print("\n[bold blue]Running Bayesian Analysis...[/bold blue]\n")
            analysis_results = _run_bayesian_analysis(spec_config, responses, console)

            # Display results after progress bar completes
            console.print()
            for oracle_name, pop_result in analysis_results.items():
                display_population_quality(pop_result, oracle_name)
                if "parameter_effects" in pop_result:
                    display_parameter_effects(
                        pop_result["parameter_effects"], oracle_name
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

        # Save analysis to DB and finalize run
        if store and run_id is not None:
            if analyze and "analysis_results" in locals() and analysis_results:
                for oracle_name, pop_result in analysis_results.items():
                    store.save_analysis(run_id, oracle_name, pop_result)
                console.print(
                    f"[green]✓ Analysis results saved to {db}[/green]"
                )
            store.finish_run(run_id)
            store.close()

    except Exception as e:
        # Mark run as failed if DB is active
        if "store" in locals() and store and "run_id" in locals() and run_id:
            store.finish_run(run_id, status="failed")
            store.close()
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
@click.option(
    "--agreement",
    is_flag=True,
    help="Compute inter-judge agreement metrics (requires multiple oracles)",
)
def analyze(results_json, spec, oracle, report, agreement):
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

        # Determine which oracles to analyze
        if oracle:
            oracle_names = [oracle] if oracle in spec_config.oracles else []
            if not oracle_names:
                console.print(f"[red]Error: Oracle '{oracle}' not found in spec[/red]")
                return
        else:
            oracle_names = None

        # Analyze each oracle
        console.print("[bold blue]Running Bayesian Analysis...[/bold blue]\n")
        analysis_results = _run_bayesian_analysis(
            spec_config, results, console, oracle_names=oracle_names
        )

        # Display results after progress bar completes
        console.print()
        for oracle_name, pop_result in analysis_results.items():
            display_population_quality(pop_result, oracle_name)
            if "parameter_effects" in pop_result:
                display_parameter_effects(pop_result["parameter_effects"], oracle_name)

        # Compute agreement metrics if requested
        agreement_result = None
        if agreement:
            scores_by_oracle = extract_scores_by_oracle(results)
            if len(scores_by_oracle) >= 2:
                agreement_result = compute_agreement_summary(scores_by_oracle)
                console.print("\n[bold magenta]Inter-Judge Agreement[/bold magenta]")
                console.print(
                    f"  Krippendorff's alpha: "
                    f"{agreement_result['krippendorff_alpha']:.3f}"
                )
                if agreement_result["mean_pearson"] is not None:
                    console.print(
                        f"  Mean Pearson: {agreement_result['mean_pearson']:.3f}"
                    )
            else:
                console.print(
                    "[yellow]--agreement requires multiple oracles in results[/yellow]"
                )

        # Save analysis results
        if analysis_results:
            console.print("\n[bold blue]Saving Analysis Results...[/bold blue]")
            for oracle_name, pop_result in analysis_results.items():
                save_data = dict(pop_result)
                if agreement_result:
                    save_data["agreement"] = agreement_result
                # Save population quality results as JSON
                analysis_path = results_path.with_suffix("").with_suffix(
                    f".{oracle_name}_population_quality.json"
                )
                with open(analysis_path, "w") as f:
                    json.dump(save_data, f, indent=2)
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
        analysis_results = _run_bayesian_analysis(spec_config, results, console)

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


def display_calibration_results(result: dict, oracle_name: str, calibrate_config):
    """Display judge calibration results focused on bias and consistency."""
    hdi_prob_pct = int(result["hdi_prob"] * 100)

    console.print(f"\n[bold magenta]Judge Calibration: {oracle_name}[/bold magenta]\n")

    if "bias_mean" in result:
        # With expected_score: report bias
        expected = result["expected_score"]
        bias_lo, bias_hi = result["bias_hdi"]
        direction = "higher" if result["bias_mean"] > 0 else "lower"
        console.print(
            f"[bold green]This judge scores {result['bias_mean']:+.2f} vs ground truth "
            f"({hdi_prob_pct}% CI: [{bias_lo:+.2f}, {bias_hi:+.2f}])[/bold green]\n"
        )

        console.print(
            f"[cyan]Accuracy (bias) vs expected score of {expected:.1f}:[/cyan]"
        )
        console.print(
            f"  Error: {result['bias_mean']:+.2f} "
            f"(judge scores {direction} than ground truth)"
        )
        console.print(f"  {hdi_prob_pct}% CI: [{bias_lo:+.2f}, {bias_hi:+.2f}]")
    else:
        # Without expected_score: report estimated quality (secondary)
        eq_lo, eq_hi = result["estimated_quality_hdi"]
        console.print(
            f"[bold green]Estimated quality: {result['estimated_quality_mean']:.2f} "
            f"({hdi_prob_pct}% CI: [{eq_lo:.2f}, {eq_hi:.2f}])[/bold green]\n"
        )
        console.print("[dim]Bias cannot be estimated without an expected score.[/dim]")

    noise_lo, noise_hi = result["noise_hdi"]
    console.print("\n[cyan]Consistency (noise):[/cyan]")
    console.print(
        f"  Inconsistency: ±{result['noise_mean']:.2f} "
        f"({hdi_prob_pct}% CI: [{noise_lo:.2f}, {noise_hi:.2f}])"
    )
    console.print(f"  Based on {result['n_samples']} repeated evaluations")

    console.print("\n[cyan]Raw Scores:[/cyan]")
    console.print(
        f"  Mean: {result['raw_score_mean']:.2f}  "
        f"Std: {result['raw_score_std']:.2f}"
    )

    # Verdict
    console.print("\n[cyan]Verdict:[/cyan]")
    noise_level = result["noise_mean"]
    if "bias_mean" in result:
        abs_bias = abs(result["bias_mean"])
        if abs_bias < 0.2 and noise_level < 0.2:
            console.print("  [green]Well-calibrated. Accurate and consistent.[/green]")
        elif abs_bias >= 0.2 and noise_level < 0.2:
            console.print(
                f"  [yellow]Consistently {direction}. "
                f"Usable if you adjust by ~{result['bias_mean']:+.2f}.[/yellow]"
            )
        elif abs_bias < 0.2 and noise_level >= 0.2:
            console.print(
                "  [yellow]Accurate on average but inconsistent. "
                "Consider more repeats or lower temperature.[/yellow]"
            )
        else:
            console.print(
                f"  [red]Both biased ({result['bias_mean']:+.2f}) and noisy "
                f"(±{noise_level:.2f}). Consider a different judge.[/red]"
            )
    else:
        if noise_level < 0.2:
            console.print(
                "  [green]Consistent judge. "
                "Add expected_score to measure accuracy.[/green]"
            )
        else:
            console.print(
                f"  [yellow]Noisy judge (±{noise_level:.2f}). "
                "Consider lower temperature or a different model.[/yellow]"
            )


@metareason.command()
@click.argument("spec")
@click.option("--output", "-o", help="Output file for results (JSON format)")
@click.option(
    "--report",
    is_flag=True,
    help="Generate HTML report",
)
def calibrate(spec, output, report):
    """Calibrate an LLM judge by measuring scoring consistency and reliability."""
    try:
        spec_path = Path(spec)
        calibrate_config = load_calibrate_spec(spec_path)

        console.print(
            f"[bold blue]Calibrating judge with spec:[/bold blue] {spec_path}"
        )
        console.print(
            f"[dim]Running {calibrate_config.repeats} repeated evaluations...[/dim]\n"
        )

        # Initialize the judge oracle
        judge = LLMJudge(calibrate_config.oracle)

        # Create evaluation context from fixed prompt+response
        eval_context = EvaluationContext(
            prompt=calibrate_config.prompt,
            response=calibrate_config.response,
        )

        # Run repeated evaluations
        failures = 0

        async def run_evaluations():
            nonlocal failures
            results = []
            for _ in range(calibrate_config.repeats):
                try:
                    result = await judge.evaluate(eval_context)
                    results.append(result)
                except Exception as eval_err:
                    failures += 1
                    console.print(
                        f"[yellow]Evaluation {len(results) + failures} failed: "
                        f"{eval_err}[/yellow]"
                    )
            return results

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating ({calibrate_config.repeats} repeats)...",
                total=None,
            )
            eval_results = asyncio.run(run_evaluations())
            progress.update(task, completed=True)

        if not eval_results:
            console.print(
                "[bold red]Error: All evaluations failed. "
                "Cannot perform analysis.[/bold red]"
            )
            return

        succeeded = len(eval_results)
        total = succeeded + failures
        if failures > 0:
            console.print(
                f"\n[bold yellow]Completed {succeeded}/{total} evaluations "
                f"({failures} failed)[/bold yellow]\n"
            )
        else:
            console.print(
                f"\n[bold green]Completed {succeeded} evaluations[/bold green]\n"
            )

        # Display raw scores
        scores = [r.score for r in eval_results]
        console.print(f"[dim]Raw scores: {', '.join(f'{s:.1f}' for s in scores)}[/dim]")

        # Create synthetic SampleResult objects for the analyzer
        oracle_name = calibrate_config.oracle.model
        sample_results = [
            SampleResult(
                sample_params={},
                original_prompt=calibrate_config.prompt,
                final_response=calibrate_config.response,
                evaluations={oracle_name: eval_result},
            )
            for eval_result in eval_results
        ]

        # Run Bayesian analysis
        console.print("\n[bold blue]Running Bayesian Analysis...[/bold blue]\n")

        analysis_config = calibrate_config.analysis
        analyzer = BayesianAnalyzer(sample_results, analysis_config=analysis_config)

        hdi_prob = analysis_config.hdi_probability if analysis_config else 0.94

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            analysis_task = progress.add_task("[cyan]MCMC sampling...", total=None)
            cal_result = analyzer.estimate_judge_calibration(
                oracle_name,
                expected_score=calibrate_config.expected_score,
                hdi_prob=hdi_prob,
            )
            progress.update(analysis_task, completed=True)

        # Display calibration results
        display_calibration_results(cal_result, oracle_name, calibrate_config)

        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results_data = {
                "spec_id": calibrate_config.spec_id,
                "oracle": oracle_name,
                "repeats": calibrate_config.repeats,
                "scores": scores,
                "analysis": cal_result,
            }
            if calibrate_config.expected_score is not None:
                results_data["expected_score"] = calibrate_config.expected_score

            with open(output_path, "w") as f:
                json.dump(results_data, f, indent=2)
            console.print(f"\n[green]Results saved to {output_path}[/green]")

        # Generate HTML report if requested
        if report:
            from ..reporting import CalibrationReportGenerator

            generator = CalibrationReportGenerator(calibrate_config, scores, cal_result)

            if output:
                report_path = Path(output).with_suffix(".html")
            else:
                reports_dir = Path("reports")
                reports_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spec_name = Path(spec).stem
                report_path = reports_dir / f"{spec_name}_{timestamp}_report.html"

            generator.generate_html(report_path)
            console.print(f"[green]✓ HTML report saved to {report_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


def display_multi_judge_results(multi_result, calibrate_config):
    """Display multi-judge calibration results focused on per-judge assessment."""
    hdi_prob_pct = int(multi_result["hdi_prob"] * 100)

    console.print("\n[bold magenta]Multi-Judge Calibration Results[/bold magenta]\n")

    # Per-judge verdicts first
    has_expected = "expected_score" in multi_result
    if has_expected:
        expected = multi_result["expected_score"]
        console.print(
            f"[cyan]Per-Judge Assessment (vs expected score {expected:.1f}):[/cyan]\n"
        )
    else:
        console.print("[cyan]Per-Judge Assessment:[/cyan]\n")

    for name, info in multi_result["judges"].items():
        bias_lo, bias_hi = info["bias_hdi"]
        noise_lo, noise_hi = info["noise_hdi"]
        abs_bias = abs(info["bias_mean"])
        noise = info["noise_mean"]

        if has_expected:
            direction = "higher" if info["bias_mean"] > 0 else "lower"
            console.print(
                f"  [bold]{name}[/bold]: "
                f"Accuracy (bias) {info['bias_mean']:+.2f} "
                f"({hdi_prob_pct}% CI: [{bias_lo:+.2f}, {bias_hi:+.2f}]), "
                f"Consistency (noise) ±{noise:.2f} "
                f"({hdi_prob_pct}% CI: [{noise_lo:.2f}, {noise_hi:.2f}])"
            )
            # Verdict
            if abs_bias < 0.2 and noise < 0.2:
                console.print("    → [green]Well-calibrated[/green]")
            elif abs_bias >= 0.2 and noise < 0.2:
                console.print(
                    f"    → [yellow]Consistently {direction} by "
                    f"~{info['bias_mean']:+.2f}[/yellow]"
                )
            elif abs_bias < 0.2 and noise >= 0.2:
                console.print(
                    "    → [yellow]Accurate on average but inconsistent[/yellow]"
                )
            else:
                console.print(
                    f"    → [red]Biased ({info['bias_mean']:+.2f}) and noisy "
                    f"(±{noise:.2f})[/red]"
                )
        else:
            console.print(
                f"  [bold]{name}[/bold]: "
                f"Consistency (noise) ±{noise:.2f} "
                f"({hdi_prob_pct}% CI: [{noise_lo:.2f}, {noise_hi:.2f}]), "
                f"Relative accuracy (bias) {info['bias_mean']:+.2f}"
            )

    # Judge comparison table
    table = Table(title="\nJudge Comparison")
    table.add_column("Judge", style="cyan")
    table.add_column("Accuracy (bias)", justify="right")
    table.add_column("Consistency (noise)", justify="right")
    table.add_column("Consistency Weight", justify="right")
    table.add_column("Raw Mean", justify="right")
    table.add_column("N", justify="right")

    for name, info in multi_result["judges"].items():
        table.add_row(
            name,
            f"{info['bias_mean']:+.3f}",
            f"{info['noise_mean']:.3f}",
            f"{info['consistency_weight']:.1%}",
            f"{info['raw_score_mean']:.2f}",
            str(info["n_evaluations"]),
        )

    console.print(table)

    # Show consensus quality only when no expected_score (secondary info)
    if not has_expected and "true_quality_mean" in multi_result:
        console.print("\n[cyan]Estimated Quality (secondary):[/cyan]")
        console.print(
            f"  Mean: {multi_result['true_quality_mean']:.3f}  "
            f"{hdi_prob_pct}% HDI: [{multi_result['hdi_lower']:.2f}, "
            f"{multi_result['hdi_upper']:.2f}]"
        )

    console.print(
        f"\n[dim]Based on {multi_result['n_judges']} judges, "
        f"{multi_result['n_total_evaluations']} total evaluations[/dim]"
    )


@metareason.command(name="calibrate-multi")
@click.argument("spec")
@click.option("--output", "-o", help="Output file for results (JSON format)")
@click.option("--report", is_flag=True, help="Generate HTML report")
def calibrate_multi(spec, output, report):
    """Calibrate multiple LLM judges and measure inter-judge agreement."""
    try:
        spec_path = Path(spec)
        calibrate_config = load_calibrate_multi_spec(spec_path)

        console.print(
            f"[bold blue]Multi-judge calibration with spec:[/bold blue] {spec_path}"
        )
        console.print(
            f"[dim]Running {calibrate_config.repeats} repeated evaluations "
            f"across {len(calibrate_config.oracles)} judges...[/dim]\n"
        )

        # Initialize judges
        judges = {}
        for name, oracle_config in calibrate_config.oracles.items():
            judges[name] = LLMJudge(oracle_config)

        # Create evaluation context
        eval_context = EvaluationContext(
            prompt=calibrate_config.prompt,
            response=calibrate_config.response,
        )

        # Run evaluations for each judge
        per_judge_scores = {name: [] for name in judges}
        per_judge_failures = {name: 0 for name in judges}

        async def run_all_evaluations():
            for name, judge in judges.items():
                for _ in range(calibrate_config.repeats):
                    try:
                        result = await judge.evaluate(eval_context)
                        per_judge_scores[name].append(result)
                    except Exception as eval_err:
                        per_judge_failures[name] += 1
                        console.print(
                            f"[yellow]{name} evaluation failed: {eval_err}[/yellow]"
                        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating ({len(judges)} judges x "
                f"{calibrate_config.repeats} repeats)...",
                total=None,
            )
            asyncio.run(run_all_evaluations())
            progress.update(task, completed=True)

        # Check if any judges succeeded
        active_judges = {
            name: scores for name, scores in per_judge_scores.items() if scores
        }
        if len(active_judges) < 2:
            console.print(
                "[bold red]Error: Fewer than 2 judges produced results. "
                "Cannot perform multi-judge analysis.[/bold red]"
            )
            return

        # Display per-judge raw scores
        for name, eval_results in active_judges.items():
            scores = [r.score for r in eval_results]
            failures = per_judge_failures[name]
            total = len(scores) + failures
            status = f"({len(scores)}/{total})" if failures else f"({len(scores)})"
            console.print(
                f"[dim]{name} {status}: "
                f"{', '.join(f'{s:.1f}' for s in scores)}[/dim]"
            )

        # Build SampleResult objects — one per repeat
        oracle_names = list(active_judges.keys())
        max_evals = max(len(v) for v in active_judges.values())
        sample_results = []

        for i in range(max_evals):
            evaluations = {}
            for name in oracle_names:
                if i < len(active_judges[name]):
                    evaluations[name] = active_judges[name][i]
            if evaluations:
                sample_results.append(
                    SampleResult(
                        sample_params={},
                        original_prompt=calibrate_config.prompt,
                        final_response=calibrate_config.response,
                        evaluations=evaluations,
                    )
                )

        # Run hierarchical Bayesian analysis
        console.print("\n[bold blue]Running Bayesian Analysis...[/bold blue]\n")

        analysis_config = calibrate_config.analysis
        analyzer = BayesianAnalyzer(sample_results, analysis_config=analysis_config)

        hdi_prob = analysis_config.hdi_probability if analysis_config else 0.94

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            analysis_task = progress.add_task(
                "[cyan]Hierarchical MCMC sampling...", total=None
            )
            multi_result = analyzer.estimate_multi_judge_quality(
                oracle_names,
                hdi_prob=hdi_prob,
                expected_score=calibrate_config.expected_score,
            )
            progress.update(analysis_task, completed=True)

        # Build scores dict for report
        scores_by_oracle = {}
        for name in oracle_names:
            scores_by_oracle[name] = [r.score for r in active_judges[name]]

        # Display results (no agreement metrics — those belong in `metareason run`)
        display_multi_judge_results(multi_result, calibrate_config)

        # Save JSON if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results_data = {
                "spec_id": calibrate_config.spec_id,
                "oracles": oracle_names,
                "repeats": calibrate_config.repeats,
                "multi_judge_analysis": multi_result,
                "per_judge_scores": {
                    name: [r.score for r in results]
                    for name, results in active_judges.items()
                },
            }
            if calibrate_config.expected_score is not None:
                results_data["expected_score"] = calibrate_config.expected_score

            with open(output_path, "w") as f:
                json.dump(results_data, f, indent=2)
            console.print(f"\n[green]Results saved to {output_path}[/green]")

        # Generate HTML report if requested
        if report:
            from ..reporting import MultiJudgeReportGenerator

            generator = MultiJudgeReportGenerator(
                calibrate_config,
                scores_by_oracle,
                multi_result,
            )

            if output:
                report_path = Path(output).with_suffix(".html")
            else:
                reports_dir = Path("reports")
                reports_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spec_name = Path(spec).stem
                report_path = reports_dir / f"{spec_name}_{timestamp}_report.html"

            generator.generate_html(report_path)
            console.print(f"[green]✓ HTML report saved to {report_path}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


@metareason.group()
@click.argument("db_path", type=click.Path(exists=True))
@click.pass_context
def db(ctx, db_path):
    """Query and export data from a run database."""
    from ..storage import RunStore

    ctx.ensure_object(dict)
    ctx.obj["store"] = RunStore(db_path)


@db.command(name="runs")
@click.option("--limit", "-n", default=20, help="Number of runs to show")
@click.pass_context
def db_runs(ctx, limit):
    """List recent runs."""
    store = ctx.obj["store"]
    runs = store.list_runs(limit=limit)
    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Recent Runs")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Spec ID")
    table.add_column("Started")
    table.add_column("Status")
    table.add_column("Variants", justify="right")
    table.add_column("Oracles", justify="right")

    for r in runs:
        status_style = {
            "completed": "green",
            "running": "yellow",
            "failed": "red",
        }.get(r["status"], "dim")
        table.add_row(
            str(r["id"]),
            r["spec_id"],
            r["started_at"][:19],
            f"[{status_style}]{r['status']}[/{status_style}]",
            str(r["n_variants"] or ""),
            str(r["n_oracles"] or ""),
        )

    console.print(table)
    store.close()


@db.command(name="scores")
@click.argument("run_id", type=int)
@click.option("--oracle", "-o", default=None, help="Filter by oracle name")
@click.pass_context
def db_scores(ctx, run_id, oracle):
    """Show scores for a run."""
    store = ctx.obj["store"]
    scores = store.get_scores(run_id, oracle_name=oracle)
    if not scores:
        console.print(f"[dim]No scores found for run {run_id}.[/dim]")
        return

    table = Table(title=f"Scores for Run #{run_id}")
    table.add_column("Sample", justify="right", style="cyan")
    table.add_column("Oracle")
    table.add_column("Score", justify="right", style="yellow")

    for s in scores:
        table.add_row(
            str(s["sample_index"]),
            s["oracle_name"],
            f"{s['score']:.2f}",
        )

    console.print(table)
    store.close()


@db.command(name="export")
@click.option("--run-id", "-r", type=int, default=None, help="Filter to a run")
@click.option(
    "--min-score", "-s", type=float, default=4.0, help="Minimum score threshold"
)
@click.option("--oracle", "-o", default=None, help="Filter by oracle name")
@click.option("--output", "-f", default=None, help="Output JSONL file path")
@click.pass_context
def db_export(ctx, run_id, min_score, oracle, output):
    """Export high-quality prompt/response pairs for fine-tuning."""
    store = ctx.obj["store"]
    pairs = store.export_for_finetuning(
        run_id=run_id, min_score=min_score, oracle_name=oracle
    )

    if not pairs:
        console.print("[dim]No pairs match the criteria.[/dim]")
        return

    console.print(f"[green]Found {len(pairs)} pairs with score >= {min_score}[/green]")

    if output:
        from pathlib import Path

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        console.print(f"[green]✓ Exported to {output_path}[/green]")
    else:
        for pair in pairs:
            console.print(
                f"[cyan]Score {pair['score']:.1f}[/cyan] ({pair['oracle_name']}): "
                f"{pair['prompt'][:80]}..."
            )

    store.close()


if __name__ == "__main__":
    metareason()
