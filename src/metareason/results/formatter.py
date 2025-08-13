"""Results formatter for CLI output and reporting."""

import logging

from rich.console import Console
from rich.table import Table

from ..pipeline.models import ExecutionPlan, PipelineResult

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Format pipeline results for CLI output and reports."""

    def __init__(self, console: Console = None):
        """Initialize result formatter.

        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()

    def format_summary(self, result: PipelineResult) -> str:
        """Format a concise summary of pipeline results.

        Args:
            result: Pipeline result to format

        Returns:
            Formatted summary string
        """
        summary = result.get_summary()

        lines = []

        # Header
        lines.append(
            f"🚀 [bold blue]Evaluation Complete: {result.config.spec_id}[/bold blue]"
        )
        lines.append(f"📄 Execution ID: {result.execution_id}")
        lines.append("")

        # Status
        if summary["status"] == "success":
            lines.append("✅ [green]Status: SUCCESS[/green]")
        else:
            lines.append("⚠️  [yellow]Status: PARTIAL SUCCESS[/yellow]")

        # Key metrics
        lines.append(
            f"📊 Overall Success Rate: [bold]{summary['overall_success_rate']:.1%}[/bold]"
        )
        lines.append(f"🔢 Total Samples: {summary['total_samples']:,}")
        lines.append(f"💬 Total Responses: {summary['total_responses']:,}")
        lines.append(f"⏱️  Execution Time: {summary['execution_time']:.2f}s")
        lines.append("")

        # Pipeline steps
        lines.append("📋 [bold]Pipeline Steps:[/bold]")
        for i, step in enumerate(result.step_results):
            status_icon = "✅" if step.is_successful else "❌"
            lines.append(
                f"   {i+1}. {status_icon} {step.step_name} "
                f"({step.success_rate:.1%}, {len(step.responses)} responses)"
            )
        lines.append("")

        # Oracle results
        if result.oracle_results:
            lines.append("🔍 [bold]Oracle Evaluations:[/bold]")
            for oracle_name, oracle_results in result.oracle_results.items():
                if oracle_results:
                    avg_score = sum(r.score for r in oracle_results) / len(
                        oracle_results
                    )
                    lines.append(
                        f"   • {oracle_name}: {avg_score:.3f} avg "
                        f"({len(oracle_results)} evaluations)"
                    )
            lines.append("")

        # Bayesian analysis
        if result.bayesian_results:
            if result.bayesian_results.all_converged:
                lines.append(
                    "📈 [green]Bayesian Analysis: All oracles converged[/green]"
                )
            else:
                lines.append(
                    "📈 [yellow]Bayesian Analysis: Some convergence issues[/yellow]"
                )

            if result.bayesian_results.combined_confidence:
                lines.append(
                    f"🎯 Combined Confidence: {result.bayesian_results.combined_confidence:.3f}"
                )

        return "\n".join(lines)

    def format_detailed_report(self, result: PipelineResult) -> str:
        """Format a detailed report of pipeline results.

        Args:
            result: Pipeline result to format

        Returns:
            Formatted detailed report string
        """
        lines = []

        # Header with full details
        lines.append("=" * 80)
        lines.append(f"MetaReason Evaluation Report: {result.config.spec_id}")
        lines.append("=" * 80)
        lines.append(f"Execution ID: {result.execution_id}")
        lines.append(f"Start Time: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if result.end_time:
            lines.append(f"End Time: {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Duration: {result.total_duration:.2f} seconds")
        lines.append("")

        # Configuration summary
        lines.append("Configuration:")
        lines.append(f"  Samples: {result.config.n_variants:,}")
        lines.append(f"  Pipeline Steps: {len(result.config.pipeline)}")
        if result.config.sampling:
            lines.append(f"  Sampling Method: {result.config.sampling.method}")
        lines.append("")

        # Detailed step results
        lines.append("Pipeline Step Results:")
        for i, step in enumerate(result.step_results):
            lines.append(f"  Step {i+1}: {step.step_name}")
            lines.append(f"    Success Rate: {step.success_rate:.1%}")
            lines.append(f"    Prompts: {len(step.prompts)}")
            lines.append(f"    Responses: {len(step.responses)}")
            lines.append(f"    Errors: {len(step.errors)}")
            if step.timing:
                lines.append(
                    f"    Execution Time: {step.timing.get('total_time', 0):.2f}s"
                )
            if step.errors:
                lines.append("    Error Details:")
                for error in step.errors[:3]:  # Show first 3 errors
                    lines.append(f"      - {str(error)[:100]}...")
            lines.append("")

        # Oracle evaluation details
        if result.oracle_results:
            lines.append("Oracle Evaluation Results:")
            for oracle_name, oracle_results in result.oracle_results.items():
                lines.append(f"  {oracle_name}:")
                lines.append(f"    Evaluations: {len(oracle_results)}")
                if oracle_results:
                    scores = [r.score for r in oracle_results]
                    lines.append(f"    Average Score: {sum(scores)/len(scores):.3f}")
                    lines.append(f"    Min Score: {min(scores):.3f}")
                    lines.append(f"    Max Score: {max(scores):.3f}")
                    if len(scores) > 1:
                        variance = sum(
                            (s - sum(scores) / len(scores)) ** 2 for s in scores
                        ) / len(scores)
                        lines.append(f"    Std Dev: {variance**0.5:.3f}")
                lines.append("")

        # Bayesian analysis details
        if result.bayesian_results:
            lines.append("Bayesian Analysis Results:")
            lines.append(
                f"  All Converged: {'Yes' if result.bayesian_results.all_converged else 'No'}"
            )
            lines.append(
                f"  All Reliable: {'Yes' if result.bayesian_results.all_reliable else 'No'}"
            )
            if result.bayesian_results.combined_confidence:
                lines.append(
                    f"  Combined Confidence: {result.bayesian_results.combined_confidence:.3f}"
                )
            lines.append("")

            lines.append("  Individual Oracle Results:")
            for (
                oracle_name,
                bayesian_result,
            ) in result.bayesian_results.individual_results.items():
                lines.append(f"    {oracle_name}:")
                lines.append(
                    f"      Posterior Mean: {bayesian_result.posterior_mean:.3f}"
                )
                lines.append(
                    f"      95% HDI: [{bayesian_result.hdi_lower:.3f}, {bayesian_result.hdi_upper:.3f}]"
                )
                lines.append(f"      Success Rate: {bayesian_result.success_rate:.3f}")
                lines.append(
                    f"      Converged: {'Yes' if bayesian_result.converged else 'No'}"
                )
                lines.append(
                    f"      Reliable: {'Yes' if bayesian_result.reliable else 'No'}"
                )
                lines.append(f"      R-hat: {bayesian_result.r_hat:.3f}")
                lines.append(f"      ESS: {bayesian_result.effective_sample_size:.0f}")
                lines.append("")

        return "\n".join(lines)

    def create_summary_table(self, result: PipelineResult) -> Table:
        """Create a rich table summarizing pipeline results.

        Args:
            result: Pipeline result to summarize

        Returns:
            Rich table with results summary
        """
        table = Table(
            title=f"Evaluation Results: {result.config.spec_id}",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        summary = result.get_summary()

        # Basic metrics
        table.add_row("Execution ID", result.execution_id)
        table.add_row("Status", summary["status"])
        table.add_row("Overall Success Rate", f"{summary['overall_success_rate']:.1%}")
        table.add_row("Total Samples", f"{summary['total_samples']:,}")
        table.add_row("Total Responses", f"{summary['total_responses']:,}")
        table.add_row("Execution Time", f"{summary['execution_time']:.2f}s")
        table.add_row("Pipeline Steps", str(summary["pipeline_steps"]))
        table.add_row("Oracles Evaluated", str(summary["oracles_evaluated"]))

        if result.bayesian_results:
            table.add_row("Bayesian Analysis", "✅ Completed")
            table.add_row(
                "All Converged",
                "✅ Yes" if result.bayesian_results.all_converged else "❌ No",
            )
            if result.bayesian_results.combined_confidence:
                table.add_row(
                    "Combined Confidence",
                    f"{result.bayesian_results.combined_confidence:.3f}",
                )

        return table

    def create_step_results_table(self, result: PipelineResult) -> Table:
        """Create a table showing individual step results.

        Args:
            result: Pipeline result

        Returns:
            Rich table with step results
        """
        table = Table(
            title="Pipeline Step Results", show_header=True, header_style="bold blue"
        )
        table.add_column("Step", style="cyan")
        table.add_column("Adapter/Model", style="yellow")
        table.add_column("Success Rate", style="green")
        table.add_column("Responses", style="magenta")
        table.add_column("Errors", style="red")
        table.add_column("Time (s)", style="blue")

        for i, step in enumerate(result.step_results):
            success_rate = f"{step.success_rate:.1%}"
            if step.success_rate >= 0.95:
                success_rate = f"[green]{success_rate}[/green]"
            elif step.success_rate >= 0.8:
                success_rate = f"[yellow]{success_rate}[/yellow]"
            else:
                success_rate = f"[red]{success_rate}[/red]"

            table.add_row(
                str(i + 1),
                step.step_name,
                success_rate,
                str(len(step.responses)),
                str(len(step.errors)),
                f"{step.timing.get('total_time', 0):.2f}",
            )

        return table

    def create_oracle_results_table(self, result: PipelineResult) -> Table:
        """Create a table showing oracle evaluation results.

        Args:
            result: Pipeline result

        Returns:
            Rich table with oracle results
        """
        table = Table(
            title="Oracle Evaluation Results",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Oracle", style="cyan")
        table.add_column("Evaluations", style="yellow")
        table.add_column("Avg Score", style="green")
        table.add_column("Std Dev", style="magenta")
        table.add_column("Min", style="blue")
        table.add_column("Max", style="blue")

        for oracle_name, oracle_results in result.oracle_results.items():
            if oracle_results:
                scores = [r.score for r in oracle_results]
                avg_score = sum(scores) / len(scores)
                std_dev = (
                    sum((s - avg_score) ** 2 for s in scores) / len(scores)
                ) ** 0.5
                min_score = min(scores)
                max_score = max(scores)

                table.add_row(
                    oracle_name,
                    str(len(oracle_results)),
                    f"{avg_score:.3f}",
                    f"{std_dev:.3f}",
                    f"{min_score:.3f}",
                    f"{max_score:.3f}",
                )

        return table

    def format_execution_plan(self, plan: ExecutionPlan) -> str:
        """Format execution plan for dry-run output.

        Args:
            plan: Execution plan to format

        Returns:
            Formatted plan string
        """
        lines = []

        lines.append("🗓️  [bold blue]Execution Plan[/bold blue]")
        lines.append(f"📄 Spec: {plan.config.spec_id}")
        lines.append("")

        lines.append("📊 [bold]Estimates:[/bold]")
        lines.append(f"   Samples: {plan.estimated_samples:,}")
        lines.append(f"   Prompts: {plan.estimated_prompts:,}")
        lines.append(f"   API Calls: {plan.estimated_api_calls:,}")
        if plan.estimated_duration:
            lines.append(f"   Duration: {plan.estimated_duration:.1f}s")
        if plan.estimated_cost:
            lines.append(f"   Cost: ${plan.estimated_cost:.2f}")
        lines.append("")

        lines.append("🔧 [bold]Pipeline Steps:[/bold]")
        for step in plan.steps:
            lines.append(
                f"   {step['step_index'] + 1}. {step['adapter']}/{step['model']} "
                f"({step['estimated_calls']:,} calls)"
            )
        lines.append("")

        # Warnings
        if plan.warnings:
            lines.append("⚠️  [bold yellow]Warnings:[/bold yellow]")
            for warning in plan.warnings:
                lines.append(f"   • [yellow]{warning}[/yellow]")
            lines.append("")

        return "\n".join(lines)
