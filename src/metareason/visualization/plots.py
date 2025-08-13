"""Statistical plots and visualizations for evaluation results."""

import logging
from pathlib import Path
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..pipeline.models import PipelineResult

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Create statistical visualizations for evaluation results."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (10, 6)):
        """Initialize result visualizer.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Visualization features disabled.")

    def plot_oracle_scores(
        self,
        result: PipelineResult,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot oracle score distributions.

        Args:
            result: Pipeline result with oracle evaluations
            output_path: Optional path to save plot
            show: Whether to display plot

        Returns:
            Path to saved plot or None if not saved
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot create plots: matplotlib not available")
            return None

        if not result.oracle_results:
            logger.warning("No oracle results to plot")
            return None

        plt.style.use(self.style)

        # Create subplots for each oracle
        n_oracles = len(result.oracle_results)
        fig, axes = plt.subplots(
            1, n_oracles, figsize=(self.figsize[0] * n_oracles, self.figsize[1])
        )

        if n_oracles == 1:
            axes = [axes]

        for i, (oracle_name, oracle_results) in enumerate(
            result.oracle_results.items()
        ):
            if oracle_results:
                scores = [r.score for r in oracle_results]

                # Create histogram
                axes[i].hist(
                    scores, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
                )
                axes[i].set_title(f"{oracle_name} Score Distribution")
                axes[i].set_xlabel("Score")
                axes[i].set_ylabel("Frequency")

                # Add mean line
                mean_score = np.mean(scores)
                axes[i].axvline(
                    mean_score,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_score:.3f}",
                )
                axes[i].legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Oracle scores plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_bayesian_posteriors(
        self,
        result: PipelineResult,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot Bayesian posterior distributions.

        Args:
            result: Pipeline result with Bayesian analysis
            output_path: Optional path to save plot
            show: Whether to display plot

        Returns:
            Path to saved plot or None if not saved
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot create plots: matplotlib not available")
            return None

        if (
            not result.bayesian_results
            or not result.bayesian_results.individual_results
        ):
            logger.warning("No Bayesian results to plot")
            return None

        plt.style.use(self.style)

        # Create subplots for each oracle
        n_oracles = len(result.bayesian_results.individual_results)
        fig, axes = plt.subplots(
            1, n_oracles, figsize=(self.figsize[0] * n_oracles, self.figsize[1])
        )

        if n_oracles == 1:
            axes = [axes]

        for i, (oracle_name, bayesian_result) in enumerate(
            result.bayesian_results.individual_results.items()
        ):
            # Create normal distribution approximation of posterior
            x = np.linspace(0, 1, 1000)
            posterior_pdf = self._normal_pdf(
                x, bayesian_result.posterior_mean, bayesian_result.posterior_std
            )

            # Plot posterior
            axes[i].plot(x, posterior_pdf, "b-", linewidth=2, label="Posterior")

            # Fill HDI area
            hdi_mask = (x >= bayesian_result.hdi_lower) & (
                x <= bayesian_result.hdi_upper
            )
            axes[i].fill_between(
                x,
                0,
                posterior_pdf,
                where=hdi_mask,
                alpha=0.3,
                color="blue",
                label="95% HDI",
            )

            # Add mean line
            axes[i].axvline(
                bayesian_result.posterior_mean,
                color="red",
                linestyle="--",
                label=f"Mean: {bayesian_result.posterior_mean:.3f}",
            )

            axes[i].set_title(f"{oracle_name} Posterior Distribution")
            axes[i].set_xlabel("Success Probability")
            axes[i].set_ylabel("Density")
            axes[i].legend()

            # Add convergence info in corner
            conv_text = f"R-hat: {bayesian_result.r_hat:.3f}\n"
            conv_text += f"ESS: {bayesian_result.effective_sample_size:.0f}\n"
            conv_text += f"Converged: {'Yes' if bayesian_result.converged else 'No'}"
            axes[i].text(
                0.02,
                0.98,
                conv_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Bayesian posteriors plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def plot_pipeline_performance(
        self,
        result: PipelineResult,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot pipeline step performance metrics.

        Args:
            result: Pipeline result
            output_path: Optional path to save plot
            show: Whether to display plot

        Returns:
            Path to saved plot or None if not saved
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot create plots: matplotlib not available")
            return None

        if not result.step_results:
            logger.warning("No step results to plot")
            return None

        plt.style.use(self.style)

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(self.figsize[0] * 2, self.figsize[1])
        )

        # Success rates
        step_names = [f"Step {i+1}" for i in range(len(result.step_results))]
        success_rates = [step.success_rate for step in result.step_results]

        bars1 = ax1.bar(step_names, success_rates, color="lightgreen", alpha=0.7)
        ax1.set_title("Pipeline Step Success Rates")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )

        # Execution times
        exec_times = [step.timing.get("total_time", 0) for step in result.step_results]

        bars2 = ax2.bar(step_names, exec_times, color="lightblue", alpha=0.7)
        ax2.set_title("Pipeline Step Execution Times")
        ax2.set_ylabel("Time (seconds)")

        # Add value labels on bars
        for bar, time in zip(bars2, exec_times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(exec_times) * 0.01,
                f"{time:.1f}s",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Pipeline performance plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return output_path

    def generate_dashboard(
        self, result: PipelineResult, output_dir: Path, include_plots: bool = True
    ) -> Path:
        """Generate complete dashboard with plots and HTML report.

        Args:
            result: Pipeline result
            output_dir: Directory to save dashboard files
            include_plots: Whether to generate plot images

        Returns:
            Path to main dashboard HTML file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = {}

        if include_plots and MATPLOTLIB_AVAILABLE:
            # Generate all plots
            oracle_plot = output_dir / "oracle_scores.png"
            bayesian_plot = output_dir / "bayesian_posteriors.png"
            performance_plot = output_dir / "pipeline_performance.png"

            plot_paths["oracle_scores"] = self.plot_oracle_scores(result, oracle_plot)
            plot_paths["bayesian_posteriors"] = self.plot_bayesian_posteriors(
                result, bayesian_plot
            )
            plot_paths["pipeline_performance"] = self.plot_pipeline_performance(
                result, performance_plot
            )

        # Generate enhanced HTML report
        html_path = output_dir / "dashboard.html"
        html_content = self._generate_dashboard_html(result, plot_paths)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Dashboard generated at {html_path}")
        return html_path

    def _normal_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Calculate normal probability density function."""
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def _generate_dashboard_html(
        self, result: PipelineResult, plot_paths: Dict[str, Optional[Path]]
    ) -> str:
        """Generate enhanced HTML dashboard with plots.

        Args:
            result: Pipeline result
            plot_paths: Dictionary of plot file paths

        Returns:
            HTML content string
        """
        from ..results.exporter import ResultExporter

        # Get basic HTML report
        exporter = ResultExporter()
        basic_html = exporter._generate_html_report(result)

        # Add plots section before closing body tag
        plots_section = """
            <div class="section">
                <h3>Visualizations</h3>
        """

        if plot_paths.get("oracle_scores"):
            plots_section += """
                <div class="plot">
                    <h4>Oracle Score Distributions</h4>
                    <img src="oracle_scores.png" alt="Oracle Scores" style="max-width: 100%;">
                </div>
            """

        if plot_paths.get("bayesian_posteriors"):
            plots_section += """
                <div class="plot">
                    <h4>Bayesian Posterior Distributions</h4>
                    <img src="bayesian_posteriors.png" alt="Bayesian Posteriors" style="max-width: 100%;">
                </div>
            """

        if plot_paths.get("pipeline_performance"):
            plots_section += """
                <div class="plot">
                    <h4>Pipeline Performance</h4>
                    <img src="pipeline_performance.png" alt="Pipeline Performance" style="max-width: 100%;">
                </div>
            """

        plots_section += """
            </div>
        """

        # Insert plots section before closing body tag
        enhanced_html = basic_html.replace("</body>", plots_section + "</body>")

        # Add CSS for plots
        plot_css = """
            .plot { margin: 20px 0; }
            .plot h4 { margin-bottom: 10px; }
            .plot img { border: 1px solid #ddd; border-radius: 5px; }
        """

        enhanced_html = enhanced_html.replace("</style>", plot_css + "</style>")

        return enhanced_html
