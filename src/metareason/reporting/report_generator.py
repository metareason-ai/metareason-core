from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from jinja2 import Environment, PackageLoader

from metareason.config.models import SpecConfig
from metareason.pipeline.runner import SampleResult

from .visualizations import (
    figure_to_base64,
    plot_oracle_variability,
    plot_parameter_space,
    plot_posterior_distribution,
    plot_score_distribution,
)


class ReportGenerator:
    """Generates self-contained HTML reports from evaluation results.

    Args:
        results: List of SampleResult from evaluation pipeline.
        spec_config: The SpecConfig used for the evaluation.
        analysis_results: Dict mapping oracle_name -> population quality dict
            (output of BayesianAnalyzer.estimate_population_quality).
    """

    def __init__(
        self,
        results: List[SampleResult],
        spec_config: SpecConfig,
        analysis_results: Dict[str, dict],
    ):
        self.results = results
        self.spec_config = spec_config
        self.analysis_results = analysis_results
        self.env = (
            Environment(  # nosec B701 - autoescape off for base64 image embedding
                loader=PackageLoader("metareason.reporting", "templates"),
                autoescape=False,
            )
        )

    def generate_html(self, output_path: Path) -> Path:
        """Generate HTML report and save to output_path."""
        figures = self._generate_figures()
        data = self._collect_data()
        html = self._render_template(data, figures)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        return output_path

    def _collect_data(self) -> dict:
        """Extract data for template context."""
        hdi_prob = 0.94
        if self.spec_config.analysis:
            hdi_prob = self.spec_config.analysis.hdi_probability

        return {
            "title": f"MetaReason Report: {self.spec_config.spec_id}",
            "spec_id": self.spec_config.spec_id,
            "n_variants": len(self.results),
            "n_oracles": len(self.spec_config.oracles),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hdi_pct": int(hdi_prob * 100),
            "oracle_analyses": self.analysis_results,
            "results": self.results,
        }

    def _generate_figures(self) -> dict:
        """Generate all visualization figures as base64 strings."""
        figures = {}

        for oracle_name, analysis in self.analysis_results.items():
            oracle_figs = {}
            scores = np.array([r.evaluations[oracle_name].score for r in self.results])

            # Posterior distribution
            mean = analysis["population_mean"]
            hdi_width = analysis["hdi_upper"] - analysis["hdi_lower"]
            std_approx = hdi_width / 3.3
            posterior_samples = np.random.normal(mean, std_approx, 4000)

            fig = plot_posterior_distribution(
                posterior_samples,
                analysis["hdi_lower"],
                analysis["hdi_upper"],
                analysis["hdi_prob"],
                oracle_name,
            )
            oracle_figs["posterior"] = figure_to_base64(fig)

            # Score distribution
            fig = plot_score_distribution(scores, oracle_name)
            oracle_figs["scores"] = figure_to_base64(fig)

            # Oracle variability
            noise_mean = analysis["oracle_noise_mean"]
            noise_hdi = analysis["oracle_noise_hdi"]
            noise_width = noise_hdi[1] - noise_hdi[0]
            noise_std = max(noise_width / 3.3, 0.01)
            noise_samples = np.abs(np.random.normal(noise_mean, noise_std, 4000))

            fig = plot_oracle_variability(
                noise_samples,
                noise_hdi[0],
                noise_hdi[1],
                analysis["hdi_prob"],
                oracle_name,
            )
            oracle_figs["variability"] = figure_to_base64(fig)

            # Parameter space (if axes exist)
            if self.spec_config.axes:
                samples = [r.sample_params for r in self.results]
                fig = plot_parameter_space(
                    samples, self.spec_config.axes, scores, oracle_name
                )
                oracle_figs["parameter_space"] = figure_to_base64(fig)

            # Convergence diagnostics not available from saved results
            oracle_figs["convergence"] = None

            figures[oracle_name] = oracle_figs

        return figures

    def _render_template(self, data: dict, figures: dict) -> str:
        """Render the Jinja2 template with data and figures."""
        template = self.env.get_template("report.html.j2")
        return template.render(figures=figures, **data)
