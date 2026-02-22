import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from jinja2 import Environment, PackageLoader
from scipy.stats import gaussian_kde

from metareason.config.models import CalibrateConfig
from metareason.reporting.report_generator import _load_vendor_asset


class CalibrationReportGenerator:
    """Generates self-contained HTML reports from calibration results.

    Args:
        config: CalibrateConfig used for the calibration.
        scores: List of raw scores from repeated evaluations.
        analysis_result: Dict from BayesianAnalyzer.estimate_population_quality().
    """

    def __init__(
        self,
        config: CalibrateConfig,
        scores: List[float],
        analysis_result: dict,
    ):
        self.config = config
        self.scores = scores
        self.analysis = analysis_result
        self.env = Environment(  # nosec B701 - autoescape off for JSON embedding
            loader=PackageLoader("metareason.reporting", "templates"),
            autoescape=False,
        )
        self.env.filters["tojson"] = json.dumps

    def generate_html(self, output_path: Path) -> Path:
        """Generate HTML report and save to output_path."""
        chart_data = self._generate_chart_data()
        data = self._collect_data()
        html = self._render_template(data, chart_data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        return output_path

    def _collect_data(self) -> dict:
        """Extract data for template context."""
        hdi_prob = 0.94
        if self.config.analysis:
            hdi_prob = self.config.analysis.hdi_probability

        has_expected = self.config.expected_score is not None
        bias = None
        within_hdi = None
        if has_expected:
            bias = self.analysis["population_mean"] - self.config.expected_score
            within_hdi = (
                self.analysis["hdi_lower"]
                <= self.config.expected_score
                <= self.analysis["hdi_upper"]
            )

        return {
            "title": f"MetaReason Calibration Report: {self.config.spec_id}",
            "spec_id": self.config.spec_id,
            "repeats": self.config.repeats,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hdi_pct": int(hdi_prob * 100),
            "analysis": self.analysis,
            "scores": self.scores,
            "oracle_model": self.config.oracle.model,
            "oracle_adapter": self.config.oracle.adapter.name,
            "oracle_temperature": self.config.oracle.temperature,
            "oracle_rubric": self.config.oracle.rubric or "",
            "prompt": self.config.prompt,
            "response": self.config.response,
            "has_expected": has_expected,
            "expected_score": self.config.expected_score,
            "bias": bias,
            "within_hdi": within_hdi,
        }

    def _generate_chart_data(self) -> dict:
        """Generate JSON-serializable chart data for Chart.js."""
        chart_data = {}
        scores = np.array(self.scores)

        # Posterior KDE
        mean = self.analysis["population_mean"]
        hdi_width = self.analysis["hdi_upper"] - self.analysis["hdi_lower"]
        std_approx = hdi_width / 3.3
        posterior_samples = np.random.normal(mean, std_approx, 4000)

        kde = gaussian_kde(posterior_samples)
        x = np.linspace(
            posterior_samples.min() - 0.5, posterior_samples.max() + 0.5, 80
        )
        y = kde(x)
        chart_data["posterior_x"] = [round(float(v), 4) for v in x]
        chart_data["posterior_y"] = [round(float(v), 4) for v in y]

        # Score histogram
        bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        counts, _ = np.histogram(scores, bins=bins)
        chart_data["histogram_labels"] = ["1", "2", "3", "4", "5"]
        chart_data["histogram_counts"] = [int(c) for c in counts]
        chart_data["score_mean"] = round(float(np.mean(scores)), 3)

        # Noise KDE
        noise_mean = self.analysis["oracle_noise_mean"]
        noise_hdi = self.analysis["oracle_noise_hdi"]
        noise_width = noise_hdi[1] - noise_hdi[0]
        noise_std = max(noise_width / 3.3, 0.01)
        noise_samples = np.abs(np.random.normal(noise_mean, noise_std, 4000))

        noise_kde = gaussian_kde(noise_samples)
        nx = np.linspace(
            max(0, noise_samples.min() - 0.2),
            noise_samples.max() + 0.2,
            80,
        )
        ny = noise_kde(nx)
        chart_data["noise_x"] = [round(float(v), 4) for v in nx]
        chart_data["noise_y"] = [round(float(v), 4) for v in ny]

        # Analysis values for chart annotations
        chart_data["hdi_lower"] = round(float(self.analysis["hdi_lower"]), 4)
        chart_data["hdi_upper"] = round(float(self.analysis["hdi_upper"]), 4)
        chart_data["population_mean"] = round(
            float(self.analysis["population_mean"]), 4
        )
        chart_data["population_median"] = round(
            float(self.analysis["population_median"]), 4
        )
        chart_data["noise_mean"] = round(float(noise_mean), 4)
        chart_data["noise_hdi_lower"] = round(float(noise_hdi[0]), 4)
        chart_data["noise_hdi_upper"] = round(float(noise_hdi[1]), 4)

        return chart_data

    def _render_template(self, data: dict, chart_data: dict) -> str:
        """Render the Jinja2 template with data and chart data."""
        template = self.env.get_template("calibration_report.html.j2")
        return template.render(
            chart_data=chart_data,
            chartjs_source=_load_vendor_asset("chart.umd.min.js"),
            chartjs_annotation_source=_load_vendor_asset(
                "chartjs-plugin-annotation.min.js"
            ),
            **data,
        )
