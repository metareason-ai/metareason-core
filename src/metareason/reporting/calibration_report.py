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
    """Generates self-contained HTML reports from judge calibration results.

    Args:
        config: CalibrateConfig used for the calibration.
        scores: List of raw scores from repeated evaluations.
        analysis_result: Dict from BayesianAnalyzer.estimate_judge_calibration().
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

        has_expected = "expected_score" in self.analysis

        # Build verdict
        noise = self.analysis["noise_mean"]
        if has_expected:
            abs_bias = abs(self.analysis["bias_mean"])
            direction = "higher" if self.analysis["bias_mean"] > 0 else "lower"
            if abs_bias < 0.2 and noise < 0.2:
                verdict = "Well-calibrated. Accurate and consistent."
                verdict_class = "good"
            elif abs_bias >= 0.2 and noise < 0.2:
                verdict = (
                    f"Consistently {direction}. "
                    f"Usable if you adjust by ~{self.analysis['bias_mean']:+.2f}."
                )
                verdict_class = "warn"
            elif abs_bias < 0.2 and noise >= 0.2:
                verdict = (
                    "Accurate on average but inconsistent. "
                    "Consider more repeats or lower temperature."
                )
                verdict_class = "warn"
            else:
                verdict = (
                    f"Both biased ({self.analysis['bias_mean']:+.2f}) and noisy "
                    f"(±{noise:.1f}). Consider a different judge."
                )
                verdict_class = "bad"
        else:
            verdict = None
            verdict_class = None

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
            "expected_score": self.analysis.get("expected_score"),
            "verdict": verdict,
            "verdict_class": verdict_class,
        }

    def _generate_chart_data(self) -> dict:
        """Generate JSON-serializable chart data for Chart.js."""
        chart_data = {}
        scores = np.array(self.scores)
        has_expected = "expected_score" in self.analysis

        if has_expected:
            # Bias posterior KDE
            bias_mean = self.analysis["bias_mean"]
            bias_lo, bias_hi = self.analysis["bias_hdi"]
            bias_width = bias_hi - bias_lo
            bias_std = max(bias_width / 3.3, 0.01)
            bias_samples = np.random.normal(bias_mean, bias_std, 4000)

            kde = gaussian_kde(bias_samples)
            x = np.linspace(bias_samples.min() - 0.5, bias_samples.max() + 0.5, 80)
            y = kde(x)
            chart_data["bias_x"] = [round(float(v), 4) for v in x]
            chart_data["bias_y"] = [round(float(v), 4) for v in y]
            chart_data["bias_mean"] = round(float(bias_mean), 4)
            chart_data["bias_hdi_lower"] = round(float(bias_lo), 4)
            chart_data["bias_hdi_upper"] = round(float(bias_hi), 4)

        # Score histogram
        bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        counts, _ = np.histogram(scores, bins=bins)
        chart_data["histogram_labels"] = ["1", "2", "3", "4", "5"]
        chart_data["histogram_counts"] = [int(c) for c in counts]
        chart_data["score_mean"] = round(float(np.mean(scores)), 3)

        # Noise KDE
        noise_mean = self.analysis["noise_mean"]
        noise_lo, noise_hi = self.analysis["noise_hdi"]
        noise_width = noise_hi - noise_lo
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
        chart_data["noise_mean"] = round(float(noise_mean), 4)
        chart_data["noise_hdi_lower"] = round(float(noise_lo), 4)
        chart_data["noise_hdi_upper"] = round(float(noise_hi), 4)

        chart_data["has_expected"] = has_expected

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
