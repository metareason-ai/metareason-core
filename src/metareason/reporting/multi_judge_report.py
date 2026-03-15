import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from jinja2 import Environment, PackageLoader
from scipy.stats import gaussian_kde

from metareason.config.models import CalibrateMultiConfig
from metareason.reporting.report_generator import _load_vendor_asset


class MultiJudgeReportGenerator:
    """Generates self-contained HTML reports from multi-judge calibration results.

    Args:
        config: CalibrateMultiConfig used for the calibration.
        scores_by_oracle: Dict mapping oracle name to list of scores.
        multi_judge_result: Dict from BayesianAnalyzer.estimate_multi_judge_quality().
    """

    def __init__(
        self,
        config: CalibrateMultiConfig,
        scores_by_oracle: Dict[str, List[float]],
        multi_judge_result: dict,
    ):
        self.config = config
        self.scores_by_oracle = scores_by_oracle
        self.multi_judge = multi_judge_result
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

        has_expected = "expected_score" in self.multi_judge

        # Per-judge verdicts
        judge_verdicts = {}
        for name, info in self.multi_judge["judges"].items():
            abs_bias = abs(info["bias_mean"])
            noise = info["noise_mean"]
            direction = "higher" if info["bias_mean"] > 0 else "lower"

            if has_expected:
                if abs_bias < 0.2 and noise < 0.2:
                    verdict = "Well-calibrated"
                    verdict_class = "good"
                elif abs_bias >= 0.2 and noise < 0.2:
                    verdict = f"Consistently {direction} by ~{info['bias_mean']:+.2f}"
                    verdict_class = "warn"
                elif abs_bias < 0.2 and noise >= 0.2:
                    verdict = "Accurate on average but inconsistent"
                    verdict_class = "warn"
                else:
                    verdict = f"Biased ({info['bias_mean']:+.2f}) and noisy"
                    verdict_class = "bad"
            else:
                if noise < 0.2:
                    verdict = "Consistent"
                    verdict_class = "good"
                else:
                    verdict = f"Inconsistent (±{noise:.2f})"
                    verdict_class = "warn"

            judge_verdicts[name] = {
                "verdict": verdict,
                "verdict_class": verdict_class,
            }

        # Noise threshold recommendation
        recommendations = []
        for name, info in self.multi_judge["judges"].items():
            if info["noise_mean"] > 1.0:
                recommendations.append(
                    f"Consider dropping {name} (noise={info['noise_mean']:.2f})"
                )

        return {
            "title": f"MetaReason Multi-Judge Report: {self.config.spec_id}",
            "spec_id": self.config.spec_id,
            "repeats": self.config.repeats,
            "n_judges": self.multi_judge["n_judges"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hdi_pct": int(hdi_prob * 100),
            "multi_judge": self.multi_judge,
            "has_expected": has_expected,
            "expected_score": self.multi_judge.get("expected_score"),
            "judge_verdicts": judge_verdicts,
            "recommendations": recommendations,
            "prompt": self.config.prompt,
            "response": self.config.response,
        }

    def _generate_chart_data(self) -> dict:
        """Generate JSON-serializable chart data for Chart.js."""
        chart_data = {}
        has_expected = "expected_score" in self.multi_judge

        # Per-judge score histograms
        judge_names = list(self.scores_by_oracle.keys())
        chart_data["judge_names"] = judge_names
        chart_data["judge_histograms"] = {}
        bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        for name, scores in self.scores_by_oracle.items():
            counts, _ = np.histogram(scores, bins=bins)
            chart_data["judge_histograms"][name] = [int(c) for c in counts]

        chart_data["histogram_labels"] = ["1", "2", "3", "4", "5"]

        # Per-judge bias posterior KDEs
        chart_data["bias_posteriors"] = {}
        for name, info in self.multi_judge["judges"].items():
            bias_mean = info["bias_mean"]
            bias_lo, bias_hi = info["bias_hdi"]
            bias_width = bias_hi - bias_lo
            bias_std = max(bias_width / 3.3, 0.01)
            samples = np.random.normal(bias_mean, bias_std, 4000)

            kde = gaussian_kde(samples)
            x = np.linspace(samples.min() - 0.3, samples.max() + 0.3, 60)
            y = kde(x)
            chart_data["bias_posteriors"][name] = {
                "x": [round(float(v), 4) for v in x],
                "y": [round(float(v), 4) for v in y],
                "mean": round(float(bias_mean), 4),
                "hdi_lower": round(float(bias_lo), 4),
                "hdi_upper": round(float(bias_hi), 4),
            }

        # Per-judge noise posterior KDEs
        chart_data["noise_posteriors"] = {}
        for name, info in self.multi_judge["judges"].items():
            noise_mean = info["noise_mean"]
            noise_lo, noise_hi = info["noise_hdi"]
            noise_width = noise_hi - noise_lo
            noise_std = max(noise_width / 3.3, 0.01)
            samples = np.abs(np.random.normal(noise_mean, noise_std, 4000))

            kde = gaussian_kde(samples)
            x = np.linspace(max(0, samples.min() - 0.1), samples.max() + 0.1, 60)
            y = kde(x)
            chart_data["noise_posteriors"][name] = {
                "x": [round(float(v), 4) for v in x],
                "y": [round(float(v), 4) for v in y],
                "mean": round(float(noise_mean), 4),
                "hdi_lower": round(float(noise_lo), 4),
                "hdi_upper": round(float(noise_hi), 4),
            }

        chart_data["has_expected"] = has_expected

        return chart_data

    def _render_template(self, data: dict, chart_data: dict) -> str:
        """Render the Jinja2 template with data and chart data."""
        template = self.env.get_template("multi_judge_report.html.j2")
        return template.render(
            chart_data=chart_data,
            chartjs_source=_load_vendor_asset("chart.umd.min.js"),
            chartjs_annotation_source=_load_vendor_asset(
                "chartjs-plugin-annotation.min.js"
            ),
            **data,
        )
