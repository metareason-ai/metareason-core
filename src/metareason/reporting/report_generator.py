import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from jinja2 import Environment, PackageLoader
from scipy.stats import gaussian_kde

from metareason.config.models import SpecConfig
from metareason.pipeline.runner import SampleResult


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
        if self.spec_config.analysis:
            hdi_prob = self.spec_config.analysis.hdi_probability

        pipeline_stages = []
        for stage in self.spec_config.pipeline:
            pipeline_stages.append(
                {
                    "model": stage.model,
                    "adapter": stage.adapter.name,
                    "temperature": stage.temperature,
                    "top_p": stage.top_p,
                    "max_tokens": stage.max_tokens,
                }
            )

        oracle_configs = {}
        for name, cfg in self.spec_config.oracles.items():
            oracle_configs[name] = {
                "type": cfg.type,
                "model": cfg.model,
                "adapter": cfg.adapter.name,
                "temperature": cfg.temperature,
            }

        return {
            "title": f"MetaReason Evaluation Report: {self.spec_config.spec_id}",
            "spec_id": self.spec_config.spec_id,
            "n_variants": len(self.results),
            "n_oracles": len(self.spec_config.oracles),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hdi_pct": int(hdi_prob * 100),
            "oracle_analyses": self.analysis_results,
            "results": self.results,
            "pipeline_stages": pipeline_stages,
            "oracle_configs": oracle_configs,
            "primary_model": self.spec_config.pipeline[0].model,
        }

    def _generate_chart_data(self) -> dict:
        """Generate JSON-serializable chart data for Chart.js."""
        chart_data = {}

        for oracle_name, analysis in self.analysis_results.items():
            oracle_data = {}
            scores = np.array([r.evaluations[oracle_name].score for r in self.results])

            # Posterior KDE
            mean = analysis["population_mean"]
            hdi_width = analysis["hdi_upper"] - analysis["hdi_lower"]
            std_approx = hdi_width / 3.3
            posterior_samples = np.random.normal(mean, std_approx, 4000)

            kde = gaussian_kde(posterior_samples)
            x = np.linspace(
                posterior_samples.min() - 0.5, posterior_samples.max() + 0.5, 80
            )
            y = kde(x)
            oracle_data["posterior_x"] = [round(float(v), 4) for v in x]
            oracle_data["posterior_y"] = [round(float(v), 4) for v in y]

            # Score histogram
            bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
            counts, _ = np.histogram(scores, bins=bins)
            oracle_data["histogram_labels"] = ["1", "2", "3", "4", "5"]
            oracle_data["histogram_counts"] = [int(c) for c in counts]
            oracle_data["score_mean"] = round(float(np.mean(scores)), 3)

            # Noise KDE
            noise_mean = analysis["oracle_noise_mean"]
            noise_hdi = analysis["oracle_noise_hdi"]
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
            oracle_data["noise_x"] = [round(float(v), 4) for v in nx]
            oracle_data["noise_y"] = [round(float(v), 4) for v in ny]

            # Analysis values for chart annotations
            oracle_data["hdi_lower"] = round(float(analysis["hdi_lower"]), 4)
            oracle_data["hdi_upper"] = round(float(analysis["hdi_upper"]), 4)
            oracle_data["population_mean"] = round(
                float(analysis["population_mean"]), 4
            )
            oracle_data["population_median"] = round(
                float(analysis["population_median"]), 4
            )
            oracle_data["noise_mean"] = round(float(noise_mean), 4)
            oracle_data["noise_hdi_lower"] = round(float(noise_hdi[0]), 4)
            oracle_data["noise_hdi_upper"] = round(float(noise_hdi[1]), 4)

            # Parameter space scatter data
            if self.spec_config.axes:
                continuous_axes = [
                    a for a in self.spec_config.axes if a.type == "continuous"
                ]
                if len(continuous_axes) >= 2:
                    oracle_data["has_parameter_space"] = True
                    pairs = []
                    from itertools import combinations

                    for ax_x, ax_y in combinations(continuous_axes, 2):
                        points = []
                        for i, r in enumerate(self.results):
                            points.append(
                                {
                                    "x": round(float(r.sample_params[ax_x.name]), 4),
                                    "y": round(float(r.sample_params[ax_y.name]), 4),
                                    "score": float(scores[i]),
                                }
                            )
                        pairs.append(
                            {
                                "x_label": ax_x.name,
                                "y_label": ax_y.name,
                                "points": points,
                            }
                        )
                    oracle_data["parameter_pairs"] = pairs
                else:
                    oracle_data["has_parameter_space"] = False
            else:
                oracle_data["has_parameter_space"] = False

            chart_data[oracle_name] = oracle_data

        return chart_data

    def _render_template(self, data: dict, chart_data: dict) -> str:
        """Render the Jinja2 template with data and chart data."""
        template = self.env.get_template("report.html.j2")
        return template.render(chart_data=chart_data, **data)
