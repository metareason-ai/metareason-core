import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from jinja2 import Environment, PackageLoader
from scipy.stats import gaussian_kde

from metareason.config.models import SpecConfig
from metareason.pipeline.runner import SampleResult

_VENDOR_DIR = Path(__file__).parent / "vendor"


def _load_vendor_asset(filename: str) -> str:
    """Load a vendored JavaScript file as a string."""
    return (_VENDOR_DIR / filename).read_text()


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

            # Posterior KDE — use real MCMC samples when available
            if "posterior_samples" in analysis:
                posterior_samples = np.array(analysis["posterior_samples"])
            else:
                # Fallback for legacy JSON files without trace data
                mean = analysis["population_mean"]
                hdi_width = analysis["hdi_upper"] - analysis["hdi_lower"]
                std_approx = hdi_width / 3.3
                rng = np.random.default_rng(42)
                posterior_samples = rng.normal(mean, std_approx, 4000)

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

            # Noise KDE — use real MCMC samples when available
            if "noise_posterior_samples" in analysis:
                noise_samples = np.array(analysis["noise_posterior_samples"])
            else:
                # Fallback for legacy JSON files without trace data
                noise_mean = analysis["oracle_noise_mean"]
                noise_hdi = analysis["oracle_noise_hdi"]
                noise_width = noise_hdi[1] - noise_hdi[0]
                noise_std = max(noise_width / 3.3, 0.01)
                rng = np.random.default_rng(43)
                noise_samples = np.abs(rng.normal(noise_mean, noise_std, 4000))

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
            oracle_data["noise_mean"] = round(
                float(analysis["oracle_noise_mean"]), 4
            )
            oracle_data["noise_hdi_lower"] = round(
                float(analysis["oracle_noise_hdi"][0]), 4
            )
            oracle_data["noise_hdi_upper"] = round(
                float(analysis["oracle_noise_hdi"][1]), 4
            )

            # Score by Parameter — mean score per value of each axis
            cat_axes = [a for a in self.spec_config.axes if a.type == "categorical"]
            if cat_axes:
                breakdowns = []
                for ax in cat_axes:
                    values = []
                    means = []
                    counts = []
                    for val in ax.values:
                        mask = [
                            str(r.sample_params.get(ax.name)) == str(val)
                            for r in self.results
                        ]
                        group_scores = scores[mask]
                        values.append(str(val))
                        means.append(
                            round(float(group_scores.mean()), 2)
                            if len(group_scores) > 0
                            else 0
                        )
                        counts.append(int(group_scores.shape[0]))
                    breakdowns.append(
                        {
                            "axis_name": ax.name,
                            "values": values,
                            "means": means,
                            "counts": counts,
                        }
                    )
                oracle_data["has_breakdowns"] = True
                oracle_data["category_breakdowns"] = breakdowns
            else:
                oracle_data["has_breakdowns"] = False

            # Interaction heatmap — 2 most impactful categoricals
            if len(cat_axes) >= 2:
                # Pick the 2 categoricals with the most score variance
                axis_variance = []
                for ax in cat_axes:
                    group_means = []
                    for val in ax.values:
                        mask = [
                            str(r.sample_params.get(ax.name)) == str(val)
                            for r in self.results
                        ]
                        gs = scores[mask]
                        if len(gs) > 0:
                            group_means.append(float(gs.mean()))
                    axis_variance.append(
                        (np.var(group_means) if group_means else 0, ax)
                    )
                axis_variance.sort(key=lambda t: t[0], reverse=True)
                ax_row = axis_variance[0][1]
                ax_col = axis_variance[1][1]

                row_vals = [str(v) for v in ax_row.values]
                col_vals = [str(v) for v in ax_col.values]
                grid = []
                count_grid = []
                for rv in row_vals:
                    row = []
                    crow = []
                    for cv in col_vals:
                        mask = [
                            str(r.sample_params.get(ax_row.name)) == rv
                            and str(r.sample_params.get(ax_col.name)) == cv
                            for r in self.results
                        ]
                        cell_scores = scores[mask]
                        if len(cell_scores) > 0:
                            row.append(round(float(cell_scores.mean()), 2))
                        else:
                            row.append(None)
                        crow.append(int(cell_scores.shape[0]))
                    grid.append(row)
                    count_grid.append(crow)

                oracle_data["has_interaction"] = True
                oracle_data["interaction"] = {
                    "row_axis": ax_row.name,
                    "col_axis": ax_col.name,
                    "row_values": row_vals,
                    "col_values": col_vals,
                    "means": grid,
                    "counts": count_grid,
                }
            else:
                oracle_data["has_interaction"] = False

            chart_data[oracle_name] = oracle_data

        return chart_data

    def _render_template(self, data: dict, chart_data: dict) -> str:
        """Render the Jinja2 template with data and chart data."""
        template = self.env.get_template("report.html.j2")
        return template.render(
            chart_data=chart_data,
            chartjs_source=_load_vendor_asset("chart.umd.min.js"),
            chartjs_annotation_source=_load_vendor_asset(
                "chartjs-plugin-annotation.min.js"
            ),
            chartjs_matrix_source=_load_vendor_asset("chartjs-chart-matrix.min.js"),
            **data,
        )
