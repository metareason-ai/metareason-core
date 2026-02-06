import matplotlib

matplotlib.use("Agg")

import arviz  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from metareason.config.models import AxisConfig  # noqa: E402
from metareason.reporting.visualizations import (  # noqa: E402
    figure_to_base64,
    plot_convergence_diagnostics,
    plot_oracle_variability,
    plot_parameter_space,
    plot_posterior_distribution,
    plot_score_distribution,
)


class TestFigureToBase64:
    def test_figure_to_base64(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = figure_to_base64(fig)
        assert isinstance(result, str)
        assert len(result) > 0
        # Valid base64 characters only
        import base64

        decoded = base64.b64decode(result)
        # PNG files start with specific magic bytes
        assert decoded[:4] == b"\x89PNG"


class TestPlotPosteriorDistribution:
    def test_plot_posterior_distribution(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(3.5, 0.5, size=1000)
        fig = plot_posterior_distribution(
            samples=samples,
            hdi_lower=3.0,
            hdi_upper=4.0,
            hdi_prob=0.94,
            oracle_name="coherence",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotParameterSpace:
    def test_plot_parameter_space(self):
        axes_config = [
            AxisConfig(
                name="temperature",
                type="continuous",
                distribution="uniform",
                params={"low": 0.0, "high": 1.0},
            ),
            AxisConfig(
                name="complexity",
                type="continuous",
                distribution="uniform",
                params={"low": 1.0, "high": 10.0},
            ),
        ]
        samples = [
            {"temperature": 0.2, "complexity": 3.0},
            {"temperature": 0.5, "complexity": 5.0},
            {"temperature": 0.8, "complexity": 7.0},
            {"temperature": 0.3, "complexity": 9.0},
        ]
        scores = np.array([3.0, 4.0, 2.0, 5.0])
        fig = plot_parameter_space(
            samples=samples,
            axes_config=axes_config,
            scores=scores,
            oracle_name="quality",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_parameter_space_insufficient_axes(self):
        axes_config = [
            AxisConfig(
                name="tone",
                type="categorical",
                values=["formal", "casual"],
                weights=[0.5, 0.5],
            ),
        ]
        samples = [{"tone": "formal"}, {"tone": "casual"}]
        scores = np.array([3.0, 4.0])
        fig = plot_parameter_space(
            samples=samples,
            axes_config=axes_config,
            scores=scores,
            oracle_name="quality",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotScoreDistribution:
    def test_plot_score_distribution(self):
        scores = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
        fig = plot_score_distribution(scores=scores, oracle_name="accuracy")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotOracleVariability:
    def test_plot_oracle_variability(self):
        rng = np.random.default_rng(42)
        noise_samples = np.abs(rng.normal(0.5, 0.2, size=1000))
        fig = plot_oracle_variability(
            noise_samples=noise_samples,
            hdi_lower=0.2,
            hdi_upper=0.8,
            hdi_prob=0.94,
            oracle_name="coherence",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotConvergenceDiagnostics:
    def test_plot_convergence_diagnostics(self):
        rng = np.random.default_rng(42)
        idata = arviz.from_dict(
            posterior={
                "overall_quality": rng.normal(3.5, 0.5, size=(1, 100)),
                "oracle_noise": np.abs(rng.normal(0.5, 0.2, size=(1, 100))),
            }
        )
        fig = plot_convergence_diagnostics(idata=idata, oracle_name="coherence")
        assert isinstance(fig, Figure)
        plt.close(fig)
