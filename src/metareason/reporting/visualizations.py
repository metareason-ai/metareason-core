import base64
import io
import itertools
from typing import List

import matplotlib

matplotlib.use("Agg")

import arviz  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from metareason.config.models import AxisConfig  # noqa: E402


def figure_to_base64(fig: Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Args:
        fig: The matplotlib Figure to convert.

    Returns:
        Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def plot_posterior_distribution(
    samples: np.ndarray,
    hdi_lower: float,
    hdi_upper: float,
    hdi_prob: float,
    oracle_name: str,
) -> Figure:
    """Plot the posterior distribution of overall quality with HDI region.

    Args:
        samples: 1D array of posterior samples for overall_quality.
        hdi_lower: Lower bound of the HDI.
        hdi_upper: Upper bound of the HDI.
        hdi_prob: Probability mass of the HDI (e.g. 0.94).
        oracle_name: Name of the oracle for the title.

    Returns:
        Matplotlib Figure with the posterior density plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    kde = gaussian_kde(samples)
    x = np.linspace(samples.min() - 0.5, samples.max() + 0.5, 500)
    density = kde(x)

    ax.plot(x, density, color="steelblue", linewidth=2)

    hdi_mask = (x >= hdi_lower) & (x <= hdi_upper)
    ax.fill_between(
        x,
        density,
        where=hdi_mask,
        alpha=0.3,
        color="steelblue",
        label=f"{int(hdi_prob * 100)}% HDI [{hdi_lower:.2f}, {hdi_upper:.2f}]",
    )

    mean_val = np.mean(samples)
    median_val = np.median(samples)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_val:.2f}",
    )

    ax.set_title(
        f"Estimated True Quality ({oracle_name}): "
        f"{int(hdi_prob * 100)}% HDI [{hdi_lower:.2f}, {hdi_upper:.2f}]"
    )
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_parameter_space(
    samples: List[dict],
    axes_config: List[AxisConfig],
    scores: np.ndarray,
    oracle_name: str,
) -> Figure:
    """Plot 2D scatter plots of parameter space colored by score.

    Args:
        samples: List of parameter dictionaries from sampling.
        axes_config: List of AxisConfig defining the parameter axes.
        scores: 1D array of oracle scores corresponding to each sample.
        oracle_name: Name of the oracle for the title.

    Returns:
        Matplotlib Figure with parameter space scatter plots.
    """
    continuous_axes = [a for a in axes_config if a.type == "continuous"]

    if len(continuous_axes) < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "Insufficient continuous axes for parameter space plot",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.suptitle(f"Parameter Space Coverage ({oracle_name})")
        return fig

    pairs = list(itertools.combinations(continuous_axes, 2))
    n_pairs = len(pairs)
    ncols = min(n_pairs, 3)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    for idx, (ax_x, ax_y) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        x_vals = [s[ax_x.name] for s in samples]
        y_vals = [s[ax_y.name] for s in samples]
        scatter = ax.scatter(
            x_vals,
            y_vals,
            c=scores,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
        )
        ax.set_xlabel(ax_x.name)
        ax.set_ylabel(ax_y.name)
        fig.colorbar(scatter, ax=ax, label="Score")

    # Hide unused subplots
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Parameter Space Coverage ({oracle_name})", fontsize=14)
    fig.tight_layout()
    return fig


def plot_score_distribution(scores: np.ndarray, oracle_name: str) -> Figure:
    """Plot histogram of observed oracle scores.

    Args:
        scores: 1D array of oracle scores.
        oracle_name: Name of the oracle for the title.

    Returns:
        Matplotlib Figure with the score histogram.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.arange(0.5, 6.5, 1.0)
    ax.hist(scores, bins=bins, edgecolor="black", color="steelblue", alpha=0.7)

    mean_val = np.mean(scores)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution: {oracle_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_oracle_variability(
    noise_samples: np.ndarray,
    hdi_lower: float,
    hdi_upper: float,
    hdi_prob: float,
    oracle_name: str,
) -> Figure:
    """Plot the distribution of oracle measurement noise with HDI.

    Args:
        noise_samples: 1D array of posterior noise samples.
        hdi_lower: Lower bound of the noise HDI.
        hdi_upper: Upper bound of the noise HDI.
        hdi_prob: Probability mass of the HDI.
        oracle_name: Name of the oracle for the title.

    Returns:
        Matplotlib Figure with the noise density plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    kde = gaussian_kde(noise_samples)
    x = np.linspace(noise_samples.min() - 0.5, noise_samples.max() + 0.5, 500)
    density = kde(x)

    ax.plot(x, density, color="steelblue", linewidth=2)

    hdi_mask = (x >= hdi_lower) & (x <= hdi_upper)
    ax.fill_between(
        x,
        density,
        where=hdi_mask,
        alpha=0.3,
        color="steelblue",
        label=f"{int(hdi_prob * 100)}% HDI [{hdi_lower:.2f}, {hdi_upper:.2f}]",
    )

    mean_val = np.mean(noise_samples)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )

    ax.set_title(f"Judge Measurement Noise: {oracle_name}")
    ax.set_xlabel("Noise Magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_convergence_diagnostics(idata, oracle_name: str) -> Figure:
    """Plot MCMC convergence trace plots for quality and noise parameters.

    Args:
        idata: ArviZ InferenceData object with posterior samples.
        oracle_name: Name of the oracle for the title.

    Returns:
        Matplotlib Figure with trace plots.
    """
    axes_array = arviz.plot_trace(idata, var_names=["overall_quality", "oracle_noise"])
    fig = axes_array.ravel()[0].figure
    fig.suptitle(f"Convergence Diagnostics: {oracle_name}", fontsize=14)
    fig.tight_layout()
    return fig
