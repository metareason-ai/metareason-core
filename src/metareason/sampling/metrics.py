"""Quality metrics for evaluating sampling distributions."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def compute_discrepancy(samples: np.ndarray) -> float:
    """Compute the discrepancy of samples.

    Discrepancy measures how uniformly distributed the samples are.
    Lower values indicate better uniformity.

    Args:
        samples: Samples in unit hypercube [0,1]^d

    Returns:
        Discrepancy value
    """
    return float(qmc.discrepancy(samples))


def compute_correlation_metrics(samples: np.ndarray) -> Dict[str, float]:
    """Compute correlation metrics between dimensions.

    Args:
        samples: Samples array

    Returns:
        Dictionary with correlation metrics
    """
    n_dims = samples.shape[1]

    if n_dims == 1:
        # For single dimension, no correlation exists
        return {
            "max_correlation": 0.0,
            "mean_correlation": 0.0,
            "std_correlation": 0.0,
        }

    corr_matrix = np.corrcoef(samples.T)
    np.fill_diagonal(corr_matrix, 0)

    metrics = {
        "max_correlation": float(np.max(np.abs(corr_matrix))),
        "mean_correlation": float(np.mean(np.abs(corr_matrix))),
        "std_correlation": float(np.std(np.abs(corr_matrix))),
    }

    upper_tri = corr_matrix[np.triu_indices(n_dims, k=1)]
    metrics["median_correlation"] = float(np.median(np.abs(upper_tri)))

    return metrics


def compute_distance_metrics(
    samples: np.ndarray, subset_size: Optional[int] = 1000
) -> Dict[str, float]:
    """Compute distance-based metrics.

    Args:
        samples: Samples array
        subset_size: Maximum subset size for efficiency

    Returns:
        Dictionary with distance metrics
    """
    n_samples = samples.shape[0]

    if subset_size and n_samples > subset_size:
        indices = np.random.choice(n_samples, size=subset_size, replace=False)
        samples_subset = samples[indices]
    else:
        samples_subset = samples

    distances = cdist(samples_subset, samples_subset)
    np.fill_diagonal(distances, np.inf)

    min_distances = np.min(distances, axis=1)

    metrics = {
        "min_distance": float(np.min(min_distances)),
        "mean_min_distance": float(np.mean(min_distances)),
        "std_min_distance": float(np.std(min_distances)),
        "median_min_distance": float(np.median(min_distances)),
    }

    coverage_radius = np.max(min_distances)
    metrics["coverage_radius"] = float(coverage_radius)

    return metrics


def compute_uniformity_metrics(
    samples: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Compute uniformity metrics for each dimension.

    Args:
        samples: Samples in unit hypercube
        n_bins: Number of bins for histogram

    Returns:
        Dictionary with uniformity metrics
    """
    metrics = {}
    n_samples, n_dims = samples.shape
    expected_count = n_samples / n_bins

    chi_squared_values = []

    for dim in range(n_dims):
        hist, _ = np.histogram(samples[:, dim], bins=n_bins, range=(0, 1))
        chi_squared = np.sum((hist - expected_count) ** 2 / expected_count)
        chi_squared_values.append(chi_squared)

        if dim < 5:
            metrics[f"chi_squared_dim_{dim}"] = float(chi_squared)

    metrics["mean_chi_squared"] = float(np.mean(chi_squared_values))
    metrics["max_chi_squared"] = float(np.max(chi_squared_values))

    chi_squared_critical = stats.chi2.ppf(0.95, df=n_bins - 1)
    metrics["chi_squared_critical_95"] = float(chi_squared_critical)

    return metrics


def compute_coverage_metrics(samples: np.ndarray) -> Dict[str, float]:
    """Compute space coverage metrics.

    Args:
        samples: Samples in unit hypercube

    Returns:
        Dictionary with coverage metrics
    """
    n_samples, n_dims = samples.shape
    metrics = {}

    grid_size = int(np.ceil(n_samples ** (1 / n_dims)))
    if grid_size**n_dims > 1e6:
        grid_size = int(np.ceil(1e6 ** (1 / n_dims)))

    grid_indices = np.floor(samples * grid_size).astype(int)
    grid_indices = np.clip(grid_indices, 0, grid_size - 1)

    unique_cells = len(np.unique(grid_indices.dot(grid_size ** np.arange(n_dims))))
    total_cells = min(grid_size**n_dims, n_samples)

    metrics["coverage_ratio"] = float(unique_cells / total_cells)

    for dim in range(min(n_dims, 3)):
        unique_in_dim = len(np.unique(grid_indices[:, dim]))
        metrics[f"coverage_dim_{dim}"] = float(unique_in_dim / grid_size)

    return metrics


def validate_against_theoretical(
    samples: np.ndarray, distribution_type: str = "uniform"
) -> Dict[str, float]:
    """Validate samples against theoretical expectations.

    Args:
        samples: Samples to validate
        distribution_type: Expected distribution type

    Returns:
        Dictionary with validation metrics
    """
    metrics = {}
    n_samples, n_dims = samples.shape

    if distribution_type == "uniform":
        for dim in range(min(n_dims, 5)):
            ks_stat, p_value = stats.kstest(samples[:, dim], stats.uniform().cdf)
            metrics[f"ks_test_dim_{dim}_stat"] = float(ks_stat)
            metrics[f"ks_test_dim_{dim}_pvalue"] = float(p_value)

        expected_mean = 0.5
        expected_std = 1 / np.sqrt(12)

        for dim in range(min(n_dims, 5)):
            actual_mean = np.mean(samples[:, dim])
            actual_std = np.std(samples[:, dim])

            metrics[f"mean_error_dim_{dim}"] = float(abs(actual_mean - expected_mean))
            metrics[f"std_error_dim_{dim}"] = float(abs(actual_std - expected_std))

    return metrics


def compute_all_metrics(
    samples: np.ndarray, unit_hypercube: bool = True
) -> Dict[str, float]:
    """Compute all quality metrics for samples.

    Args:
        samples: Samples array
        unit_hypercube: Whether samples are in [0,1]^d

    Returns:
        Comprehensive dictionary of metrics
    """
    all_metrics = {}

    if unit_hypercube:
        all_metrics["discrepancy"] = compute_discrepancy(samples)
        all_metrics.update(compute_uniformity_metrics(samples))
        all_metrics.update(compute_coverage_metrics(samples))
        all_metrics.update(validate_against_theoretical(samples))

    all_metrics.update(compute_correlation_metrics(samples))
    all_metrics.update(compute_distance_metrics(samples))

    return all_metrics


def visualize_sample_distribution(
    samples: np.ndarray,
    max_dims: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """Visualize the sample distribution.

    Args:
        samples: Samples to visualize
        max_dims: Maximum dimensions to plot
        save_path: Optional path to save the figure
    """
    n_samples, n_dims = samples.shape
    dims_to_plot = min(n_dims, max_dims)

    if dims_to_plot == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(samples[:, 0], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("1D Sample Distribution")

    elif dims_to_plot == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
        axes[0].set_xlabel("Dimension 0")
        axes[0].set_ylabel("Dimension 1")
        axes[0].set_title("2D Scatter Plot")
        axes[0].grid(True, alpha=0.3)

        axes[1].hexbin(samples[:, 0], samples[:, 1], gridsize=20, cmap="YlOrRd")
        axes[1].set_xlabel("Dimension 0")
        axes[1].set_ylabel("Dimension 1")
        axes[1].set_title("2D Density Plot")

    else:
        n_plots = dims_to_plot * (dims_to_plot - 1) // 2
        n_rows = int(np.ceil(np.sqrt(n_plots)))
        n_cols = int(np.ceil(n_plots / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        plot_idx = 0
        for i in range(dims_to_plot):
            for j in range(i + 1, dims_to_plot):
                if plot_idx < len(axes):
                    axes[plot_idx].scatter(samples[:, i], samples[:, j], alpha=0.3, s=5)
                    axes[plot_idx].set_xlabel(f"Dim {i}")
                    axes[plot_idx].set_ylabel(f"Dim {j}")
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])

    plt.suptitle(f"Sample Distribution ({n_samples} samples, {n_dims} dimensions)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def visualize_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Compare metrics across different sampling strategies.

    Args:
        metrics_dict: Dictionary mapping strategy names to metrics
        save_path: Optional path to save the figure
    """
    if not metrics_dict:
        return

    metric_names = list(next(iter(metrics_dict.values())).keys())
    strategy_names = list(metrics_dict.keys())

    selected_metrics = [
        "discrepancy",
        "max_correlation",
        "mean_min_distance",
        "coverage_ratio",
        "mean_chi_squared",
    ]

    available_metrics = [m for m in selected_metrics if m in metric_names]

    if not available_metrics:
        available_metrics = metric_names[:5]

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3, 4))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        values = [metrics_dict[strategy][metric] for strategy in strategy_names]

        axes[idx].bar(strategy_names, values)
        axes[idx].set_title(metric.replace("_", " ").title())
        axes[idx].set_xlabel("Strategy")
        axes[idx].tick_params(axis="x", rotation=45)

    plt.suptitle("Sampling Quality Metrics Comparison")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
