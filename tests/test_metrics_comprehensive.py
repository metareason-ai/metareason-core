"""Comprehensive tests for sampling metrics module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from metareason.sampling.metrics import (
    compute_all_metrics,
    compute_correlation_metrics,
    compute_coverage_metrics,
    compute_discrepancy,
    compute_distance_metrics,
    compute_uniformity_metrics,
    validate_against_theoretical,
    visualize_metrics_comparison,
    visualize_sample_distribution,
)


class TestComputeDiscrepancy:
    """Test discrepancy computation."""

    def test_discrepancy_uniform_samples(self):
        """Test discrepancy for uniform samples."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (100, 2))
        discrepancy = compute_discrepancy(samples)
        assert isinstance(discrepancy, float)
        assert discrepancy >= 0


class TestCorrelationMetrics:
    """Test correlation metrics computation."""

    def test_correlation_metrics_uncorrelated(self):
        """Test correlation metrics for uncorrelated samples."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (100, 3))
        metrics = compute_correlation_metrics(samples)

        assert "max_correlation" in metrics
        assert "mean_correlation" in metrics
        assert "std_correlation" in metrics
        assert "median_correlation" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_correlation_metrics_single_dimension(self):
        """Test correlation metrics for single dimension."""
        samples = np.random.uniform(0, 1, (100, 1))
        metrics = compute_correlation_metrics(samples)

        # For single dimension, correlation matrix is scalar, so all metrics should be 0
        assert "max_correlation" in metrics
        assert "mean_correlation" in metrics
        assert "std_correlation" in metrics
        # No median_correlation for single dimension
        assert "median_correlation" not in metrics

        # All correlation metrics should be 0 for single dimension
        assert metrics["max_correlation"] == 0.0
        assert metrics["mean_correlation"] == 0.0
        assert metrics["std_correlation"] == 0.0

    def test_correlation_metrics_perfectly_correlated(self):
        """Test correlation metrics for perfectly correlated samples."""
        x = np.random.uniform(0, 1, 100)
        samples = np.column_stack([x, x, x * 2])
        metrics = compute_correlation_metrics(samples)

        assert metrics["max_correlation"] > 0.9  # Should be high correlation


class TestDistanceMetrics:
    """Test distance-based metrics computation."""

    def test_distance_metrics_small_subset(self):
        """Test distance metrics with small sample set."""
        samples = np.random.uniform(0, 1, (50, 2))
        metrics = compute_distance_metrics(samples, subset_size=1000)

        expected_keys = [
            "min_distance",
            "mean_min_distance",
            "std_min_distance",
            "median_min_distance",
            "coverage_radius",
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_distance_metrics_large_subset(self):
        """Test distance metrics with large sample set requiring subsetting."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (2000, 2))
        metrics = compute_distance_metrics(samples, subset_size=500)

        expected_keys = [
            "min_distance",
            "mean_min_distance",
            "std_min_distance",
            "median_min_distance",
            "coverage_radius",
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0

    def test_distance_metrics_no_subset_limit(self):
        """Test distance metrics without subset limit."""
        samples = np.random.uniform(0, 1, (100, 2))
        metrics = compute_distance_metrics(samples, subset_size=None)

        assert "min_distance" in metrics
        assert isinstance(metrics["min_distance"], float)


class TestUniformityMetrics:
    """Test uniformity metrics computation."""

    def test_uniformity_metrics_uniform_samples(self):
        """Test uniformity metrics for uniform samples."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (1000, 3))
        metrics = compute_uniformity_metrics(samples, n_bins=10)

        assert "mean_chi_squared" in metrics
        assert "max_chi_squared" in metrics
        assert "chi_squared_critical_95" in metrics

        # Check individual dimension metrics (first 5 dims only)
        for dim in range(min(3, 5)):
            assert f"chi_squared_dim_{dim}" in metrics

    def test_uniformity_metrics_many_dimensions(self):
        """Test uniformity metrics with many dimensions."""
        samples = np.random.uniform(0, 1, (500, 10))
        metrics = compute_uniformity_metrics(samples, n_bins=5)

        # Should only have metrics for first 5 dimensions
        dim_keys = [k for k in metrics.keys() if k.startswith("chi_squared_dim_")]
        assert len(dim_keys) == 5

        for dim in range(5):
            assert f"chi_squared_dim_{dim}" in metrics


class TestCoverageMetrics:
    """Test space coverage metrics computation."""

    def test_coverage_metrics_normal_case(self):
        """Test coverage metrics for normal case."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (100, 2))
        metrics = compute_coverage_metrics(samples)

        assert "coverage_ratio" in metrics
        assert 0 <= metrics["coverage_ratio"] <= 1

        # Check individual dimension coverage
        for dim in range(min(2, 3)):
            key = f"coverage_dim_{dim}"
            assert key in metrics
            assert 0 <= metrics[key] <= 1

    def test_coverage_metrics_high_dimensions(self):
        """Test coverage metrics with high dimensions to trigger grid size limit."""
        samples = np.random.uniform(0, 1, (100, 20))  # High dimension count
        metrics = compute_coverage_metrics(samples)

        assert "coverage_ratio" in metrics
        assert isinstance(metrics["coverage_ratio"], float)

        # Should have coverage for first 3 dimensions only
        coverage_keys = [k for k in metrics.keys() if k.startswith("coverage_dim_")]
        assert len(coverage_keys) == 3

    def test_coverage_metrics_large_grid_adjustment(self):
        """Test coverage metrics when grid size needs adjustment."""
        # Create scenario where grid_size**n_dims > 1e6
        samples = np.random.uniform(0, 1, (1000, 10))
        metrics = compute_coverage_metrics(samples)

        assert "coverage_ratio" in metrics
        assert isinstance(metrics["coverage_ratio"], float)


class TestValidateAgainstTheoretical:
    """Test theoretical validation."""

    def test_validate_uniform_distribution(self):
        """Test validation against uniform distribution."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (1000, 3))
        metrics = validate_against_theoretical(samples, "uniform")

        # Check KS test metrics for first 5 dimensions
        for dim in range(min(3, 5)):
            assert f"ks_test_dim_{dim}_stat" in metrics
            assert f"ks_test_dim_{dim}_pvalue" in metrics
            assert f"mean_error_dim_{dim}" in metrics
            assert f"std_error_dim_{dim}" in metrics

    def test_validate_uniform_many_dimensions(self):
        """Test validation with many dimensions."""
        samples = np.random.uniform(0, 1, (500, 10))
        metrics = validate_against_theoretical(samples, "uniform")

        # Should only test first 5 dimensions
        ks_keys = [k for k in metrics.keys() if "ks_test_dim_" in k]
        mean_keys = [k for k in metrics.keys() if "mean_error_dim_" in k]

        assert len([k for k in ks_keys if "_stat" in k]) == 5
        assert len([k for k in ks_keys if "_pvalue" in k]) == 5
        assert len(mean_keys) == 5

    def test_validate_non_uniform_distribution(self):
        """Test validation with non-uniform distribution type."""
        samples = np.random.uniform(0, 1, (100, 2))
        metrics = validate_against_theoretical(samples, "normal")

        # Should return empty dict for non-uniform distributions
        assert metrics == {}


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""

    def test_compute_all_metrics_unit_hypercube(self):
        """Test computing all metrics for unit hypercube samples."""
        np.random.seed(42)
        samples = np.random.uniform(0, 1, (100, 2))
        metrics = compute_all_metrics(samples, unit_hypercube=True)

        # Should include all metric types
        assert "discrepancy" in metrics
        assert "max_correlation" in metrics
        assert "min_distance" in metrics
        assert "coverage_ratio" in metrics
        assert "mean_chi_squared" in metrics

        # Validate against theoretical should be included
        assert any("ks_test" in k for k in metrics.keys())

    def test_compute_all_metrics_not_unit_hypercube(self):
        """Test computing all metrics for non-unit hypercube samples."""
        samples = np.random.normal(0, 1, (100, 2))
        metrics = compute_all_metrics(samples, unit_hypercube=False)

        # Should not include unit hypercube specific metrics
        assert "discrepancy" not in metrics
        assert "coverage_ratio" not in metrics
        assert "mean_chi_squared" not in metrics

        # But should include general metrics
        assert "max_correlation" in metrics
        assert "min_distance" in metrics


class TestVisualizeSampleDistribution:
    """Test sample distribution visualization."""

    def test_visualize_1d_samples(self):
        """Test visualization of 1D samples."""
        samples = np.random.uniform(0, 1, (100, 1))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_1d.png"

            # Should not raise any errors
            visualize_sample_distribution(samples, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_2d_samples(self):
        """Test visualization of 2D samples."""
        samples = np.random.uniform(0, 1, (100, 2))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_2d.png"

            visualize_sample_distribution(samples, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_high_dimensional_samples(self):
        """Test visualization of high dimensional samples."""
        samples = np.random.uniform(0, 1, (100, 5))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_5d.png"

            visualize_sample_distribution(samples, max_dims=3, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_samples_without_save(self):
        """Test visualization without saving."""
        samples = np.random.uniform(0, 1, (50, 2))

        with patch("matplotlib.pyplot.show") as mock_show:
            visualize_sample_distribution(samples)
            mock_show.assert_called_once()

    def test_visualize_many_dimensions_subplot_handling(self):
        """Test visualization with many dimensions that require subplot management."""
        samples = np.random.uniform(0, 1, (100, 6))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_6d.png"

            # Test with max_dims to ensure subplot deletion is covered
            visualize_sample_distribution(samples, max_dims=6, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_edge_case_subplot_overflow(self):
        """Test visualization where plot_idx exceeds number of axes."""
        # Create a scenario where we have fewer subplots than expected pairs
        # This can happen when n_plots calculation results in more pairs than grid allows
        samples = np.random.uniform(0, 1, (50, 4))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_edge_case.png"

            # This should trigger the condition where plot_idx < len(axes) might be false
            visualize_sample_distribution(samples, max_dims=4, save_path=str(save_path))
            assert save_path.exists()


class TestVisualizeMetricsComparison:
    """Test metrics comparison visualization."""

    def test_visualize_metrics_comparison_empty_dict(self):
        """Test visualization with empty metrics dictionary."""
        metrics_dict = {}

        # Should return early without error
        visualize_metrics_comparison(metrics_dict)

    def test_visualize_metrics_comparison_normal_case(self):
        """Test visualization with normal metrics dictionary."""
        metrics_dict = {
            "strategy1": {
                "discrepancy": 0.1,
                "max_correlation": 0.2,
                "mean_min_distance": 0.05,
                "coverage_ratio": 0.8,
                "mean_chi_squared": 15.0,
            },
            "strategy2": {
                "discrepancy": 0.15,
                "max_correlation": 0.18,
                "mean_min_distance": 0.04,
                "coverage_ratio": 0.75,
                "mean_chi_squared": 18.0,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "metrics_comparison.png"

            visualize_metrics_comparison(metrics_dict, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_metrics_comparison_without_save(self):
        """Test visualization without saving."""
        metrics_dict = {
            "strategy1": {"metric1": 0.5, "metric2": 0.3},
            "strategy2": {"metric1": 0.6, "metric2": 0.4},
        }

        with patch("matplotlib.pyplot.show") as mock_show:
            visualize_metrics_comparison(metrics_dict)
            mock_show.assert_called_once()

    def test_visualize_metrics_comparison_no_standard_metrics(self):
        """Test visualization when standard metrics are not available."""
        metrics_dict = {
            "strategy1": {"custom_metric1": 1.0, "custom_metric2": 2.0},
            "strategy2": {"custom_metric1": 1.5, "custom_metric2": 2.5},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "custom_metrics.png"

            # Should fall back to first 5 metrics available
            visualize_metrics_comparison(metrics_dict, save_path=str(save_path))
            assert save_path.exists()

    def test_visualize_metrics_comparison_single_metric(self):
        """Test visualization with single metric."""
        metrics_dict = {
            "strategy1": {"single_metric": 0.5},
            "strategy2": {"single_metric": 0.7},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "single_metric.png"

            visualize_metrics_comparison(metrics_dict, save_path=str(save_path))
            assert save_path.exists()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_small_sample_sizes(self):
        """Test metrics with very small sample sizes."""
        samples = np.random.uniform(0, 1, (5, 2))

        # Should handle small samples gracefully
        metrics = compute_all_metrics(samples)
        assert all(isinstance(v, float) for v in metrics.values())

    def test_single_sample(self):
        """Test metrics with single sample."""
        samples = np.array([[0.5, 0.5]])

        # Suppress expected warnings for single sample edge case
        with np.errstate(invalid="ignore"):
            # Distance metrics might have special behavior with single sample
            distance_metrics = compute_distance_metrics(samples)
            assert all(isinstance(v, float) for v in distance_metrics.values())

    def test_identical_samples(self):
        """Test metrics with identical samples."""
        samples = np.ones((10, 2)) * 0.5

        # Suppress expected warnings for identical samples (zero variance/correlation)
        with np.errstate(invalid="ignore", divide="ignore"):
            # Should handle identical samples without errors
            metrics = compute_correlation_metrics(samples)
            assert all(isinstance(v, float) for v in metrics.values())

            distance_metrics = compute_distance_metrics(samples)
            # Min distance should be 0 for identical samples
            assert distance_metrics["min_distance"] == 0.0

    def test_extreme_values(self):
        """Test metrics with extreme values."""
        # Samples at boundaries
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

        metrics = compute_all_metrics(samples)
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(not np.isnan(v) for v in metrics.values())
