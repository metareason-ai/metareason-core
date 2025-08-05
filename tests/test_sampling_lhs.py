"""Tests for Latin Hypercube Sampling implementation."""

import numpy as np
import pytest
from scipy.stats import kstest, uniform

from metareason.config.axes import CategoricalAxis, ContinuousAxis
from metareason.sampling import (
    LatinHypercubeSampler,
    benchmark_optimizers,
    compute_all_metrics,
    compute_correlation_metrics,
    compute_discrepancy,
    compute_distance_metrics,
    parallel_sample_generation,
    stratified_sampling,
)
from metareason.sampling.optimization import (
    CorrelationMinimizer,
    ESIOptimizer,
    MaximinOptimizer,
)


class TestLatinHypercubeSampler:
    """Test suite for LatinHypercubeSampler."""

    @pytest.fixture
    def continuous_axes(self):
        """Create continuous axis configurations."""
        return {
            "temperature": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "top_p": ContinuousAxis(type="uniform", min=0.5, max=1.0),
            "frequency_penalty": ContinuousAxis(
                type="truncated_normal", mu=0.0, sigma=0.5, min=-2.0, max=2.0
            ),
        }

    @pytest.fixture
    def categorical_axes(self):
        """Create categorical axis configurations."""
        return {
            "model": CategoricalAxis(
                type="categorical", values=["gpt-3.5", "gpt-4", "claude"]
            ),
            "prompt_style": CategoricalAxis(
                type="categorical", values=["formal", "casual", "technical"]
            ),
        }

    @pytest.fixture
    def mixed_axes(self, continuous_axes, categorical_axes):
        """Create mixed continuous and categorical axes."""
        return {**continuous_axes, **categorical_axes}

    def test_basic_lhs_generation(self, continuous_axes):
        """Test basic LHS sample generation."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=100, random_seed=42
        )
        result = sampler.sample()

        assert result.samples.shape == (100, 3)
        assert result.metadata["n_samples"] == 100
        assert result.metadata["n_dimensions"] == 3
        assert result.metadata["n_continuous"] == 3
        assert result.metadata["n_categorical"] == 0

    def test_unit_hypercube_samples(self, continuous_axes):
        """Test that raw samples are in unit hypercube."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=200, random_seed=42
        )
        unit_samples = sampler.generate_samples()

        assert unit_samples.shape == (200, 3)
        assert np.all(unit_samples >= 0)
        assert np.all(unit_samples <= 1)

    def test_transformed_samples(self, continuous_axes):
        """Test that transformed samples respect axis bounds."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=100, random_seed=42
        )
        unit_samples = sampler.generate_samples()
        transformed = sampler.transform_samples(unit_samples)

        assert np.all(transformed["temperature"] >= 0.0)
        assert np.all(transformed["temperature"] <= 1.0)

        assert np.all(transformed["top_p"] >= 0.5)
        assert np.all(transformed["top_p"] <= 1.0)

        assert np.all(transformed["frequency_penalty"] >= -2.0)
        assert np.all(transformed["frequency_penalty"] <= 2.0)

    def test_categorical_handling(self, categorical_axes):
        """Test handling of categorical axes."""
        sampler = LatinHypercubeSampler(
            axes=categorical_axes, n_samples=90, random_seed=42
        )
        result = sampler.sample()

        assert result.samples.shape == (90, 2)

        models = result.samples[:, 0]
        assert all(m in ["gpt-3.5", "gpt-4", "claude"] for m in models)

        styles = result.samples[:, 1]
        assert all(s in ["formal", "casual", "technical"] for s in styles)

    def test_mixed_axes(self, mixed_axes):
        """Test handling of mixed continuous and categorical axes."""
        sampler = LatinHypercubeSampler(axes=mixed_axes, n_samples=150, random_seed=42)
        result = sampler.sample()

        assert result.samples.shape == (150, 5)
        assert result.metadata["n_continuous"] == 3
        assert result.metadata["n_categorical"] == 2

    def test_optimization_maximin(self, continuous_axes):
        """Test maximin optimization."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes,
            n_samples=50,
            random_seed=42,
            optimization="maximin",
        )
        result = sampler.sample()

        assert result.quality_metrics is not None
        assert "min_distance" in result.quality_metrics
        assert result.quality_metrics["min_distance"] > 0

    def test_optimization_correlation(self, continuous_axes):
        """Test correlation minimization."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes,
            n_samples=100,
            random_seed=42,
            optimization="correlation",
        )
        result = sampler.sample()

        assert result.quality_metrics is not None
        assert "max_correlation" in result.quality_metrics
        assert result.quality_metrics["max_correlation"] < 0.5

    def test_optimization_esi(self, continuous_axes):
        """Test ESI optimization."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes,
            n_samples=50,
            random_seed=42,
            optimization="esi",
        )
        result = sampler.sample()

        assert result.quality_metrics is not None
        assert "mean_min_distance" in result.quality_metrics

    def test_strength_parameter(self, continuous_axes):
        """Test LHS strength parameter."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes,
            n_samples=100,
            random_seed=42,
            strength=1,
        )
        result = sampler.sample()
        assert result.samples.shape == (100, 3)

    def test_strength_2_validation(self, continuous_axes):
        """Test strength=2 validation."""
        with pytest.raises(ValueError, match="must be prime"):
            LatinHypercubeSampler(
                axes=continuous_axes,
                n_samples=100,
                strength=2,
            )

        sampler = LatinHypercubeSampler(
            axes={"x": continuous_axes["temperature"]},
            n_samples=49,
            strength=2,
        )
        result = sampler.sample()
        assert result.samples.shape == (49, 1)

    def test_batch_generation(self, continuous_axes):
        """Test batch generation for large samples."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes,
            n_samples=5000,
            random_seed=42,
            batch_size=1000,
            show_progress=False,
        )
        result = sampler.sample()

        assert result.samples.shape == (5000, 3)

    def test_reproducibility(self, continuous_axes):
        """Test that same seed produces same results."""
        sampler1 = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=100, random_seed=42
        )
        result1 = sampler1.sample()

        sampler2 = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=100, random_seed=42
        )
        result2 = sampler2.sample()

        np.testing.assert_array_equal(result1.samples, result2.samples)

    def test_quality_metrics(self, continuous_axes):
        """Test quality metrics computation."""
        sampler = LatinHypercubeSampler(
            axes=continuous_axes, n_samples=100, random_seed=42
        )
        result = sampler.sample()

        assert result.quality_metrics is not None
        assert "discrepancy" in result.quality_metrics
        assert "max_correlation" in result.quality_metrics
        assert "mean_correlation" in result.quality_metrics
        assert "min_distance" in result.quality_metrics
        assert "mean_min_distance" in result.quality_metrics

        for i in range(3):
            assert f"uniformity_dim_{i}" in result.quality_metrics


class TestOptimizers:
    """Test suite for optimization strategies."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.rand(100, 3)

    def test_maximin_optimizer(self, sample_data):
        """Test maximin optimizer."""
        optimizer = MaximinOptimizer(random_seed=42)
        optimized = optimizer.optimize(sample_data, n_iterations=10)

        initial_criterion = optimizer.compute_criterion(sample_data)
        final_criterion = optimizer.compute_criterion(optimized)

        assert optimized.shape == sample_data.shape
        assert final_criterion >= initial_criterion

    def test_correlation_minimizer(self, sample_data):
        """Test correlation minimizer."""
        optimizer = CorrelationMinimizer(random_seed=42)
        optimized = optimizer.optimize(sample_data, n_iterations=10)

        initial_criterion = optimizer.compute_criterion(sample_data)
        final_criterion = optimizer.compute_criterion(optimized)

        assert optimized.shape == sample_data.shape
        assert final_criterion <= initial_criterion

    def test_esi_optimizer(self, sample_data):
        """Test ESI optimizer."""
        optimizer = ESIOptimizer(random_seed=42)
        optimized = optimizer.optimize(sample_data, n_iterations=10)

        assert optimized.shape == sample_data.shape

    def test_benchmark_optimizers(self, sample_data):
        """Test optimizer benchmarking."""
        optimizers = {
            "maximin": MaximinOptimizer(random_seed=42),
            "correlation": CorrelationMinimizer(random_seed=42),
            "esi": ESIOptimizer(random_seed=42),
        }

        results = benchmark_optimizers(
            sample_data, optimizers, n_iterations=5, show_progress=False
        )

        assert len(results) == 3
        for name, metrics in results.items():
            assert "initial_criterion" in metrics
            assert "final_criterion" in metrics
            assert "improvement" in metrics
            assert "time_seconds" in metrics


class TestQualityMetrics:
    """Test suite for quality metrics."""

    @pytest.fixture
    def uniform_samples(self):
        """Generate uniform samples."""
        np.random.seed(42)
        return np.random.rand(200, 4)

    def test_discrepancy_metric(self, uniform_samples):
        """Test discrepancy computation."""
        discrepancy = compute_discrepancy(uniform_samples)

        assert isinstance(discrepancy, float)
        assert 0 <= discrepancy <= 1

    def test_correlation_metrics(self, uniform_samples):
        """Test correlation metrics."""
        metrics = compute_correlation_metrics(uniform_samples)

        assert "max_correlation" in metrics
        assert "mean_correlation" in metrics
        assert "std_correlation" in metrics
        assert "median_correlation" in metrics

        assert 0 <= metrics["max_correlation"] <= 1
        assert 0 <= metrics["mean_correlation"] <= 1

    def test_distance_metrics(self, uniform_samples):
        """Test distance metrics."""
        metrics = compute_distance_metrics(uniform_samples, subset_size=100)

        assert "min_distance" in metrics
        assert "mean_min_distance" in metrics
        assert "std_min_distance" in metrics
        assert "median_min_distance" in metrics
        assert "coverage_radius" in metrics

        assert metrics["min_distance"] > 0
        assert metrics["mean_min_distance"] > metrics["min_distance"]

    def test_all_metrics(self, uniform_samples):
        """Test comprehensive metrics computation."""
        metrics = compute_all_metrics(uniform_samples, unit_hypercube=True)

        assert "discrepancy" in metrics
        assert "max_correlation" in metrics
        assert "min_distance" in metrics
        assert "coverage_ratio" in metrics
        assert "mean_chi_squared" in metrics


class TestSamplingUtilities:
    """Test suite for sampling utilities."""

    @pytest.fixture
    def axes_config(self):
        """Create axis configuration for testing."""
        return {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "y": ContinuousAxis(type="uniform", min=-1.0, max=1.0),
            "category": CategoricalAxis(type="categorical", values=["A", "B", "C"]),
        }

    def test_stratified_sampling(self, axes_config):
        """Test stratified sampling."""
        samples = stratified_sampling(
            LatinHypercubeSampler,
            {"axes": axes_config, "n_samples": 90, "random_seed": 42},
            stratify_by=["category"],
            ensure_balance=True,
        )

        assert samples.shape == (90, 3)

        categories, counts = np.unique(samples[:, 2], return_counts=True)
        assert len(categories) == 3
        assert all(count == 30 for count in counts)

    def test_parallel_generation(self):
        """Test parallel sample generation."""
        pytest.skip("Parallel generation requires multiprocessing setup")
        axes = {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "y": ContinuousAxis(type="uniform", min=0.0, max=1.0),
        }

        samples = parallel_sample_generation(
            LatinHypercubeSampler,
            {"axes": axes, "n_samples": 200, "random_seed": 42},
            n_batches=4,
            n_workers=2,
            show_progress=False,
        )

        assert samples.shape == (200, 2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_dimension(self):
        """Test sampling with single dimension."""
        axes = {"x": ContinuousAxis(type="uniform", min=0.0, max=1.0)}
        sampler = LatinHypercubeSampler(axes=axes, n_samples=50)
        result = sampler.sample()

        assert result.samples.shape == (50, 1)

    def test_large_dimensions(self):
        """Test sampling with many dimensions."""
        axes = {
            f"dim_{i}": ContinuousAxis(type="uniform", min=0.0, max=1.0)
            for i in range(20)
        }
        sampler = LatinHypercubeSampler(axes=axes, n_samples=100, show_progress=False)
        result = sampler.sample()

        assert result.samples.shape == (100, 20)

    def test_small_sample_size(self):
        """Test with very small sample size."""
        axes = {"x": ContinuousAxis(type="uniform", min=0.0, max=1.0)}
        sampler = LatinHypercubeSampler(axes=axes, n_samples=2)
        result = sampler.sample()

        assert result.samples.shape == (2, 1)

    def test_uniformity_validation(self):
        """Test uniformity of LHS samples."""
        axes = {"x": ContinuousAxis(type="uniform", min=0.0, max=1.0)}
        sampler = LatinHypercubeSampler(axes=axes, n_samples=1000, random_seed=42)
        unit_samples = sampler.generate_samples()

        stat, p_value = kstest(unit_samples[:, 0], uniform(0, 1).cdf)
        assert p_value > 0.01
