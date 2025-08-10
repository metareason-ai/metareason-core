"""Tests for the sampling factory module."""

import pytest
from pydantic import ValidationError

from metareason.config.axes import CategoricalAxis, ContinuousAxis
from metareason.config.sampling import SamplingConfig
from metareason.sampling.factory import create_sampler
from metareason.sampling.lhs import LatinHypercubeSampler


class TestCreateSampler:
    """Test suite for the create_sampler factory function."""

    @pytest.fixture
    def simple_axes(self):
        """Create simple axis configurations for testing."""
        return {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "y": ContinuousAxis(type="uniform", min=-1.0, max=1.0),
        }

    @pytest.fixture
    def mixed_axes(self):
        """Create mixed axis configurations for testing."""
        return {
            "temperature": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "model": CategoricalAxis(type="categorical", values=["gpt-3.5", "gpt-4"]),
        }

    def test_create_sampler_with_defaults(self, simple_axes):
        """Test creating a sampler with default configuration."""
        sampler = create_sampler(axes=simple_axes)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_samples == 1000
        assert sampler.random_seed == 42  # Default from SamplingConfig
        assert sampler.optimization == "maximin"
        assert sampler.scramble is True
        assert sampler.strength == 1

    def test_create_sampler_with_custom_samples(self, simple_axes):
        """Test creating a sampler with custom sample count."""
        sampler = create_sampler(axes=simple_axes, n_samples=500)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_samples == 500

    def test_create_sampler_with_custom_seed(self, simple_axes):
        """Test creating a sampler with custom random seed."""
        sampler = create_sampler(axes=simple_axes, random_seed=123)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.random_seed == 123

    def test_create_sampler_with_none_config(self, simple_axes):
        """Test creating a sampler with None configuration (uses defaults)."""
        sampler = create_sampler(
            axes=simple_axes, sampling_config=None, n_samples=200, random_seed=999
        )

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_samples == 200
        assert sampler.random_seed == 999

    def test_create_sampler_with_custom_config(self, mixed_axes):
        """Test creating a sampler with custom sampling configuration."""
        config = SamplingConfig(
            method="latin_hypercube",
            optimization_criterion="correlation",
            random_seed=100,
            lhs_scramble=False,
            lhs_strength=1,  # Use strength=1 to avoid prime requirement complexity
            batch_size=5000,
            show_progress=False,
        )

        sampler = create_sampler(
            axes=mixed_axes, sampling_config=config, n_samples=1000
        )

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_samples == 1000
        assert sampler.random_seed == 100
        assert sampler.optimization == "correlation"
        assert sampler.scramble is False
        assert sampler.strength == 1
        assert sampler.batch_size == 5000
        assert sampler.show_progress is False

    def test_create_sampler_seed_override(self, simple_axes):
        """Test that random_seed parameter overrides config seed."""
        config = SamplingConfig(random_seed=42)

        sampler = create_sampler(
            axes=simple_axes,
            sampling_config=config,
            random_seed=999,  # Should override config seed
        )

        assert sampler.random_seed == 999

    def test_create_sampler_various_optimization_criteria(self, simple_axes):
        """Test creating samplers with different optimization criteria."""
        criteria = ["maximin", "correlation", "esi", "lloyd"]

        for criterion in criteria:
            config = SamplingConfig(optimization_criterion=criterion)
            sampler = create_sampler(axes=simple_axes, sampling_config=config)

            assert isinstance(sampler, LatinHypercubeSampler)
            assert sampler.optimization == criterion

    def test_create_sampler_none_optimization(self, simple_axes):
        """Test creating a sampler with no optimization."""
        config = SamplingConfig(optimization_criterion=None)
        sampler = create_sampler(axes=simple_axes, sampling_config=config)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.optimization is None

    def test_create_sampler_different_strengths(self, simple_axes):
        """Test creating samplers with different strength values."""
        # Test strength=1
        config1 = SamplingConfig(lhs_strength=1)
        sampler1 = create_sampler(axes=simple_axes, sampling_config=config1)
        assert sampler1.strength == 1

        # Test strength=2 (requires perfect square samples and fewer dimensions)
        single_axis = {"x": simple_axes["x"]}
        config2 = SamplingConfig(lhs_strength=2)
        sampler2 = create_sampler(
            axes=single_axis,
            sampling_config=config2,
            n_samples=49,  # 7^2, where 7 is prime
        )
        assert sampler2.strength == 2

    def test_create_sampler_different_batch_sizes(self, simple_axes):
        """Test creating samplers with different batch sizes."""
        batch_sizes = [100, 1000, 5000, 10000]

        for batch_size in batch_sizes:
            config = SamplingConfig(batch_size=batch_size)
            sampler = create_sampler(axes=simple_axes, sampling_config=config)

            assert isinstance(sampler, LatinHypercubeSampler)
            assert sampler.batch_size == batch_size

    def test_create_sampler_progress_settings(self, simple_axes):
        """Test creating samplers with different progress settings."""
        # Test with progress enabled
        config1 = SamplingConfig(show_progress=True)
        sampler1 = create_sampler(axes=simple_axes, sampling_config=config1)
        assert sampler1.show_progress is True

        # Test with progress disabled
        config2 = SamplingConfig(show_progress=False)
        sampler2 = create_sampler(axes=simple_axes, sampling_config=config2)
        assert sampler2.show_progress is False

    def test_create_sampler_scramble_settings(self, simple_axes):
        """Test creating samplers with different scramble settings."""
        # Test with scramble enabled
        config1 = SamplingConfig(lhs_scramble=True)
        sampler1 = create_sampler(axes=simple_axes, sampling_config=config1)
        assert sampler1.scramble is True

        # Test with scramble disabled
        config2 = SamplingConfig(lhs_scramble=False)
        sampler2 = create_sampler(axes=simple_axes, sampling_config=config2)
        assert sampler2.scramble is False

    def test_create_sampler_unsupported_method(self, simple_axes):
        """Test creating a sampler with unsupported method raises ValueError."""
        # Since Pydantic validates the method field, we need to test with a valid but unimplemented method
        config = SamplingConfig(
            method="random"
        )  # This is a valid enum value but not implemented

        with pytest.raises(ValueError, match="Unsupported sampling method: random"):
            create_sampler(axes=simple_axes, sampling_config=config)

    def test_create_sampler_sobol_method(self, simple_axes):
        """Test creating a sampler with 'sobol' method raises ValueError."""
        config = SamplingConfig(method="sobol")

        with pytest.raises(ValueError, match="Unsupported sampling method: sobol"):
            create_sampler(axes=simple_axes, sampling_config=config)

    def test_create_sampler_invalid_method_pydantic_error(self, simple_axes):
        """Test that creating SamplingConfig with invalid method raises Pydantic ValidationError."""
        with pytest.raises(
            ValidationError,
            match="Input should be 'latin_hypercube', 'random' or 'sobol'",
        ):
            SamplingConfig(method="invalid_method")

    def test_create_sampler_with_many_axes(self):
        """Test creating a sampler with many axes."""
        many_axes = {}
        for i in range(10):
            many_axes[f"axis_{i}"] = ContinuousAxis(type="uniform", min=0.0, max=1.0)

        sampler = create_sampler(axes=many_axes, n_samples=100)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_dimensions == 10
        assert len(sampler.axes) == 10

    def test_create_sampler_categorical_only(self):
        """Test creating a sampler with only categorical axes."""
        categorical_axes = {
            "model": CategoricalAxis(type="categorical", values=["a", "b", "c"]),
            "style": CategoricalAxis(type="categorical", values=["x", "y"]),
        }

        sampler = create_sampler(axes=categorical_axes)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_categorical == 2
        assert sampler.n_continuous == 0

    def test_create_sampler_continuous_only(self):
        """Test creating a sampler with only continuous axes."""
        continuous_axes = {
            "temp": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "pressure": ContinuousAxis(type="uniform", min=0.5, max=2.0),
        }

        sampler = create_sampler(axes=continuous_axes)

        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.n_continuous == 2
        assert sampler.n_categorical == 0

    def test_create_sampler_parameter_passing(self, simple_axes):
        """Test that all configuration parameters are correctly passed to sampler."""
        config = SamplingConfig(
            method="latin_hypercube",
            optimization_criterion="esi",
            random_seed=456,
            lhs_scramble=False,
            lhs_strength=1,
            batch_size=2000,
            show_progress=True,
        )

        sampler = create_sampler(
            axes=simple_axes,
            sampling_config=config,
            n_samples=750,
            random_seed=789,  # Should override config
        )

        # Verify all parameters are correctly set
        assert isinstance(sampler, LatinHypercubeSampler)
        assert sampler.axes == simple_axes
        assert sampler.n_samples == 750
        assert sampler.random_seed == 789  # Overridden
        assert sampler.optimization == "esi"
        assert sampler.scramble is False
        assert sampler.strength == 1
        assert sampler.batch_size == 2000
        assert sampler.show_progress is True

    def test_create_sampler_functional_test(self, simple_axes):
        """Test that created sampler actually works by generating samples."""
        sampler = create_sampler(axes=simple_axes, n_samples=50, random_seed=42)

        result = sampler.sample()

        assert result.samples.shape == (50, 2)
        assert result.metadata["n_samples"] == 50
        assert result.metadata["n_dimensions"] == 2
        assert result.quality_metrics is not None
