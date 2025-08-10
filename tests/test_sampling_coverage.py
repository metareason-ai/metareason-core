"""Additional tests to improve sampling module coverage."""

import numpy as np
import pytest

from metareason.config.axes import CategoricalAxis, ContinuousAxis
from metareason.sampling import LatinHypercubeSampler


class TestBaseSamplerCoverage:
    """Additional tests for base sampler coverage."""

    def test_beta_distribution_transform(self):
        """Test beta distribution transformation."""
        axes = {"param": ContinuousAxis(type="beta", alpha=2.0, beta=5.0)}
        sampler = LatinHypercubeSampler(axes=axes, n_samples=50, random_seed=42)
        result = sampler.sample()

        assert result.samples.shape == (50, 1)
        values = result.samples[:, 0]
        assert np.all(values >= 0)
        assert np.all(values <= 1)

    def test_categorical_with_weights(self):
        """Test categorical axis with weights."""
        axes = {
            "category": CategoricalAxis(
                type="categorical", values=["A", "B", "C"], weights=[0.5, 0.3, 0.2]
            )
        }
        sampler = LatinHypercubeSampler(axes=axes, n_samples=100, random_seed=42)
        result = sampler.sample()

        assert result.samples.shape == (100, 1)
        values = result.samples[:, 0]
        assert all(v in ["A", "B", "C"] for v in values)

    def test_custom_distribution_not_implemented(self):
        """Test that custom distribution raises NotImplementedError."""
        axes = {
            "param": ContinuousAxis(
                type="custom", module="mymodule", class_name="MyDist"
            )
        }
        sampler = LatinHypercubeSampler(axes=axes, n_samples=10, random_seed=42)

        with pytest.raises(NotImplementedError):
            sampler.sample()

    def test_lhs_lloyd_optimization(self):
        """Test Lloyd optimization."""
        axes = {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "y": ContinuousAxis(type="uniform", min=0.0, max=1.0),
        }
        sampler = LatinHypercubeSampler(
            axes=axes,
            n_samples=50,
            random_seed=42,
            optimization="lloyd",
        )
        result = sampler.sample()

        assert result.samples.shape == (50, 2)
        assert result.metadata["optimization"] == "lloyd"

    def test_lhs_no_scramble(self):
        """Test LHS without scrambling."""
        axes = {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
        }
        sampler = LatinHypercubeSampler(
            axes=axes,
            n_samples=20,
            random_seed=42,
            scramble=False,
        )
        result = sampler.sample()

        assert result.samples.shape == (20, 1)
        assert result.metadata["scramble"] is False

    def test_sampler_properties(self):
        """Test sampler properties."""
        axes = {
            "x": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "y": ContinuousAxis(type="uniform", min=0.0, max=1.0),
            "cat": CategoricalAxis(type="categorical", values=["A", "B"]),
        }
        sampler = LatinHypercubeSampler(axes=axes, n_samples=10)

        assert sampler.n_continuous == 2
        assert sampler.n_categorical == 1
        assert sampler.n_dimensions == 3
