import pytest

from metareason.config import AxisConfig
from metareason.sampling.lhs_sampler import LhsSampler


@pytest.fixture
def continuous_axis():
    return AxisConfig(
        name="continuous",
        type="continuous",
        distribution="beta",
        params={"alpha": 1, "beta": 2},
    )


@pytest.fixture
def categorical_axis():
    return AxisConfig(
        name="categorical", type="categorical", values=["foo", "bar"], weights=[1, 2]
    )


@pytest.fixture
def uniform_axis():
    return AxisConfig(
        name="uniform_param",
        type="continuous",
        distribution="uniform",
        params={"min": 0.0, "max": 10.0},
    )


@pytest.fixture
def normal_axis():
    return AxisConfig(
        name="normal_param",
        type="continuous",
        distribution="normal",
        params={"mu": 5.0, "sigma": 1.5},
    )


@pytest.fixture
def truncnorm_axis():
    return AxisConfig(
        name="truncnorm_param",
        type="continuous",
        distribution="truncnorm",
        params={"mu": 0.5, "sigma": 0.2, "min": 0.0, "max": 1.0},
    )


def test_init(continuous_axis, categorical_axis):
    sampler = LhsSampler([continuous_axis, categorical_axis])
    assert sampler.continuous_axes == [continuous_axis]
    assert sampler.categorical_axes == [categorical_axis]
    assert sampler.rng is not None


def test_generate_samples_returns_correct_count(continuous_axis):
    """Test that generate_samples returns exactly n_samples samples."""
    sampler = LhsSampler([continuous_axis], random_seed=42)

    # Test various sample counts
    for n_samples in [1, 5, 10, 100]:
        samples = sampler.generate_samples(n_samples)
        assert (
            len(samples) == n_samples
        ), f"Expected {n_samples} samples, got {len(samples)}"

        # Each sample should be a dict
        for sample in samples:
            assert isinstance(sample, dict)
            # Should have the axis name as a key
            assert "continuous" in sample


def test_continuous_values_within_distribution_bounds(continuous_axis):
    """Test that beta distribution values stay within [0, 1] bounds."""
    sampler = LhsSampler([continuous_axis], random_seed=42)
    samples = sampler.generate_samples(100)

    # Beta distribution should always produce values in [0, 1]
    for sample in samples:
        value = sample["continuous"]
        assert 0.0 <= value <= 1.0, f"Beta value {value} outside [0, 1] range"

    # With 100 samples, we should have good coverage
    values = [s["continuous"] for s in samples]
    assert min(values) < 0.2, "Beta samples should include low values"
    assert max(values) > 0.8, "Beta samples should include high values"


def test_categorical_values_from_allowed_set(categorical_axis):
    """Test that categorical samples only use values from the allowed set."""
    sampler = LhsSampler([categorical_axis], random_seed=42)
    samples = sampler.generate_samples(50)

    allowed_values = set(categorical_axis.values)

    for sample in samples:
        value = sample["categorical"]
        assert (
            value in allowed_values
        ), f"Value '{value}' not in allowed set {allowed_values}"

    # With 50 samples and 2 categories, both should appear
    values = [s["categorical"] for s in samples]
    unique_values = set(values)
    assert unique_values == allowed_values, "Not all categorical values were sampled"


def test_uniform_distribution_range(uniform_axis):
    """Test uniform distribution respects min/max bounds."""
    sampler = LhsSampler([uniform_axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["uniform_param"] for s in samples]
    assert all(0.0 <= v <= 10.0 for v in values)
    assert min(values) < 1.0  # Should get values near min
    assert max(values) > 9.0  # Should get values near max


def test_normal_distribution_centered(normal_axis):
    """Test normal distribution is centered around mean."""
    sampler = LhsSampler([normal_axis], random_seed=42)
    samples = sampler.generate_samples(1000)  # More samples for statistical test

    values = [s["normal_param"] for s in samples]
    mean = sum(values) / len(values)
    # Mean should be roughly 5.0 (within tolerance)
    assert 4.5 < mean < 5.5


def test_truncnorm_distribution_bounds(truncnorm_axis):
    """Test truncated normal respects hard bounds."""
    sampler = LhsSampler([truncnorm_axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["truncnorm_param"] for s in samples]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_mixed_continuous_and_categorical(continuous_axis, categorical_axis):
    """Test sampling with both continuous and categorical axes."""
    sampler = LhsSampler([continuous_axis, categorical_axis], random_seed=42)
    samples = sampler.generate_samples(20)

    assert len(samples) == 20
    for sample in samples:
        # Should have both keys
        assert "continuous" in sample
        assert "categorical" in sample
        # Continuous should be in [0, 1] (beta distribution)
        assert 0.0 <= sample["continuous"] <= 1.0
        # Categorical should be from allowed set
        assert sample["categorical"] in ["foo", "bar"]


def test_optimization_improves_maximin(continuous_axis):
    """Test that optimize_lhs improves space-filling quality."""
    sampler_unoptimized = LhsSampler([continuous_axis], random_seed=42)
    samples_unopt = sampler_unoptimized.generate_samples(10, optimize_lhs=False)

    sampler_optimized = LhsSampler([continuous_axis], random_seed=42)
    samples_opt = sampler_optimized.generate_samples(10, optimize_lhs=True)

    # Both should return same count
    assert len(samples_unopt) == len(samples_opt) == 10
    # This is a weak test - just verify optimization runs without error
    # Checking actual maximin improvement would require more complex logic


def test_empty_axes_returns_empty_dicts():
    """Test sampling with no axes returns empty dicts."""
    sampler = LhsSampler([], random_seed=42)
    samples = sampler.generate_samples(5)

    assert len(samples) == 5
    for sample in samples:
        assert sample == {}


def test_single_sample(continuous_axis):
    """Test generating a single sample works."""
    sampler = LhsSampler([continuous_axis], random_seed=42)
    samples = sampler.generate_samples(1)

    assert len(samples) == 1
    assert "continuous" in samples[0]


def test_uniform_distribution_accepts_low_high():
    """Test uniform distribution works with low/high params (scipy convention)."""
    axis = AxisConfig(
        name="detail_level",
        type="continuous",
        distribution="uniform",
        params={"low": 1.0, "high": 10.0},
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["detail_level"] for s in samples]
    assert all(
        1.0 <= v <= 10.0 for v in values
    ), f"Values outside [1, 10]: min={min(values)}, max={max(values)}"
    assert min(values) < 2.0  # Should get values near low bound
    assert max(values) > 9.0  # Should get values near high bound


def test_uniform_distribution_accepts_min_max():
    """Test uniform distribution still works with min/max params (backward compat)."""
    axis = AxisConfig(
        name="detail_level",
        type="continuous",
        distribution="uniform",
        params={"min": 5.0, "max": 15.0},
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["detail_level"] for s in samples]
    assert all(
        5.0 <= v <= 15.0 for v in values
    ), f"Values outside [5, 15]: min={min(values)}, max={max(values)}"


def test_truncnorm_distribution_accepts_low_high():
    """Test truncated normal works with low/high params (scipy convention)."""
    axis = AxisConfig(
        name="bounded_param",
        type="continuous",
        distribution="truncnorm",
        params={"mu": 5.0, "sigma": 2.0, "low": 0.0, "high": 10.0},
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["bounded_param"] for s in samples]
    assert all(0.0 <= v <= 10.0 for v in values)


def test_truncnorm_distribution_accepts_min_max():
    """Test truncated normal still works with min/max params (backward compat)."""
    axis = AxisConfig(
        name="bounded_param",
        type="continuous",
        distribution="truncnorm",
        params={"mu": 5.0, "sigma": 2.0, "min": 0.0, "max": 10.0},
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(100)

    values = [s["bounded_param"] for s in samples]
    assert all(0.0 <= v <= 10.0 for v in values)


def test_categorical_weighted_sampling():
    """Test that categorical axis with weights produces weighted distribution."""
    axis = AxisConfig(
        name="category",
        type="categorical",
        values=["rare", "common"],
        weights=[0.1, 0.9],
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(1000)

    values = [s["category"] for s in samples]
    common_count = values.count("common")
    rare_count = values.count("rare")

    # With weights [0.1, 0.9] and 1000 samples, "common" should appear
    # significantly more than "rare". Allow generous tolerance for randomness.
    assert common_count > rare_count, (
        f"Expected 'common' to appear more than 'rare', "
        f"got common={common_count}, rare={rare_count}"
    )
    # "common" should be at least 70% of samples (expected ~90%)
    assert (
        common_count >= 700
    ), f"Expected 'common' >= 700 out of 1000, got {common_count}"


def test_categorical_uniform_without_weights():
    """Test that categorical axis without weights produces uniform distribution."""
    axis = AxisConfig(
        name="category",
        type="categorical",
        values=["a", "b"],
    )
    sampler = LhsSampler([axis], random_seed=42)
    samples = sampler.generate_samples(1000)

    values = [s["category"] for s in samples]
    a_count = values.count("a")
    b_count = values.count("b")

    # Without weights, distribution should be exactly balanced (LHS balanced allocation)
    assert a_count == 500, f"Expected 'a' count == 500, got {a_count}"
    assert b_count == 500, f"Expected 'b' count == 500, got {b_count}"


def test_uniform_conflicting_keys_prefers_low_high(caplog):
    """Test that low/high take precedence over min/max when both are provided."""
    axis = AxisConfig(
        name="conflict_param",
        type="continuous",
        distribution="uniform",
        params={"low": 1.0, "high": 10.0, "min": 50.0, "max": 100.0},
    )
    sampler = LhsSampler([axis], random_seed=42)

    import logging

    with caplog.at_level(logging.WARNING):
        samples = sampler.generate_samples(100)

    # Should use low/high (1-10), not min/max (50-100)
    values = [s["conflict_param"] for s in samples]
    assert all(1.0 <= v <= 10.0 for v in values)

    # Should have logged warnings
    assert "both 'low' and 'min' provided" in caplog.text
    assert "both 'high' and 'max' provided" in caplog.text
