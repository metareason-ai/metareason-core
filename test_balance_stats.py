#!/usr/bin/env python3
"""Test script for balance statistics in stratified sampling."""

from metareason.config import CategoricalAxis
from metareason.sampling.lhs import LatinHypercubeSampler
from metareason.sampling.utils import stratified_sampling


def test_stratified_balance():
    """Test balance statistics with stratification."""
    print("\n" + "=" * 60)
    print("TEST 1: STRATIFIED SAMPLING (EQUAL ALLOCATION)")
    print("=" * 60)

    # Define axes with weighted categorical variables
    axes = {
        "model_size": CategoricalAxis(
            values=["small", "medium", "large"],
            weights=[0.5, 0.3, 0.2],  # These weights are overridden by stratification
        ),
        "region": CategoricalAxis(
            values=["USA", "Europe", "Asia"], weights=None  # Uniform distribution
        ),
    }

    sampler_kwargs = {
        "axes": axes,
        "n_samples": 900,  # Divisible by 9 strata
        "random_seed": 42,
    }

    samples, stats = stratified_sampling(
        LatinHypercubeSampler,
        sampler_kwargs,
        stratify_by=["model_size", "region"],
        ensure_balance=True,
        return_stats=True,
    )

    print(f"\nTotal samples: {stats['n_samples']}")
    print(f"Number of strata: {stats['n_strata']}")

    print("\nPer-axis balance scores:")
    for axis_name, axis_stats in stats["per_axis_balance"].items():
        print(f"  {axis_name}: {axis_stats['balance_score']:.3f}")

    print(f"\nOverall balance score: {stats['overall_balance']['balance_score']:.3f}")

    # With stratification, each stratum gets equal samples
    assert all(
        count == 100 for count in stats["stratum_counts"]["actual"].values()
    ), "Each stratum should have exactly 100 samples"

    return samples, stats


def test_weighted_balance():
    """Test balance statistics with weighted sampling (no stratification)."""
    print("\n" + "=" * 60)
    print("TEST 2: WEIGHTED SAMPLING (NO STRATIFICATION)")
    print("=" * 60)

    # Define axes with weighted categorical variables
    axes = {
        "model_size": CategoricalAxis(
            values=["small", "medium", "large"],
            weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
        ),
        "region": CategoricalAxis(
            values=["USA", "Europe", "Asia"], weights=[0.6, 0.3, 0.1]  # 60%, 30%, 10%
        ),
    }

    # Use regular LHS sampling (not stratified) to respect weights
    sampler = LatinHypercubeSampler(axes=axes, n_samples=1000, random_seed=42)
    result = sampler.sample()

    # Manually compute statistics for comparison
    model_counts = {}
    region_counts = {}

    for i in range(result.samples.shape[0]):
        model = result.samples[i, 0]
        region = result.samples[i, 1]
        model_counts[model] = model_counts.get(model, 0) + 1
        region_counts[region] = region_counts.get(region, 0) + 1

    print(f"\nTotal samples: {result.samples.shape[0]}")

    print("\nModel size distribution (target: 50%, 30%, 20%):")
    for value in ["small", "medium", "large"]:
        count = model_counts.get(value, 0)
        pct = count / result.samples.shape[0] * 100
        print(f"  {value}: {count} samples ({pct:.1f}%)")

    print("\nRegion distribution (target: 60%, 30%, 10%):")
    for value in ["USA", "Europe", "Asia"]:
        count = region_counts.get(value, 0)
        pct = count / result.samples.shape[0] * 100
        print(f"  {value}: {count} samples ({pct:.1f}%)")

    # Verify weights are approximately respected
    assert 450 < model_counts.get("small", 0) < 550, "Small should be ~500 (50%)"
    assert 250 < model_counts.get("medium", 0) < 350, "Medium should be ~300 (30%)"
    assert 150 < model_counts.get("large", 0) < 250, "Large should be ~200 (20%)"

    return result


if __name__ == "__main__":
    # Test stratified sampling with balance statistics
    samples1, stats1 = test_stratified_balance()

    # Test weighted sampling (shows weights are respected)
    result2 = test_weighted_balance()

    print("\n✅ All tests passed!")
