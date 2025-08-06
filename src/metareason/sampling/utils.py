"""Utilities for sampling operations."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import AxisConfigType, CategoricalAxis, ContinuousAxis


def encode_categorical_values(values: np.ndarray, categories: List[str]) -> np.ndarray:
    """Encode categorical values as integers.

    Args:
        values: Array of categorical values
        categories: List of category names

    Returns:
        Integer-encoded values
    """
    category_map = {cat: i for i, cat in enumerate(categories)}
    encoded = np.array([category_map[val] for val in values])
    return encoded


def decode_categorical_values(encoded: np.ndarray, categories: List[str]) -> np.ndarray:
    """Decode integer-encoded categorical values.

    Args:
        encoded: Integer-encoded values
        categories: List of category names

    Returns:
        Categorical values
    """
    decoded = np.array([categories[int(i)] for i in encoded])
    return decoded


def normalize_samples(
    samples: np.ndarray, axis_configs: Dict[str, AxisConfigType]
) -> np.ndarray:
    """Normalize samples to unit hypercube [0,1]^d.

    Args:
        samples: Raw samples
        axis_configs: Axis configurations

    Returns:
        Normalized samples
    """
    normalized = np.zeros_like(samples, dtype=float)

    for i, (_name, config) in enumerate(axis_configs.items()):
        if isinstance(config, ContinuousAxis):
            if config.type == "uniform":
                normalized[:, i] = (samples[:, i] - config.min) / (
                    config.max - config.min
                )
            elif config.type == "truncated_normal":
                from scipy import stats

                a = (config.min - config.mu) / config.sigma
                b = (config.max - config.mu) / config.sigma
                dist = stats.truncnorm(a, b, loc=config.mu, scale=config.sigma)
                normalized[:, i] = dist.cdf(samples[:, i])
            elif config.type == "beta":
                from scipy import stats

                dist = stats.beta(config.alpha, config.beta)
                normalized[:, i] = dist.cdf(samples[:, i])

        elif isinstance(config, CategoricalAxis):
            encoded = encode_categorical_values(samples[:, i], config.values)
            normalized[:, i] = encoded / (len(config.values) - 1)

    return normalized


def denormalize_samples(
    normalized: np.ndarray, axis_configs: Dict[str, AxisConfigType]
) -> np.ndarray:
    """Denormalize samples from unit hypercube to actual values.

    Args:
        normalized: Normalized samples in [0,1]^d
        axis_configs: Axis configurations

    Returns:
        Denormalized samples
    """
    samples = np.empty(normalized.shape, dtype=object)

    for i, (_name, config) in enumerate(axis_configs.items()):
        if isinstance(config, ContinuousAxis):
            if config.type == "uniform":
                samples[:, i] = config.min + normalized[:, i] * (
                    config.max - config.min
                )
            elif config.type == "truncated_normal":
                from scipy import stats

                a = (config.min - config.mu) / config.sigma
                b = (config.max - config.mu) / config.sigma
                dist = stats.truncnorm(a, b, loc=config.mu, scale=config.sigma)
                samples[:, i] = dist.ppf(normalized[:, i])
            elif config.type == "beta":
                from scipy import stats

                dist = stats.beta(config.alpha, config.beta)
                samples[:, i] = dist.ppf(normalized[:, i])

        elif isinstance(config, CategoricalAxis):
            indices = np.round(normalized[:, i] * (len(config.values) - 1)).astype(int)
            indices = np.clip(indices, 0, len(config.values) - 1)
            samples[:, i] = decode_categorical_values(indices, config.values)

    return samples


def save_samples(
    samples: np.ndarray,
    metadata: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = "npz",
) -> None:
    """Save samples to file.

    Args:
        samples: Sample array
        metadata: Metadata dictionary
        file_path: Path to save file
        format: File format ('npz', 'csv', 'json')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        np.savez_compressed(file_path, samples=samples, metadata=metadata)

    elif format == "csv":
        df = pd.DataFrame(samples)
        if "axis_names" in metadata:
            df.columns = metadata["axis_names"]
        df.to_csv(file_path, index=False)

        meta_path = file_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    elif format == "json":
        data = {
            "samples": samples.tolist(),
            "metadata": metadata,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_samples(
    file_path: Union[str, Path], format: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load samples from file.

    Args:
        file_path: Path to load file
        format: File format (auto-detect if None)

    Returns:
        Tuple of (samples, metadata)
    """
    file_path = Path(file_path)

    if format is None:
        if file_path.suffix == ".npz":
            format = "npz"
        elif file_path.suffix == ".csv":
            format = "csv"
        elif file_path.suffix == ".json":
            format = "json"
        else:
            raise ValueError(f"Cannot auto-detect format for {file_path}")

    if format == "npz":
        data = np.load(file_path, allow_pickle=True)
        samples = data["samples"]
        metadata = data["metadata"].item() if "metadata" in data else {}

    elif format == "csv":
        df = pd.read_csv(file_path)
        samples = df.values

        meta_path = file_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
    elif format == "json":
        with open(file_path, "r") as f:
            data = json.load(f)
        samples = np.array(data["samples"])
        metadata = data.get("metadata", {})

    else:
        raise ValueError(f"Unsupported format: {format}")

    return samples, metadata


def parallel_sample_generation(
    sampler_class,
    sampler_kwargs: Dict[str, Any],
    n_batches: int = 4,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate samples in parallel using multiple workers.

    Args:
        sampler_class: Sampler class to use
        sampler_kwargs: Keyword arguments for sampler
        n_batches: Number of batches to split generation into
        n_workers: Number of parallel workers (default: CPU count)
        show_progress: Whether to show progress bar

    Returns:
        Combined samples array
    """

    def generate_batch(batch_idx: int, batch_size: int, seed: int):
        """Generate a single batch of samples."""
        kwargs = sampler_kwargs.copy()
        kwargs["n_samples"] = batch_size
        kwargs["random_seed"] = seed
        kwargs["show_progress"] = False

        sampler = sampler_class(**kwargs)
        result = sampler.sample()
        return result.samples

    total_samples = sampler_kwargs.get("n_samples", 1000)
    base_seed = sampler_kwargs.get("random_seed", 42)

    batch_sizes = [total_samples // n_batches] * n_batches
    for i in range(total_samples % n_batches):
        batch_sizes[i] += 1

    batch_seeds = [base_seed + i * 1000 for i in range(n_batches)]

    all_samples = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(generate_batch, i, size, seed): i
            for i, (size, seed) in enumerate(zip(batch_sizes, batch_seeds))
        }

        if show_progress:
            futures_iter = tqdm(
                as_completed(futures), total=n_batches, desc="Generating batches"
            )
        else:
            futures_iter = as_completed(futures)

        results = [None] * n_batches
        for future in futures_iter:
            batch_idx = futures[future]
            results[batch_idx] = future.result()

    all_samples = np.vstack(results)
    return all_samples[:total_samples]


def compute_balance_statistics(
    samples: np.ndarray,
    axes: Dict[str, AxisConfigType],
    stratify_by: List[str],
    expected_counts: Dict[Tuple, int],
    n_strata: int,
) -> Dict[str, Any]:
    """Compute statistics on sampling balance.

    Args:
        samples: Generated samples
        axes: Axis configurations
        stratify_by: List of axes that were stratified
        expected_counts: Expected count per stratum
        n_strata: Total number of strata

    Returns:
        Dictionary with balance statistics
    """
    from scipy import stats as scipy_stats

    statistics = {
        "stratified_axes": stratify_by,
        "n_strata": n_strata,
        "n_samples": len(samples),
        "per_axis_balance": {},
        "overall_balance": {},
    }

    # Analyze balance for each stratified axis
    axis_indices = {name: list(axes.keys()).index(name) for name in stratify_by}

    for axis_name in stratify_by:
        axis = axes[axis_name]
        axis_idx = axis_indices[axis_name]
        values = samples[:, axis_idx]

        # Count occurrences
        unique, counts = np.unique(values, return_counts=True)
        actual_dist = dict(zip(unique, counts / len(samples)))

        # Calculate expected distribution
        if isinstance(axis, CategoricalAxis):
            if axis.weights is None:
                # Uniform distribution expected
                expected_dist = {v: 1.0 / len(axis.values) for v in axis.values}
            else:
                # Weighted distribution expected
                total_weight = sum(axis.weights)
                expected_dist = {
                    v: w / total_weight for v, w in zip(axis.values, axis.weights)
                }

        # Calculate chi-squared test
        observed = [
            counts[list(unique).index(v)] if v in unique else 0 for v in axis.values
        ]
        expected = [expected_dist[v] * len(samples) for v in axis.values]

        if all(e > 0 for e in expected):
            chi2, p_value = scipy_stats.chisquare(observed, expected)
        else:
            chi2, p_value = np.nan, np.nan

        # Calculate max deviation
        max_deviation = max(
            abs(actual_dist.get(v, 0) - expected_dist[v]) for v in axis.values
        )

        statistics["per_axis_balance"][axis_name] = {
            "expected_distribution": expected_dist,
            "actual_distribution": actual_dist,
            "chi_squared": float(chi2),
            "p_value": float(p_value),
            "max_deviation": float(max_deviation),
            "balance_score": float(1.0 - max_deviation) if max_deviation < 1 else 0.0,
        }

    # Calculate overall balance metrics
    all_deviations = [
        stats["max_deviation"] for stats in statistics["per_axis_balance"].values()
    ]

    statistics["overall_balance"] = {
        "mean_deviation": float(np.mean(all_deviations)),
        "max_deviation": float(np.max(all_deviations)),
        "balance_score": float(1.0 - np.mean(all_deviations)),
    }

    # Add stratum-level statistics if multiple axes
    if len(stratify_by) > 1:
        from itertools import product

        strata_combinations = list(
            product(*[axes[name].values for name in stratify_by])
        )

        actual_stratum_counts = {}
        for combination in strata_combinations:
            mask = np.ones(len(samples), dtype=bool)
            for axis_name, value in zip(stratify_by, combination):
                axis_idx = axis_indices[axis_name]
                mask &= samples[:, axis_idx] == value
            actual_stratum_counts[combination] = np.sum(mask)

        statistics["stratum_counts"] = {
            "expected": {str(k): v for k, v in expected_counts.items()},
            "actual": {str(k): v for k, v in actual_stratum_counts.items()},
        }

    return statistics


def stratified_sampling(
    sampler_class,
    sampler_kwargs: Dict[str, Any],
    stratify_by: List[str],
    ensure_balance: bool = True,
    return_stats: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Perform stratified sampling across categorical axes.

    Args:
        sampler_class: Sampler class to use
        sampler_kwargs: Keyword arguments for sampler
        stratify_by: List of categorical axis names to stratify by
        ensure_balance: Whether to ensure balanced representation
        return_stats: Whether to return balance statistics

    Returns:
        Stratified samples, or tuple of (samples, statistics) if return_stats=True
    """
    axes = sampler_kwargs["axes"]
    n_samples = sampler_kwargs.get("n_samples", 1000)

    strata_configs = []
    for axis_name in stratify_by:
        if axis_name not in axes:
            raise ValueError(f"Axis {axis_name} not found in axes configuration")

        axis = axes[axis_name]
        if not isinstance(axis, CategoricalAxis):
            raise ValueError(f"Axis {axis_name} must be categorical for stratification")

        strata_configs.append((axis_name, axis.values))

    from itertools import product

    strata_combinations = list(product(*[config[1] for config in strata_configs]))
    n_strata = len(strata_combinations)

    if ensure_balance:
        samples_per_stratum = n_samples // n_strata
        remainder = n_samples % n_strata
    else:
        samples_per_stratum = max(1, n_samples // n_strata)
        remainder = 0

    all_samples = []
    stratum_counts = {}  # Track actual samples per stratum

    for i, combination in enumerate(strata_combinations):
        stratum_samples = samples_per_stratum
        if i < remainder:
            stratum_samples += 1

        if stratum_samples == 0:
            continue

        stratum_kwargs = sampler_kwargs.copy()
        stratum_kwargs["n_samples"] = stratum_samples
        stratum_kwargs["random_seed"] = sampler_kwargs.get("random_seed", 42) + i * 1000

        sampler = sampler_class(**stratum_kwargs)
        result = sampler.sample()

        for _j, (axis_name, value) in enumerate(zip(stratify_by, combination)):
            axis_idx = list(axes.keys()).index(axis_name)
            result.samples[:, axis_idx] = value

        all_samples.append(result.samples)
        stratum_counts[combination] = stratum_samples

    combined = np.vstack(all_samples)

    rng = np.random.default_rng(sampler_kwargs.get("random_seed", 42))
    shuffle_idx = rng.permutation(len(combined))
    final_samples = combined[shuffle_idx][:n_samples]

    if not return_stats:
        return final_samples

    # Compute balance statistics
    statistics = compute_balance_statistics(
        final_samples, axes, stratify_by, stratum_counts, n_strata
    )

    return final_samples, statistics
