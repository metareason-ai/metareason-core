"""Optimization strategies for Latin Hypercube Sampling."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


class BaseOptimizer(ABC):
    """Abstract base class for LHS optimization strategies."""

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """Initialize the optimizer.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def optimize(self, samples: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """Optimize the sample distribution.

        Args:
            samples: Initial samples to optimize
            n_iterations: Number of optimization iterations

        Returns:
            Optimized samples
        """
        pass

    @abstractmethod
    def compute_criterion(self, samples: np.ndarray) -> float:
        """Compute the optimization criterion value.

        Args:
            samples: Samples to evaluate

        Returns:
            Criterion value (lower is better)
        """
        pass


class MaximinOptimizer(BaseOptimizer):
    """Maximize the minimum distance between samples."""

    def optimize(self, samples: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """Optimize samples using maximin criterion.

        Args:
            samples: Initial samples
            n_iterations: Number of iterations

        Returns:
            Optimized samples
        """
        n_samples, n_dims = samples.shape
        best_samples = samples.copy()
        best_criterion = self.compute_criterion(best_samples)

        for _ in range(n_iterations):
            dim = self.rng.integers(0, n_dims)
            perm = self.rng.permutation(n_samples)

            candidate = samples.copy()
            candidate[:, dim] = candidate[perm, dim]

            criterion = self.compute_criterion(candidate)
            if criterion > best_criterion:
                best_samples = candidate.copy()
                best_criterion = criterion
                samples = candidate.copy()

        return best_samples

    def compute_criterion(self, samples: np.ndarray) -> float:
        """Compute minimum pairwise distance.

        Args:
            samples: Samples to evaluate

        Returns:
            Minimum distance (higher is better, so we return negative)
        """
        if samples.shape[0] > 1000:
            subset_idx = self.rng.choice(samples.shape[0], size=1000, replace=False)
            samples_subset = samples[subset_idx]
        else:
            samples_subset = samples

        distances = cdist(samples_subset, samples_subset)
        np.fill_diagonal(distances, np.inf)
        min_distance = np.min(distances)

        return float(min_distance)


class CorrelationMinimizer(BaseOptimizer):
    """Minimize correlation between dimensions."""

    def optimize(self, samples: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """Minimize correlation between dimensions.

        Args:
            samples: Initial samples
            n_iterations: Number of iterations

        Returns:
            Optimized samples
        """
        n_samples, n_dims = samples.shape
        best_samples = samples.copy()
        best_criterion = self.compute_criterion(best_samples)

        for _iteration in range(n_iterations):
            corr_matrix = np.corrcoef(samples.T)
            np.fill_diagonal(corr_matrix, 0)

            if np.max(np.abs(corr_matrix)) < 0.05:
                break

            max_corr_idx = np.unravel_index(
                np.argmax(np.abs(corr_matrix)), corr_matrix.shape
            )
            dim1, dim2 = max_corr_idx

            perm = self.rng.permutation(n_samples)
            candidate = samples.copy()
            candidate[:, dim2] = candidate[perm, dim2]

            criterion = self.compute_criterion(candidate)
            if criterion < best_criterion:
                best_samples = candidate.copy()
                best_criterion = criterion
                samples = candidate.copy()

        return best_samples

    def compute_criterion(self, samples: np.ndarray) -> float:
        """Compute maximum absolute correlation.

        Args:
            samples: Samples to evaluate

        Returns:
            Maximum absolute correlation
        """
        corr_matrix = np.corrcoef(samples.T)
        np.fill_diagonal(corr_matrix, 0)
        return float(np.max(np.abs(corr_matrix)))


class ESIOptimizer(BaseOptimizer):
    """Enhanced Stochastic Improvement optimizer."""

    def optimize(self, samples: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """Optimize using Enhanced Stochastic Improvement.

        Args:
            samples: Initial samples
            n_iterations: Number of iterations

        Returns:
            Optimized samples
        """
        n_samples = samples.shape[0]

        for _ in range(n_iterations):
            i, j = self.rng.choice(n_samples, size=2, replace=False)

            criterion_before = self._local_criterion(samples, i, j)

            samples[[i, j]] = samples[[j, i]]

            criterion_after = self._local_criterion(samples, i, j)

            if criterion_after > criterion_before:
                samples[[i, j]] = samples[[j, i]]

        return samples

    def _local_criterion(self, samples: np.ndarray, i: int, j: int) -> float:
        """Compute local criterion for two points.

        Args:
            samples: All samples
            i: First point index
            j: Second point index

        Returns:
            Local criterion value
        """
        distances_i = np.linalg.norm(samples - samples[i], axis=1)
        distances_j = np.linalg.norm(samples - samples[j], axis=1)

        distances_i[i] = np.inf
        distances_j[j] = np.inf

        min_dist_i = np.min(distances_i)
        min_dist_j = np.min(distances_j)

        return float(min_dist_i + min_dist_j)

    def compute_criterion(self, samples: np.ndarray) -> float:
        """Compute the phi_p criterion.

        Args:
            samples: Samples to evaluate

        Returns:
            Phi_p value (lower is better)
        """
        distances = cdist(samples, samples)
        np.fill_diagonal(distances, np.inf)

        p = 50
        phi_p = np.sum(distances ** (-p)) ** (1 / p)

        return float(phi_p)


class CustomOptimizer(BaseOptimizer):
    """Base class for custom optimization strategies."""

    def __init__(
        self,
        optimize_func: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        criterion_func: Optional[Callable[[np.ndarray], float]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize custom optimizer.

        Args:
            optimize_func: Custom optimization function
            criterion_func: Custom criterion function
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed)
        self.optimize_func = optimize_func
        self.criterion_func = criterion_func

    def optimize(self, samples: np.ndarray, n_iterations: int = 100) -> np.ndarray:
        """Apply custom optimization.

        Args:
            samples: Initial samples
            n_iterations: Number of iterations

        Returns:
            Optimized samples
        """
        if self.optimize_func is None:
            return samples

        return self.optimize_func(samples, n_iterations)

    def compute_criterion(self, samples: np.ndarray) -> float:
        """Compute custom criterion.

        Args:
            samples: Samples to evaluate

        Returns:
            Criterion value
        """
        if self.criterion_func is None:
            return 0.0

        return float(self.criterion_func(samples))


def benchmark_optimizers(
    samples: np.ndarray,
    optimizers: Dict[str, BaseOptimizer],
    n_iterations: int = 100,
    show_progress: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Benchmark different optimization strategies.

    Args:
        samples: Initial samples to optimize
        optimizers: Dictionary of optimizer instances
        n_iterations: Number of optimization iterations
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    iterator = tqdm(optimizers.items()) if show_progress else optimizers.items()

    for name, optimizer in iterator:
        if show_progress and hasattr(iterator, "set_description"):
            iterator.set_description(f"Benchmarking {name}")

        import time

        start_time = time.time()

        initial_criterion = optimizer.compute_criterion(samples)

        optimized = optimizer.optimize(samples.copy(), n_iterations)

        final_criterion = optimizer.compute_criterion(optimized)

        elapsed_time = time.time() - start_time

        results[name] = {
            "initial_criterion": float(initial_criterion),
            "final_criterion": float(final_criterion),
            "improvement": float(
                (final_criterion - initial_criterion) / abs(initial_criterion)
                if initial_criterion != 0
                else 0
            ),
            "time_seconds": elapsed_time,
        }

    return results
