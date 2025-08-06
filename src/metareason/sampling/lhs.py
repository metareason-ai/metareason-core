"""Latin Hypercube Sampling implementation for MetaReason."""

from typing import Dict, Literal, Optional

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from ..config import AxisConfigType
from .base import BaseSampler, SampleResult


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube Sampler with optimization and quality metrics."""

    def __init__(
        self,
        axes: Dict[str, AxisConfigType],
        n_samples: int = 2000,
        random_seed: Optional[int] = 42,
        optimization: Optional[
            Literal["maximin", "correlation", "esi", "lloyd"]
        ] = "maximin",
        scramble: bool = True,
        strength: int = 1,
        batch_size: int = 10000,
        show_progress: bool = True,
    ) -> None:
        """Initialize the Latin Hypercube Sampler.

        Args:
            axes: Dictionary of axis configurations
            n_samples: Number of samples to generate (default 2000)
            random_seed: Random seed for reproducibility (default 42)
            optimization: Optimization criterion for LHS
                - "maximin": Maximize minimum distance between points (default)
                - "correlation": Minimize correlation between dimensions
                - "esi": Enhanced Stochastic Improvement
                - "lloyd": Lloyd-Max algorithm for equal spacing
            scramble: Whether to randomly place samples within cells (default True)
            strength: Strength of the LHS (1 or 2, default 1)
            batch_size: Maximum samples per batch for memory efficiency
            show_progress: Whether to show progress bar for large generations
        """
        super().__init__(axes, n_samples, random_seed)
        self.optimization = optimization
        self.scramble = scramble
        self.strength = strength
        self.batch_size = batch_size
        self.show_progress = show_progress

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate sampler parameters."""
        if self.strength == 2:
            p = int(np.sqrt(self.n_samples))
            if p * p != self.n_samples:
                raise ValueError(
                    f"For strength=2, n_samples must be a perfect square. "
                    f"Got {self.n_samples}. Suggestion: Use {p*p} or {(p+1)*(p+1)}"
                )

            if not self._is_prime(p):
                raise ValueError(
                    f"For strength=2, sqrt(n_samples) must be prime. "
                    f"Got sqrt({self.n_samples}) = {p}"
                )

            if self.n_dimensions > p + 1:
                raise ValueError(
                    f"For strength=2 with n_samples={self.n_samples}, "
                    f"dimensions must be <= {p+1}, got {self.n_dimensions}"
                )

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _map_optimization(self) -> Optional[str]:
        """Map optimization criterion to scipy parameter."""
        mapping = {
            "maximin": "random-cd",
            "lloyd": "lloyd",
            "correlation": "random-cd",
            "esi": "random-cd",
        }
        return mapping.get(self.optimization)

    def generate_samples(self) -> np.ndarray:
        """Generate Latin Hypercube samples in the unit hypercube.

        Returns:
            Array of shape (n_samples, n_dimensions) with values in [0, 1]
        """
        if self.n_samples <= self.batch_size:
            return self._generate_batch(self.n_samples)

        n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
        samples = []

        if self.show_progress:
            batch_range = tqdm(range(n_batches), desc="Generating LHS batches")
        else:
            batch_range = range(n_batches)

        for i in batch_range:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            batch_size = end_idx - start_idx

            batch_samples = self._generate_batch(batch_size)
            samples.append(batch_samples)

        return np.vstack(samples)

    def _generate_batch(self, batch_size: int) -> np.ndarray:
        """Generate a batch of Latin Hypercube samples.

        Args:
            batch_size: Number of samples in this batch

        Returns:
            Array of shape (batch_size, n_dimensions)
        """
        lhs_optimization = self._map_optimization()

        sampler = qmc.LatinHypercube(
            d=self.n_dimensions,
            scramble=self.scramble,
            strength=self.strength,
            optimization=lhs_optimization,
            rng=self.rng,
        )

        samples = sampler.random(n=batch_size)

        if self.optimization == "correlation" and lhs_optimization == "random-cd":
            samples = self._minimize_correlation(samples)
        elif self.optimization == "esi" and lhs_optimization == "random-cd":
            samples = self._enhanced_stochastic_improvement(samples)

        return samples

    def _minimize_correlation(self, samples: np.ndarray) -> np.ndarray:
        """Post-process samples to minimize correlation between dimensions.

        Args:
            samples: Initial LHS samples

        Returns:
            Optimized samples with reduced correlation
        """
        n_iterations = min(100, samples.shape[0])

        for _ in range(n_iterations):
            corr_matrix = np.corrcoef(samples.T)
            np.fill_diagonal(corr_matrix, 0)
            max_corr_idx = np.unravel_index(
                np.argmax(np.abs(corr_matrix)), corr_matrix.shape
            )

            if np.abs(corr_matrix[max_corr_idx]) < 0.1:
                break

            dim1, dim2 = max_corr_idx
            perm = self.rng.permutation(samples.shape[0])
            samples[:, dim2] = samples[perm, dim2]

        return samples

    def _enhanced_stochastic_improvement(self, samples: np.ndarray) -> np.ndarray:
        """Apply Enhanced Stochastic Improvement optimization.

        Args:
            samples: Initial LHS samples

        Returns:
            ESI-optimized samples
        """
        n_iterations = min(50, samples.shape[0] // 2)

        for _ in range(n_iterations):
            i, j = self.rng.choice(samples.shape[0], size=2, replace=False)

            distances_before = np.sum(
                np.linalg.norm(samples - samples[i], axis=1) ** 2
            ) + np.sum(np.linalg.norm(samples - samples[j], axis=1) ** 2)

            samples[[i, j]] = samples[[j, i]]

            distances_after = np.sum(
                np.linalg.norm(samples - samples[i], axis=1) ** 2
            ) + np.sum(np.linalg.norm(samples - samples[j], axis=1) ** 2)

            if distances_after < distances_before:
                samples[[i, j]] = samples[[j, i]]

        return samples

    def sample(self) -> SampleResult:
        """Generate samples with quality metrics.

        Returns:
            SampleResult with samples, metadata, and quality metrics
        """
        unit_samples = self.generate_samples()
        transformed_samples = self.transform_samples(unit_samples)

        result_array = np.empty((self.n_samples, self.n_dimensions), dtype=object)
        for name, values in transformed_samples.items():
            idx = self._axis_indices[name]
            result_array[:, idx] = values

        quality_metrics = self._compute_quality_metrics(unit_samples)

        metadata = {
            "n_samples": self.n_samples,
            "n_dimensions": self.n_dimensions,
            "n_continuous": self.n_continuous,
            "n_categorical": self.n_categorical,
            "random_seed": self.random_seed,
            "axis_names": list(self.axes.keys()),
            "optimization": self.optimization,
            "scramble": self.scramble,
            "strength": self.strength,
        }

        return SampleResult(
            samples=result_array, metadata=metadata, quality_metrics=quality_metrics
        )

    def _compute_quality_metrics(self, samples: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for the generated samples.

        Args:
            samples: Generated samples in unit hypercube

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        metrics["discrepancy"] = float(qmc.discrepancy(samples))

        if samples.shape[1] > 1:
            corr_matrix = np.corrcoef(samples.T)
            np.fill_diagonal(corr_matrix, 0)
            metrics["max_correlation"] = float(np.max(np.abs(corr_matrix)))
            metrics["mean_correlation"] = float(np.mean(np.abs(corr_matrix)))
        else:
            metrics["max_correlation"] = 0.0
            metrics["mean_correlation"] = 0.0

        distances = []
        n_subset = min(1000, samples.shape[0])
        subset_indices = self.rng.choice(samples.shape[0], size=n_subset, replace=False)
        for i in subset_indices[:100]:
            dists = np.linalg.norm(samples - samples[i], axis=1)
            dists[i] = np.inf
            distances.append(np.min(dists))

        metrics["min_distance"] = float(np.min(distances))
        metrics["mean_min_distance"] = float(np.mean(distances))

        for d in range(min(samples.shape[1], 5)):
            hist, _ = np.histogram(samples[:, d], bins=10)
            expected = samples.shape[0] / 10
            chi_squared = np.sum((hist - expected) ** 2 / expected)
            metrics[f"uniformity_dim_{d}"] = float(chi_squared)

        return metrics
