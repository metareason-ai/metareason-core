import logging
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import beta, norm, qmc, truncnorm

from ..config import AxisConfig

logger = logging.getLogger(__name__)


class LhsSampler:
    """Latin Hypercube Sampling implementation optimized for LLM parameter exploration."""

    def __init__(self, axes_config: List[AxisConfig], random_seed: int = 42):
        self.axes_config = axes_config
        self.rng = np.random.default_rng(random_seed)

        self.continuous_axes = [ax for ax in axes_config if ax.type == "continuous"]
        self.categorical_axes = [ax for ax in axes_config if ax.type == "categorical"]

        logger.info(
            f"Initialized sampler with {len(self.categorical_axes)} categorical axes and {len(self.continuous_axes)} continuous axes"
        )

    def generate_samples(
        self, n_samples: int, optimize_lhs: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate n_samples parameter combinations using Latin Hypercube Sampling."""
        samples = []

        continuous_samples = self._generate_continuous_samples(n_samples, optimize_lhs)
        categorical_samples = self._generate_categorical_samples(n_samples)

        for i in range(n_samples):
            sample = {}

            for j, axis in enumerate(self.continuous_axes):
                sample[axis.name] = continuous_samples[i, j]

            for j, axis in enumerate(self.categorical_axes):
                sample[axis.name] = categorical_samples[i][j]

            samples.append(sample)

        return samples

    def _generate_continuous_samples(
        self, n_samples: int, optimize: bool
    ) -> np.ndarray:
        if not self.continuous_axes:
            return np.empty((n_samples, 0))

        n_dims = len(self.continuous_axes)
        lhs_sampler = qmc.LatinHypercube(d=n_dims, seed=self.rng)
        lhs_samples = lhs_sampler.random(n_samples)

        if optimize:
            lhs_samples = self._optimize_lhs_maximin(lhs_samples, lhs_sampler)

        transformed_samples = np.zeros_like(lhs_samples)

        for i, axis in enumerate(self.continuous_axes):
            uniform_samples = lhs_samples[:, i]
            transformed_samples[:, i] = self._transform_to_distribution(
                uniform_samples, axis
            )

        return transformed_samples

    def _optimize_lhs_maximin(
        self, samples: np.ndarray, sampler: qmc.LatinHypercube
    ) -> np.ndarray:
        n_candidates = 10
        best_samples = samples
        best_score = self._calculate_maximin_score(samples)

        for _ in range(n_candidates - 1):
            candidate_samples = sampler.random(samples.shape[0])
            score = self._calculate_maximin_score(candidate_samples)

            if score > best_score:
                best_score = score
                best_samples = candidate_samples

        logger.info(
            f"LHS optimization: improved maximin score from {self._calculate_maximin_score(samples):.4f} to {best_score:.4f}"
        )
        return best_samples

    def _calculate_maximin_score(self, samples: np.ndarray) -> float:
        distances = pdist(samples)
        return np.min(distances) if len(distances) > 0 else 0.0

    def _transform_to_distribution(
        self, uniform_samples: np.ndarray, axis: AxisConfig
    ) -> np.ndarray:
        distribution = axis.distribution
        params = axis.params or {}

        if distribution == "uniform":
            min_val = params.get("min", 0.0)
            max_val = params.get("max", 1.0)
            return min_val + uniform_samples * (max_val - min_val)

        elif distribution == "normal":
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", 1.0)
            return norm.ppf(uniform_samples, loc=mu, scale=sigma)

        elif distribution == "truncnorm":
            mu = params.get("mu", 0.0)
            sigma = params.get("sigma", 1.0)
            min_val = params.get("min", -2.0)
            max_val = params.get("max", 2.0)

            a = (min_val - mu) / sigma
            b = (max_val - mu) / sigma

            return truncnorm.ppf(uniform_samples, a, b, loc=mu, scale=sigma)

        elif distribution == "beta":
            alpha = params.get("alpha", 1.0)
            beta_param = params.get("beta", 1.0)

            return beta.ppf(uniform_samples, alpha, beta_param)

        else:
            raise ValueError(f"Unkown distribution: {distribution}")

    def _generate_categorical_samples(self, n_samples: int) -> np.ndarray:
        if not self.categorical_axes:
            return np.empty((n_samples, 0), dtype=object)

        all_columns = []

        for axis in self.categorical_axes:
            values_list = list(axis.values)
            base_count = n_samples // len(values_list)
            remainder = n_samples % len(values_list)
            categorical_samples = values_list * base_count + values_list[:remainder]
            self.rng.shuffle(categorical_samples)
            all_columns.append(categorical_samples)

        return np.column_stack(all_columns)
