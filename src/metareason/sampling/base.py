"""Base sampler interface for MetaReason sampling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..config import AxisConfigType, CategoricalAxis, ContinuousAxis


class SampleResult(BaseModel):
    """Result of a sampling operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    samples: np.ndarray
    metadata: Dict[str, Any]
    quality_metrics: Optional[Dict[str, float]] = None


class BaseSampler(ABC):
    """Abstract base class for all sampling strategies."""

    def __init__(
        self,
        axes: Dict[str, AxisConfigType],
        n_samples: int = 1000,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            axes: Dictionary of axis configurations
            n_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.axes = axes
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self._continuous_axes: List[Tuple[str, ContinuousAxis]] = []
        self._categorical_axes: List[Tuple[str, CategoricalAxis]] = []
        self._axis_indices: Dict[str, int] = {}

        self._categorize_axes()

    def _categorize_axes(self) -> None:
        """Categorize axes into continuous and categorical."""
        idx = 0
        for name, axis in self.axes.items():
            self._axis_indices[name] = idx
            if isinstance(axis, ContinuousAxis):
                self._continuous_axes.append((name, axis))
            elif isinstance(axis, CategoricalAxis):
                self._categorical_axes.append((name, axis))
            idx += 1

    @property
    def n_continuous(self) -> int:
        """Number of continuous dimensions."""
        return len(self._continuous_axes)

    @property
    def n_categorical(self) -> int:
        """Number of categorical dimensions."""
        return len(self._categorical_axes)

    @property
    def n_dimensions(self) -> int:
        """Total number of dimensions."""
        return len(self.axes)

    @abstractmethod
    def generate_samples(self) -> np.ndarray:
        """Generate samples in the unit hypercube [0,1]^d.

        Returns:
            Array of shape (n_samples, n_dimensions) with values in [0, 1]
        """
        pass

    def transform_samples(self, unit_samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Transform unit hypercube samples to actual parameter space.

        Args:
            unit_samples: Samples in [0,1]^d space

        Returns:
            Dictionary mapping axis names to transformed values
        """
        transformed = {}

        for name, axis in self.axes.items():
            idx = self._axis_indices[name]
            unit_values = unit_samples[:, idx]

            if isinstance(axis, ContinuousAxis):
                transformed[name] = self._transform_continuous(unit_values, axis)
            elif isinstance(axis, CategoricalAxis):
                transformed[name] = self._transform_categorical(unit_values, axis)

        return transformed

    def _transform_continuous(
        self, unit_values: np.ndarray, axis: ContinuousAxis
    ) -> np.ndarray:
        """Transform unit values to continuous distribution.

        Args:
            unit_values: Values in [0, 1]
            axis: Continuous axis configuration

        Returns:
            Transformed values according to the distribution
        """
        if axis.type == "uniform":
            return axis.min + unit_values * (axis.max - axis.min)

        elif axis.type == "truncated_normal":
            from scipy import stats

            a = (axis.min - axis.mu) / axis.sigma
            b = (axis.max - axis.mu) / axis.sigma
            dist = stats.truncnorm(a, b, loc=axis.mu, scale=axis.sigma)
            return dist.ppf(unit_values)

        elif axis.type == "beta":
            from scipy import stats

            dist = stats.beta(axis.alpha, axis.beta)
            return dist.ppf(unit_values)

        else:
            raise NotImplementedError(f"Distribution type {axis.type} not implemented")

    def _transform_categorical(
        self, unit_values: np.ndarray, axis: CategoricalAxis
    ) -> np.ndarray:
        """Transform unit values to categorical values.

        Args:
            unit_values: Values in [0, 1]
            axis: Categorical axis configuration

        Returns:
            Array of categorical values
        """
        n_categories = len(axis.values)

        if axis.weights is None:
            boundaries = np.linspace(0, 1, n_categories + 1)
        else:
            cumsum = np.cumsum([0] + axis.weights)
            boundaries = cumsum / cumsum[-1]

        indices = np.searchsorted(boundaries[1:], unit_values)
        indices = np.clip(indices, 0, n_categories - 1)

        return np.array([axis.values[i] for i in indices])

    def sample(self) -> SampleResult:
        """Generate samples and return a SampleResult.

        Returns:
            SampleResult containing samples and metadata
        """
        unit_samples = self.generate_samples()
        transformed_samples = self.transform_samples(unit_samples)

        result_array = np.empty((self.n_samples, self.n_dimensions), dtype=object)
        for name, values in transformed_samples.items():
            idx = self._axis_indices[name]
            result_array[:, idx] = values

        metadata = {
            "n_samples": self.n_samples,
            "n_dimensions": self.n_dimensions,
            "n_continuous": self.n_continuous,
            "n_categorical": self.n_categorical,
            "random_seed": self.random_seed,
            "axis_names": list(self.axes.keys()),
        }

        return SampleResult(samples=result_array, metadata=metadata)
