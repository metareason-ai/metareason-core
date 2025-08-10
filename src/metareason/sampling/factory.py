"""Factory functions for creating samplers."""

from typing import Dict, Optional

from ..config import AxisConfigType, SamplingConfig
from .base import BaseSampler
from .lhs import LatinHypercubeSampler


def create_sampler(
    axes: Dict[str, AxisConfigType],
    sampling_config: Optional[SamplingConfig] = None,
    n_samples: int = 1000,
    random_seed: Optional[int] = None,
) -> BaseSampler:
    """Create a sampler based on configuration.

    Args:
        axes: Dictionary of axis configurations
        sampling_config: Sampling configuration (uses defaults if None)
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        Configured sampler instance
    """
    # Use defaults if no sampling config provided
    if sampling_config is None:
        sampling_config = SamplingConfig()

    # Override random seed if provided
    if random_seed is not None:
        sampling_config.random_seed = random_seed

    # Currently only Latin Hypercube sampling is implemented
    if sampling_config.method == "latin_hypercube":
        # Note: stratified_by is not yet supported by LatinHypercubeSampler
        return LatinHypercubeSampler(
            axes=axes,
            n_samples=n_samples,
            random_seed=sampling_config.random_seed,
            optimization=sampling_config.optimization_criterion,
            scramble=sampling_config.lhs_scramble,
            strength=sampling_config.lhs_strength,
            batch_size=sampling_config.batch_size,
            show_progress=sampling_config.show_progress,
        )
    else:
        raise ValueError(f"Unsupported sampling method: {sampling_config.method}")
