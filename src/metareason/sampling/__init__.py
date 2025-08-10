"""Sampling strategies for LLM evaluation."""

from .base import BaseSampler, SampleResult
from .factory import create_sampler
from .lhs import LatinHypercubeSampler
from .metrics import (
    compute_all_metrics,
    compute_correlation_metrics,
    compute_coverage_metrics,
    compute_discrepancy,
    compute_distance_metrics,
    compute_uniformity_metrics,
    validate_against_theoretical,
    visualize_metrics_comparison,
    visualize_sample_distribution,
)
from .optimization import (
    BaseOptimizer,
    CorrelationMinimizer,
    CustomOptimizer,
    ESIOptimizer,
    MaximinOptimizer,
    benchmark_optimizers,
)
from .utils import (
    decode_categorical_values,
    denormalize_samples,
    encode_categorical_values,
    load_samples,
    normalize_samples,
    parallel_sample_generation,
    save_samples,
    stratified_sampling,
)

__all__ = [
    "BaseSampler",
    "SampleResult",
    "create_sampler",
    "LatinHypercubeSampler",
    "BaseOptimizer",
    "MaximinOptimizer",
    "CorrelationMinimizer",
    "ESIOptimizer",
    "CustomOptimizer",
    "benchmark_optimizers",
    "compute_discrepancy",
    "compute_correlation_metrics",
    "compute_distance_metrics",
    "compute_uniformity_metrics",
    "compute_coverage_metrics",
    "validate_against_theoretical",
    "compute_all_metrics",
    "visualize_sample_distribution",
    "visualize_metrics_comparison",
    "encode_categorical_values",
    "decode_categorical_values",
    "normalize_samples",
    "denormalize_samples",
    "save_samples",
    "load_samples",
    "parallel_sample_generation",
    "stratified_sampling",
]
