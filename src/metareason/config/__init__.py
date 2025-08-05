"""Configuration management for MetaReason."""

from .axes import AxisConfig, AxisConfigType, CategoricalAxis, ContinuousAxis
from .distributions import (
    BetaDistributionConfig,
    CustomDistributionConfig,
    DistributionConfig,
    DistributionConfigType,
    TruncatedNormalConfig,
    UniformDistributionConfig,
)
from .loader import load_yaml_config, load_yaml_configs, validate_yaml_string
from .models import DomainContext, EvaluationConfig, Metadata
from .oracles import (
    CustomOracle,
    EmbeddingSimilarityOracle,
    LLMJudgeOracle,
    OracleConfig,
    OracleConfigType,
    StatisticalCalibrationOracle,
)
from .sampling import SamplingConfig
from .statistical import InferenceConfig, OutputConfig, PriorConfig, StatisticalConfig
from .validator import ValidationReport, validate_yaml_directory, validate_yaml_file

__all__ = [
    # Main configuration
    "EvaluationConfig",
    "DomainContext",
    "Metadata",
    # Axes
    "AxisConfig",
    "AxisConfigType",
    "CategoricalAxis",
    "ContinuousAxis",
    # Distributions
    "DistributionConfig",
    "DistributionConfigType",
    "TruncatedNormalConfig",
    "BetaDistributionConfig",
    "UniformDistributionConfig",
    "CustomDistributionConfig",
    # Oracles
    "OracleConfig",
    "OracleConfigType",
    "EmbeddingSimilarityOracle",
    "LLMJudgeOracle",
    "StatisticalCalibrationOracle",
    "CustomOracle",
    # Sampling
    "SamplingConfig",
    # Statistical
    "StatisticalConfig",
    "PriorConfig",
    "InferenceConfig",
    "OutputConfig",
    # Loaders
    "load_yaml_config",
    "load_yaml_configs",
    "validate_yaml_string",
    # Validators
    "ValidationReport",
    "validate_yaml_file",
    "validate_yaml_directory",
]
