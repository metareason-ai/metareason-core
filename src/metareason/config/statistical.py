"""Statistical configuration models for MetaReason."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PriorConfig(BaseModel):
    """Configuration for prior distribution."""
    
    alpha: float = Field(
        default=1.0,
        gt=0,
        description="Alpha parameter for Beta prior"
    )
    beta: float = Field(
        default=1.0,
        gt=0,
        description="Beta parameter for Beta prior"
    )
    
    @field_validator("alpha", "beta")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"Prior parameters must be positive, got {v}. "
                f"Suggestion: Use positive values like alpha=1.0, beta=1.0 "
                f"for uniform prior."
            )
        return v


class InferenceConfig(BaseModel):
    """Configuration for statistical inference."""
    
    method: Literal["mcmc", "variational", "exact"] = Field(
        default="mcmc",
        description="Inference method to use"
    )
    samples: int = Field(
        default=4000,
        ge=1000,
        le=100000,
        description="Number of samples for MCMC"
    )
    chains: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of MCMC chains"
    )
    target_accept: float = Field(
        default=0.8,
        ge=0.6,
        le=0.99,
        description="Target acceptance rate for MCMC"
    )
    
    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v: int) -> int:
        if v < 1000:
            raise ValueError(
                f"Number of samples ({v}) is too low for reliable inference. "
                f"Suggestion: Use at least 1000 samples, preferably 4000."
            )
        if v > 100000:
            raise ValueError(
                f"Number of samples ({v}) may be computationally expensive. "
                f"Suggestion: Consider using 10000 or fewer samples."
            )
        return v
    
    @field_validator("chains")
    @classmethod
    def validate_chains(cls, v: int) -> int:
        if v < 2:
            # Warning level - single chain is allowed but not recommended
            pass
        if v > 16:
            raise ValueError(
                f"Number of chains ({v}) is excessive. "
                f"Suggestion: Use 4-8 chains for good convergence diagnostics."
            )
        return v


class OutputConfig(BaseModel):
    """Configuration for statistical output."""
    
    credible_interval: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Credible interval level (e.g., 0.95 for 95%)"
    )
    hdi_method: Literal["shortest", "central"] = Field(
        default="shortest",
        description="Method for computing highest density interval"
    )
    
    @field_validator("credible_interval")
    @classmethod
    def validate_credible_interval(cls, v: float) -> float:
        if not 0.5 <= v <= 0.999:
            raise ValueError(
                f"Credible interval must be between 0.5 and 0.999, got {v}. "
                f"Suggestion: Use standard values like 0.95 (95%) or 0.89 (89%)."
            )
        return v


class StatisticalConfig(BaseModel):
    """Statistical configuration for MetaReason evaluations."""
    
    model: Literal["beta_binomial", "binomial", "gaussian"] = Field(
        default="beta_binomial",
        description="Statistical model to use"
    )
    prior: PriorConfig = Field(
        default_factory=PriorConfig,
        description="Prior distribution configuration"
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference method configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    
    @field_validator("model")
    @classmethod
    def validate_model_prior_compatibility(cls, v: str) -> str:
        # Add any model-specific validation here
        return v