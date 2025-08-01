"""Distribution configuration models for MetaReason."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class DistributionConfig(BaseModel, ABC):
    """Base class for all distribution configurations."""

    type: str

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate distribution-specific parameters."""
        pass

    def model_post_init(self, __context: Any) -> None:
        """Validate after model initialization."""
        self.validate_parameters()


class TruncatedNormalConfig(DistributionConfig):
    """Configuration for truncated normal distribution."""

    type: Literal["truncated_normal"] = "truncated_normal"
    mu: float = Field(..., description="Mean of the distribution")
    sigma: float = Field(
        ..., gt=0, description="Standard deviation (must be positive)"
    )
    min: float = Field(..., description="Minimum value (truncation bound)")
    max: float = Field(..., description="Maximum value (truncation bound)")

    @field_validator("sigma")
    @classmethod
    def validate_sigma_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Standard deviation (sigma) must be positive")
        return v

    @model_validator(mode="after")
    def validate_bounds(self) -> "TruncatedNormalConfig":
        if self.min >= self.max:
            raise ValueError(
                f"Minimum value ({self.min}) must be less than maximum "
                f"value ({self.max}). Suggestion: Check that min < max "
                f"for proper truncation bounds."
            )
        return self

    def validate_parameters(self) -> None:
        """Validate truncated normal specific parameters."""
        # Additional validation beyond field validators if needed
        pass


class BetaDistributionConfig(DistributionConfig):
    """Configuration for beta distribution."""

    type: Literal["beta"] = "beta"
    alpha: float = Field(
        ..., gt=0, description="Alpha parameter (must be positive)"
    )
    beta: float = Field(
        ..., gt=0, description="Beta parameter (must be positive)"
    )

    @field_validator("alpha", "beta")
    @classmethod
    def validate_positive_parameters(cls, v: float, info) -> float:
        if v <= 0:
            param_name = info.field_name
            raise ValueError(
                f"Beta distribution parameter '{param_name}' must be "
                f"positive, got {v}. Suggestion: Use values > 0, e.g., "
                f"alpha=2.0, beta=5.0 for a right-skewed distribution."
            )
        return v

    def validate_parameters(self) -> None:
        """Validate beta distribution specific parameters."""
        pass


class UniformDistributionConfig(DistributionConfig):
    """Configuration for uniform distribution."""

    type: Literal["uniform"] = "uniform"
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")

    @model_validator(mode="after")
    def validate_bounds(self) -> "UniformDistributionConfig":
        if self.min >= self.max:
            raise ValueError(
                f"Minimum value ({self.min}) must be less than maximum "
                f"value ({self.max}). Suggestion: Ensure min < max, "
                f"e.g., min: 0.0, max: 1.0"
            )
        return self

    def validate_parameters(self) -> None:
        """Validate uniform distribution specific parameters."""
        pass


class CustomDistributionConfig(DistributionConfig):
    """Configuration for custom distribution implementations."""

    type: Literal["custom"] = "custom"
    module: str = Field(
        ..., description="Python module containing the distribution class"
    )
    class_name: str = Field(..., description="Name of the distribution class")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration parameters for the distribution",
    )

    @field_validator("module")
    @classmethod
    def validate_module_format(cls, v: str) -> str:
        if not v or "." not in v:
            raise ValueError(
                f"Module path '{v}' appears invalid. Suggestion: Use "
                f"full module path like 'metareason.distributions.custom'"
            )
        return v

    @field_validator("class_name")
    @classmethod
    def validate_class_name_format(cls, v: str) -> str:
        if not v or not v[0].isupper():
            raise ValueError(
                f"Class name '{v}' should follow Python naming "
                f"conventions. Suggestion: Use PascalCase like "
                f"'CustomDistribution'"
            )
        return v

    def validate_parameters(self) -> None:
        """Validate custom distribution parameters."""
        pass


# Type alias for all distribution configurations
DistributionConfigType = Union[
    TruncatedNormalConfig,
    BetaDistributionConfig,
    UniformDistributionConfig,
    CustomDistributionConfig,
]
