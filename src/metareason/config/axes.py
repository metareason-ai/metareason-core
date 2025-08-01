"""Axis configuration models for MetaReason schema."""

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AxisConfig(BaseModel, ABC):
    """Base class for all axis configurations."""

    type: str

    @abstractmethod
    def validate_axis_parameters(self) -> None:
        """Validate axis-specific parameters."""
        pass

    def model_post_init(self, __context: Any) -> None:
        """Validate after model initialization."""
        self.validate_axis_parameters()


class CategoricalAxis(AxisConfig):
    """Configuration for categorical axes with discrete values."""

    type: Literal["categorical"] = "categorical"
    values: List[str] = Field(
        ..., min_length=1, description="List of categorical values"
    )
    weights: Optional[List[float]] = Field(
        None,
        description="Optional probability weights for each value (must sum to 1.0)",
    )

    @field_validator("values")
    @classmethod
    def validate_values_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError(
                "Categorical axis must have at least one value. "
                "Suggestion: Add at least one string value to the "
                "'values' list."
            )
        if len(set(v)) != len(v):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"Categorical values must be unique. Found duplicates: "
                f"{list(set(duplicates))}. Suggestion: Remove duplicate "
                f"values from the list."
            )
        return v

    @field_validator("weights")
    @classmethod
    def validate_weights_positive(
        cls, v: Optional[List[float]]
    ) -> Optional[List[float]]:
        if v is not None:
            if any(w < 0 for w in v):
                negative_weights = [w for w in v if w < 0]
                raise ValueError(
                    f"All weights must be non-negative. Found negative "
                    f"weights: {negative_weights}. Suggestion: Use only "
                    f"positive values that sum to 1.0."
                )
        return v

    @model_validator(mode="after")
    def validate_weights_consistency(self) -> "CategoricalAxis":
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of values ({len(self.values)}). Suggestion: "
                    f"Provide one weight for each categorical value, or "
                    f"omit weights for uniform distribution."
                )

            weight_sum = sum(self.weights)
            tolerance = 0.001
            if abs(weight_sum - 1.0) > tolerance:
                raise ValueError(
                    f"Weights must sum to 1.0 (±{tolerance}), got "
                    f"{weight_sum:.6f}. Suggestion: Adjust weights so they "
                    f"sum to exactly 1.0. Current weights: {self.weights}"
                )
        return self

    def validate_axis_parameters(self) -> None:
        """Validate categorical axis specific parameters."""
        # Best practice validation
        if len(self.values) > 10:
            # This is a warning-level validation, could be logged instead
            pass  # Could add warning logging here

        if len(self.values) < 3:
            # This is a recommendation, not a hard requirement
            pass  # Could add info logging here


class ContinuousAxis(AxisConfig):
    """Configuration for continuous axes using probability distributions."""

    type: Literal["truncated_normal", "beta", "uniform", "custom"]
    mu: Optional[float] = Field(
        None, description="Mean (for truncated_normal)"
    )
    sigma: Optional[float] = Field(
        None, gt=0, description="Standard deviation (for truncated_normal)"
    )
    min: Optional[float] = Field(
        None, description="Minimum value (for truncated_normal, uniform)"
    )
    max: Optional[float] = Field(
        None, description="Maximum value (for truncated_normal, uniform)"
    )
    alpha: Optional[float] = Field(
        None, gt=0, description="Alpha parameter (for beta)"
    )
    beta: Optional[float] = Field(
        None, gt=0, description="Beta parameter (for beta)"
    )
    module: Optional[str] = Field(
        None, description="Module path (for custom)"
    )
    class_name: Optional[str] = Field(
        None, description="Class name (for custom)"
    )
    config: Optional[dict] = Field(
        None, description="Custom config (for custom)"
    )

    @model_validator(mode="after")
    def validate_distribution_parameters(self) -> "ContinuousAxis":
        """Validate required parameters for each distribution type."""
        if self.type == "truncated_normal":
            missing = []
            if self.mu is None:
                missing.append("mu")
            if self.sigma is None:
                missing.append("sigma")
            if self.min is None:
                missing.append("min")
            if self.max is None:
                missing.append("max")

            if missing:
                raise ValueError(
                    f"Truncated normal distribution requires: "
                    f"{', '.join(missing)}. Suggestion: Add missing "
                    f"parameters, e.g., mu: 0.7, sigma: 0.1, "
                    f"min: 0.0, max: 1.0"
                )

            if self.sigma is not None and self.sigma <= 0:
                raise ValueError(
                    f"Standard deviation (sigma) must be positive, got "
                    f"{self.sigma}. Suggestion: Use a positive value "
                    f"like sigma: 0.1"
                )

            if (self.min is not None and self.max is not None 
                    and self.min >= self.max):
                raise ValueError(
                    f"Minimum ({self.min}) must be less than maximum "
                    f"({self.max}). Suggestion: Ensure min < max for "
                    f"proper bounds."
                )

        elif self.type == "beta":
            missing = []
            if self.alpha is None:
                missing.append("alpha")
            if self.beta is None:
                missing.append("beta")

            if missing:
                raise ValueError(
                    f"Beta distribution requires: {', '.join(missing)}. "
                    f"Suggestion: Add missing parameters, e.g., "
                    f"alpha: 2.0, beta: 5.0"
                )

            if self.alpha is not None and self.alpha <= 0:
                raise ValueError(
                    f"Alpha parameter must be positive, got "
                    f"{self.alpha}. Suggestion: Use a positive value "
                    f"like alpha: 2.0"
                )

            if self.beta is not None and self.beta <= 0:
                raise ValueError(
                    f"Beta parameter must be positive, got {self.beta}. "
                    f"Suggestion: Use a positive value like beta: 5.0"
                )

        elif self.type == "uniform":
            missing = []
            if self.min is None:
                missing.append("min")
            if self.max is None:
                missing.append("max")

            if missing:
                raise ValueError(
                    f"Uniform distribution requires: {', '.join(missing)}. "
                    f"Suggestion: Add missing parameters, e.g., "
                    f"min: 0.0, max: 1.0"
                )

            if (self.min is not None and self.max is not None 
                    and self.min >= self.max):
                raise ValueError(
                    f"Minimum ({self.min}) must be less than maximum "
                    f"({self.max}). Suggestion: Ensure min < max for "
                    f"uniform distribution."
                )

        elif self.type == "custom":
            missing = []
            if self.module is None:
                missing.append("module")
            if self.class_name is None:
                missing.append("class_name")

            if missing:
                raise ValueError(
                    f"Custom distribution requires: {', '.join(missing)}. "
                    f"Suggestion: Specify module and class_name for "
                    f"custom distributions."
                )

        return self

    def validate_axis_parameters(self) -> None:
        """Validate continuous axis specific parameters."""
        pass


# Type alias for all axis configurations
AxisConfigType = Union[CategoricalAxis, ContinuousAxis]
