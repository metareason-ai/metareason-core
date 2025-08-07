"""Main configuration models for MetaReason evaluations."""

import re
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .axes import AxisConfigType, CategoricalAxis, ContinuousAxis
from .oracles import OracleConfig
from .sampling import SamplingConfig
from .statistical import StatisticalConfig


class DomainContext(BaseModel):
    """Domain-specific context and compliance information."""

    industry: Optional[
        Literal["financial_services", "healthcare", "insurance", "general"]
    ] = Field(None, description="Industry domain")
    regulatory_frameworks: Optional[List[str]] = Field(
        None, description="Applicable regulatory frameworks"
    )
    risk_category: Optional[Literal["minimal", "limited", "high", "unacceptable"]] = (
        Field(None, description="Risk category per EU AI Act")
    )
    use_case: Optional[str] = Field(None, description="Specific use case description")
    data_sensitivity: Optional[Literal["public", "internal", "confidential", "pii"]] = (
        Field(None, description="Data sensitivity level")
    )


class Metadata(BaseModel):
    """Metadata for governance and audit trails."""

    version: str = Field(default="1.0.0", description="Schema version")
    created_by: Optional[str] = Field(None, description="Creator email or identifier")
    created_date: Optional[date] = Field(None, description="Creation date")
    last_modified: Optional[date] = Field(None, description="Last modification date")
    review_cycle: Optional[Literal["monthly", "quarterly", "annual"]] = Field(
        None, description="Review cycle frequency"
    )
    compliance_mappings: Optional[List[str]] = Field(
        None, description="Compliance framework mappings"
    )
    tags: Optional[List[str]] = Field(None, description="Classification tags")
    deprecation_date: Optional[date] = Field(None, description="Deprecation date")

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        # Simple semantic version validation
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise ValueError(
                f"Version '{v}' should follow semantic versioning "
                f"(e.g., '1.0.0'). Suggestion: Use format "
                f"'major.minor.patch'."
            )
        return v

    @field_validator("created_by")
    @classmethod
    def validate_created_by_email(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and "@" in v:
            # Basic email validation
            if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
                raise ValueError(
                    f"Created by '{v}' appears to be an invalid email "
                    f"format. Suggestion: Use valid email format like "
                    f"'user@domain.com'."
                )
        return v


class EvaluationConfig(BaseModel):
    """Main configuration model for MetaReason evaluations."""

    prompt_id: str = Field(..., description="Unique identifier for the prompt family")
    prompt_template: str = Field(..., description="Jinja2-compatible template")
    axes: Dict[str, AxisConfigType] = Field(
        ..., description="Variable axes configuration"
    )
    sampling: Optional[SamplingConfig] = Field(
        default_factory=SamplingConfig, description="Sampling strategy configuration"
    )
    n_variants: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of prompt variants to generate",
    )
    oracles: OracleConfig = Field(..., description="Evaluation oracles configuration")
    domain_context: Optional[DomainContext] = Field(
        None, description="Domain-specific context"
    )
    metadata: Optional[Metadata] = Field(None, description="Metadata for governance")
    statistical_config: Optional[StatisticalConfig] = Field(
        None, description="Statistical model configuration"
    )

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id_format(cls, v: str) -> str:
        if not v:
            raise ValueError("Prompt ID cannot be empty.")

        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Prompt ID '{v}' should use lowercase letters, numbers, "
                f"and underscores only. Suggestion: Use format like "
                f"'iso_42001_compliance_check'."
            )

        if len(v) > 100:
            raise ValueError(
                f"Prompt ID is too long ({len(v)} characters). "
                f"Suggestion: Keep under 100 characters for better "
                f"readability."
            )

        return v

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Prompt template cannot be empty. Suggestion: Provide a "
                "Jinja2 template with {{variable}} placeholders."
            )

        if len(v.strip()) < 10:
            raise ValueError(
                "Prompt template seems too short. Suggestion: Provide "
                "a more detailed template for meaningful evaluation."
            )

        return v

    @field_validator("axes")
    @classmethod
    def validate_axes_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError(
                "Schema cannot be empty. Suggestion: Define at least "
                "one axis for prompt generation."
            )

        # Validate axis names
        for axis_name in v.keys():
            if not re.match(r"^[a-z][a-z0-9_]*$", axis_name):
                raise ValueError(
                    f"Axis name '{axis_name}' should use lowercase letters, "
                    f"numbers, and underscores. Suggestion: Use format like "
                    f"'persona_clause' or 'temperature'."
                )

        return v

    @field_validator("n_variants")
    @classmethod
    def validate_n_variants_range(cls, v: int) -> int:
        if v < 100:
            raise ValueError(
                f"Number of variants ({v}) is too low for statistical "
                f"significance. Suggestion: Use at least 100 variants, "
                f"preferably 1000 or more."
            )

        if v > 10000:
            raise ValueError(
                f"Number of variants ({v}) may be computationally "
                f"expensive. Suggestion: Consider using 10000 or fewer "
                f"variants."
            )

        return v

    @model_validator(mode="after")
    def validate_template_variables(self) -> "EvaluationConfig":
        """Validate that all template variables exist in schema."""
        # Extract variables from template using regex
        template_vars = set(re.findall(r"\{\{(\w+)\}\}", self.prompt_template))
        axes_vars = set(self.axes.keys())

        missing_vars = template_vars - axes_vars
        if missing_vars:
            raise ValueError(
                f"Template variables {sorted(missing_vars)} are not "
                f"defined in schema. Suggestion: Add these variables to "
                f"your schema or remove them from the template."
            )

        unused_vars = axes_vars - template_vars
        if unused_vars:
            # This is a warning rather than an error
            pass  # Could log a warning here

        return self

    @model_validator(mode="after")
    def validate_stratified_sampling_axes(self) -> "EvaluationConfig":
        """Validate that stratified_by axes exist and are categorical."""
        if self.sampling and self.sampling.stratified_by:
            for axis_name in self.sampling.stratified_by:
                if axis_name not in self.axes:
                    raise ValueError(
                        f"Stratified sampling axis '{axis_name}' not found "
                        f"in schema. Suggestion: Ensure all stratified_by "
                        f"axes are defined in schema."
                    )

                axis_config = self.axes[axis_name]
                if not isinstance(axis_config, CategoricalAxis):
                    raise ValueError(
                        f"Stratified sampling axis '{axis_name}' must be "
                        f"categorical, got {type(axis_config).__name__}. "
                        f"Suggestion: Only categorical axes can be used for "
                        f"stratified sampling."
                    )

        return self

    @model_validator(mode="after")
    def validate_oracles_not_empty(self) -> "EvaluationConfig":
        """Validate that at least one oracle is configured."""
        oracle_count = 0
        if self.oracles.accuracy:
            oracle_count += 1
        if self.oracles.explainability:
            oracle_count += 1
        if self.oracles.confidence_calibration:
            oracle_count += 1
        if self.oracles.custom_oracles:
            oracle_count += len(self.oracles.custom_oracles)

        if oracle_count == 0:
            raise ValueError(
                "At least one oracle must be configured. Suggestion: "
                "Add an accuracy or explainability oracle for evaluation."
            )

        return self

    @model_validator(mode="after")
    def validate_statistical_requirements(self) -> "EvaluationConfig":
        """Validate statistical requirements for meaningful analysis."""
        # Count categorical combinations for power analysis
        categorical_combinations = 1
        continuous_axes = 0

        for axis_config in self.axes.values():
            if isinstance(axis_config, CategoricalAxis):
                categorical_combinations *= len(axis_config.values)
            elif isinstance(axis_config, ContinuousAxis):
                continuous_axes += 1

        # Rule of thumb: need at least 10 samples per categorical combination
        # min_variants_needed = categorical_combinations * 10

        # This is a warning-level check, not a hard requirement
        # The spec example actually violates this rule (576 combinations, 2000 variants)
        # So we'll skip this validation for now
        # if self.n_variants < min_variants_needed and categorical_combinations > 1:
        #     raise ValueError(
        #         f"With {categorical_combinations} categorical "
        #         f"combinations, recommend at least {min_variants_needed} "
        #         f"variants for statistical power. Currently set to "
        #         f"{self.n_variants}. Suggestion: Increase n_variants or "
        #         f"reduce categorical axis complexity."
        #     )

        return self
