"""Main configuration models for MetaReason evaluations."""

import re
from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .axes import AxisConfigType, CategoricalAxis, ContinuousAxis
from .oracles import OracleConfig
from .sampling import SamplingConfig
from .statistical import StatisticalConfig


class PipelineStep(BaseModel):
    """Self-contained pipeline step with template, model config, and axes."""

    # Template for this stage
    template: str = Field(..., description="Jinja2 template for this stage")

    # Model configuration
    adapter: str = Field(
        ..., description="Adapter type (openai, anthropic, ollama, etc.)"
    )
    model: str = Field(..., description="Specific model name")
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum output tokens"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    frequency_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    json_schema: Optional[str] = Field(
        default=None,
        description="Relative path to JSON schema file for structured output",
    )

    # Stage-specific axes
    axes: Dict[str, AxisConfigType] = Field(
        default_factory=dict, description="Variables that can vary for this stage"
    )

    @field_validator("adapter")
    @classmethod
    def validate_adapter_type(cls, v: str) -> str:
        if not v:
            raise ValueError("Adapter type cannot be empty")

        supported_adapters = {
            "openai",
            "anthropic",
            "ollama",
            "google",
            "azure_openai",
            "huggingface",
            "custom",
        }
        if v not in supported_adapters:
            raise ValueError(
                f"Adapter type '{v}' not supported. "
                f"Supported types: {sorted(supported_adapters)}"
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("JSON schema path cannot be empty string")

            # Basic validation for relative path format
            if v.startswith("/") or ":" in v:
                raise ValueError(
                    f"JSON schema path '{v}' must be a relative path. "
                    f"Use paths like 'schemas/response_format.json'"
                )

            # Ensure it ends with .json
            if not v.endswith(".json"):
                raise ValueError(f"JSON schema path '{v}' must end with '.json'")

        return v

    @field_validator("template")
    @classmethod
    def validate_template_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Template cannot be empty. Provide a "
                "Jinja2 template with {{variable}} placeholders."
            )

        if len(v.strip()) < 10:
            raise ValueError(
                "Template seems too short. Provide "
                "a more detailed template for meaningful evaluation."
            )

        return v

    @model_validator(mode="after")
    def validate_template_variables(self) -> "PipelineStep":
        """Validate that all template variables exist in axes."""
        # Extract variables from template using regex
        template_vars = set(re.findall(r"\{\{(\w+)\}\}", self.template))
        axes_vars = set(self.axes.keys())

        # Allow stage output variables (stage_N_output)
        stage_vars = {
            var for var in template_vars if re.match(r"stage_\d+_output", var)
        }
        template_vars = template_vars - stage_vars

        missing_vars = template_vars - axes_vars
        if missing_vars:
            raise ValueError(
                f"Template variables {sorted(missing_vars)} are not "
                f"defined in axes. Add these variables to "
                f"your axes or remove them from the template."
            )

        return self


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

    spec_id: str = Field(..., description="Unique identifier for the specification")
    pipeline: List[PipelineStep] = Field(
        ..., description="Pipeline steps for evaluation"
    )
    sampling: Optional[SamplingConfig] = Field(
        default_factory=SamplingConfig, description="Sampling strategy configuration"
    )
    n_variants: int = Field(
        default=1000,
        ge=10,
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

    @field_validator("spec_id")
    @classmethod
    def validate_spec_id_format(cls, v: str) -> str:
        if not v:
            raise ValueError("Spec ID cannot be empty.")

        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(
                f"Spec ID '{v}' should use lowercase letters, numbers, "
                f"and underscores only. Use format like "
                f"'iso_42001_compliance_check'."
            )

        if len(v) > 100:
            raise ValueError(
                f"Spec ID is too long ({len(v)} characters). "
                f"Keep under 100 characters for better "
                f"readability."
            )

        return v

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline_not_empty(cls, v: List[PipelineStep]) -> List[PipelineStep]:
        if not v:
            raise ValueError(
                "Pipeline cannot be empty. Define at least one pipeline step."
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
    def validate_stratified_sampling_axes(self) -> "EvaluationConfig":
        """Validate that stratified_by axes exist and are categorical."""
        if self.sampling and self.sampling.stratified_by:
            # Collect all axes from all pipeline steps
            all_axes = {}
            for step in self.pipeline:
                all_axes.update(step.axes)

            for axis_name in self.sampling.stratified_by:
                if axis_name not in all_axes:
                    raise ValueError(
                        f"Stratified sampling axis '{axis_name}' not found "
                        f"in any pipeline step axes. Ensure all stratified_by "
                        f"axes are defined in pipeline step axes."
                    )

                axis_config = all_axes[axis_name]
                if not isinstance(axis_config, CategoricalAxis):
                    raise ValueError(
                        f"Stratified sampling axis '{axis_name}' must be "
                        f"categorical, got {type(axis_config).__name__}. "
                        f"Only categorical axes can be used for "
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
        # Count categorical combinations across all pipeline steps
        total_categorical_combinations = 1
        total_continuous_axes = 0

        for step in self.pipeline:
            for axis_config in step.axes.values():
                if isinstance(axis_config, CategoricalAxis):
                    total_categorical_combinations *= len(axis_config.values)
                elif isinstance(axis_config, ContinuousAxis):
                    total_continuous_axes += 1

        # This is a warning-level check, not a hard requirement
        # Skip validation for now as requirements may vary

        return self
