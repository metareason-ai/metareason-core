"""Sampling configuration models for MetaReason."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SamplingConfig(BaseModel):
    """Configuration for sampling methods and parameters."""

    method: Literal["latin_hypercube", "random", "sobol"] = Field(
        default="latin_hypercube",
        description="Sampling method to use for generating parameter " "combinations",
    )
    optimization_criterion: Optional[
        Literal["maximin", "correlation", "esi", "lloyd"]
    ] = Field(
        default="maximin",
        description="Optimization criterion for Latin Hypercube Sampling",
    )
    random_seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )
    stratified_by: Optional[List[str]] = Field(
        default=None,
        description="List of categorical axis names to ensure balanced " "sampling",
    )
    lhs_strength: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Strength of Latin Hypercube Sampling (1 or 2)",
    )
    lhs_scramble: bool = Field(
        default=True,
        description="Whether to randomly place samples within LHS cells",
    )
    batch_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum samples per batch for memory efficiency",
    )
    show_progress: bool = Field(
        default=True,
        description="Whether to show progress bar for large sample generations",
    )
    parallel_generation: bool = Field(
        default=False,
        description="Whether to use parallel processing for sample generation",
    )
    n_workers: Optional[int] = Field(
        default=None,
        description="Number of parallel workers (None uses CPU count)",
    )
    compute_quality_metrics: bool = Field(
        default=True,
        description="Whether to compute quality metrics after sampling",
    )

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError(
                f"Random seed must be non-negative, got {v}. "
                f"Suggestion: Use a positive integer like 42 or None for "
                f"random seed."
            )
        return v

    @field_validator("stratified_by")
    @classmethod
    def validate_stratified_by(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if not v:
                raise ValueError(
                    "If stratified_by is specified, it must contain at "
                    "least one axis name. Suggestion: Either omit "
                    "stratified_by or provide axis names like "
                    "['persona_clause']."
                )

            # Check for duplicates
            if len(set(v)) != len(v):
                duplicates = [x for x in v if v.count(x) > 1]
                raise ValueError(
                    f"Stratified axis names must be unique. Found "
                    f"duplicates: {list(set(duplicates))}. Suggestion: "
                    f"Remove duplicate axis names."
                )
        return v

    @field_validator("optimization_criterion")
    @classmethod
    def validate_optimization_criterion_compatibility(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        # Note: We can't access other fields in field_validator in Pydantic v2
        # This will be handled in model_validator if needed
        return v
