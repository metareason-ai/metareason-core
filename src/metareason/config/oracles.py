"""Oracle configuration models for MetaReason."""

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingSimilarityOracle(BaseModel):
    """Configuration for embedding similarity-based accuracy oracle."""

    type: Literal["embedding_similarity"] = "embedding_similarity"
    canonical_answer: str = Field(..., description="The expected correct answer")
    method: Literal[
        "cosine_similarity", "euclidean", "dot_product", "semantic_entropy"
    ] = Field(default="cosine_similarity", description="Similarity calculation method")
    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity threshold for pass/fail"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )
    embedding_adapter: Optional[str] = Field(
        default=None,
        description="LLM adapter to use for embedding generation (uses primary model adapter if not specified)",
    )
    batch_size: int = Field(
        default=32, ge=1, le=1000, description="Batch size for embedding processing"
    )
    use_vectorized: bool = Field(
        default=True,
        description="Use vectorized similarity calculations for performance",
    )
    parallel_workers: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Number of parallel workers for batch processing",
    )
    confidence_passthrough: bool = Field(
        default=False,
        description="Pass through confidence scores instead of binary threshold",
    )
    distribution_analysis: bool = Field(
        default=False, description="Include similarity distribution analysis in results"
    )

    @field_validator("canonical_answer")
    @classmethod
    def validate_canonical_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Canonical answer cannot be empty. Suggestion: "
                "Provide a clear, comprehensive expected answer."
            )
        if len(v.strip()) < 10:
            raise ValueError(
                f"Canonical answer seems too short ({len(v.strip())} "
                f"characters). Suggestion: Provide a more detailed "
                f"expected answer for better evaluation."
            )
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {v}. "
                f"Suggestion: Use a value like 0.85 for similarity "
                f"matching."
            )
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v > 100:
            # Warning for large batch sizes that might cause memory issues
            pass
        return v

    @field_validator("parallel_workers")
    @classmethod
    def validate_parallel_workers(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v > 16:
            # Warning for excessive parallelism
            pass
        return v

    @field_validator("embedding_adapter")
    @classmethod
    def validate_embedding_adapter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError(
                "Embedding adapter cannot be empty string. "
                "Suggestion: Use valid adapter name like 'openai' or remove field to use primary adapter."
            )
        return v


class LLMJudgeOracle(BaseModel):
    """Configuration for LLM-based judging oracle."""

    type: Literal["llm_judge"] = "llm_judge"
    rubric: str = Field(..., description="Evaluation rubric for the judge")
    judge_model: str = Field(default="gpt-4", description="LLM model to use as judge")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for judge model (0.0 for consistency)",
    )
    output_format: Literal["binary", "score", "structured"] = Field(
        default="binary", description="Format of judge output"
    )

    @field_validator("rubric")
    @classmethod
    def validate_rubric(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Rubric cannot be empty. Suggestion: Provide clear "
                "evaluation criteria for the judge."
            )

        # Check for numbered criteria (good practice)
        lines = v.strip().split("\n")
        if len(lines) == 1 and len(v.strip()) < 50:
            raise ValueError(
                "Rubric appears too brief. Suggestion: Provide "
                "detailed criteria, preferably numbered (1., 2., etc.)"
            )
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v > 1.0:
            # Warning rather than error - some use cases might want high creativity
            pass
        return v


class StatisticalCalibrationOracle(BaseModel):
    """Configuration for statistical confidence calibration oracle."""

    type: Literal["statistical_calibration"] = "statistical_calibration"
    expected_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Expected confidence level"
    )
    tolerance: float = Field(
        default=0.10,
        gt=0.0,
        le=0.5,
        description="Tolerance for confidence deviation",
    )
    calibration_method: Literal["platt_scaling", "isotonic_regression"] = Field(
        default="platt_scaling", description="Calibration method to use"
    )

    @field_validator("expected_confidence")
    @classmethod
    def validate_expected_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Expected confidence must be between 0.0 and 1.0, got "
                f"{v}. Suggestion: Use a value like 0.85 for 85% "
                f"confidence."
            )
        return v

    @field_validator("tolerance")
    @classmethod
    def validate_tolerance(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(
                f"Tolerance must be positive, got {v}. Suggestion: "
                f"Use a small positive value like 0.10."
            )
        if v > 0.5:
            raise ValueError(
                f"Tolerance seems too large ({v}). Suggestion: Use a "
                f"value between 0.01 and 0.20 for meaningful "
                f"calibration."
            )
        return v


class CustomOracle(BaseModel):
    """Configuration for custom oracle implementations."""

    type: Literal["custom"] = "custom"
    module: str = Field(..., description="Python module containing the oracle class")
    class_name: str = Field(..., description="Name of the oracle class")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Custom configuration parameters"
    )

    @field_validator("module")
    @classmethod
    def validate_module_format(cls, v: str) -> str:
        if not v or "." not in v:
            raise ValueError(
                f"Module path '{v}' appears invalid. Suggestion: Use "
                f"full module path like 'metareason.oracles.custom'"
            )
        return v

    @field_validator("class_name")
    @classmethod
    def validate_class_name_format(cls, v: str) -> str:
        if not v or not v[0].isupper():
            raise ValueError(
                f"Class name '{v}' should follow Python naming "
                f"conventions. Suggestion: Use PascalCase like "
                f"'CustomOracle'"
            )
        return v


# Type alias for all oracle configurations
OracleConfigType = Union[
    EmbeddingSimilarityOracle,
    LLMJudgeOracle,
    StatisticalCalibrationOracle,
    CustomOracle,
]


class OracleConfig(BaseModel):
    """Container for multiple oracle configurations."""

    accuracy: Optional[EmbeddingSimilarityOracle] = Field(
        None, description="Accuracy oracle"
    )
    explainability: Optional[LLMJudgeOracle] = Field(
        None, description="Explainability oracle"
    )
    confidence_calibration: Optional[StatisticalCalibrationOracle] = Field(
        None, description="Confidence calibration oracle"
    )
    custom_oracles: Optional[Dict[str, CustomOracle]] = Field(
        None, description="Custom oracle implementations"
    )

    @field_validator("custom_oracles")
    @classmethod
    def validate_custom_oracle_names(
        cls, v: Optional[Dict[str, CustomOracle]]
    ) -> Optional[Dict[str, CustomOracle]]:
        if v is not None:
            for name in v.keys():
                if not name or not name.replace("_", "").isalnum():
                    raise ValueError(
                        f"Custom oracle name '{name}' is invalid. "
                        f"Suggestion: Use alphanumeric names with "
                        f"underscores like 'regulatory_compliance'."
                    )
        return v
