"""JSON response models for LLM judge evaluations."""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


class BinaryJudgeResponse(BaseModel):
    """Binary pass/fail judge response."""

    score: int = Field(..., description="Binary score: 1 for pass, 0 for fail")
    reasoning: str = Field(..., description="Explanation of the judgment")

    @field_validator("score")
    @classmethod
    def validate_binary_score(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("Binary score must be 0 or 1")
        return v


class NumericJudgeResponse(BaseModel):
    """Numeric score judge response."""

    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score 0.0-1.0")
    reasoning: str = Field(..., description="Explanation of the judgment")


class StructuredJudgeResponse(BaseModel):
    """Structured multi-dimensional judge response."""

    score: float = Field(..., ge=0.0, le=1.0, description="Overall score 0.0-1.0")
    reasoning: str = Field(..., description="Overall explanation")
    dimensions: Dict[str, float] = Field(..., description="Individual dimension scores")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional structured details"
    )

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: Dict[str, float]) -> Dict[str, float]:
        for name, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Dimension '{name}' score must be 0.0-1.0")
        return v


# Union type for all judge response formats
JudgeResponseType = Union[
    BinaryJudgeResponse, NumericJudgeResponse, StructuredJudgeResponse
]


class JudgeResult(BaseModel):
    """Complete result from a judge evaluation."""

    response: JudgeResponseType = Field(..., description="Parsed judge response")
    raw_response: str = Field(..., description="Raw LLM response")
    judge_model: str = Field(..., description="Model used for judging")
    temperature: float = Field(..., description="Temperature used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ConsistencyMeasurement(BaseModel):
    """Measurement of judge consistency across multiple evaluations."""

    judge_model: str = Field(..., description="Judge model evaluated")
    consistency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Consistency score 0.0-1.0"
    )
    variance: float = Field(..., description="Score variance across evaluations")
    agreement_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agreement rate for binary judgments"
    )
    sample_size: int = Field(..., gt=0, description="Number of evaluations")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BiasDetectionResult(BaseModel):
    """Result from bias detection analysis."""

    bias_type: str = Field(..., description="Type of bias detected")
    severity: float = Field(
        ..., ge=0.0, le=1.0, description="Bias severity score 0.0-1.0"
    )
    affected_categories: list[str] = Field(
        ..., description="Categories affected by bias"
    )
    evidence: Dict[str, Any] = Field(..., description="Statistical evidence")
    recommendations: list[str] = Field(
        ..., description="Recommendations to mitigate bias"
    )


class CalibrationResult(BaseModel):
    """Result from judge calibration against human judgments."""

    judge_model: str = Field(..., description="Judge model evaluated")
    calibration_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibration quality 0.0-1.0"
    )
    correlation: float = Field(
        ..., ge=-1.0, le=1.0, description="Correlation with human judgments"
    )
    sample_size: int = Field(..., gt=0, description="Number of comparisons")
    confidence_intervals: Dict[str, tuple[float, float]] = Field(
        ..., description="95% confidence intervals for key metrics"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
