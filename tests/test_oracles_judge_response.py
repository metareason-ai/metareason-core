"""Tests for LLM judge response models."""

import pytest

from metareason.oracles.judge_response import (
    BiasDetectionResult,
    BinaryJudgeResponse,
    CalibrationResult,
    ConsistencyMeasurement,
    JudgeResult,
    NumericJudgeResponse,
    StructuredJudgeResponse,
)


class TestBinaryJudgeResponse:
    """Test binary judge response model."""

    def test_valid_binary_response(self):
        """Test valid binary response creation."""
        response = BinaryJudgeResponse(score=1, reasoning="Response meets all criteria")
        assert response.score == 1
        assert response.reasoning == "Response meets all criteria"

    def test_binary_score_validation(self):
        """Test binary score validation."""
        # Valid scores
        BinaryJudgeResponse(score=0, reasoning="Fail")
        BinaryJudgeResponse(score=1, reasoning="Pass")

        # Invalid scores
        with pytest.raises(ValueError, match="Binary score must be 0 or 1"):
            BinaryJudgeResponse(score=2, reasoning="Invalid")

        with pytest.raises(ValueError, match="Binary score must be 0 or 1"):
            BinaryJudgeResponse(score=-1, reasoning="Invalid")


class TestNumericJudgeResponse:
    """Test numeric judge response model."""

    def test_valid_numeric_response(self):
        """Test valid numeric response creation."""
        response = NumericJudgeResponse(
            score=0.85, reasoning="Good response with minor issues"
        )
        assert response.score == 0.85
        assert response.reasoning == "Good response with minor issues"

    def test_numeric_score_bounds(self):
        """Test numeric score bounds validation."""
        # Valid scores
        NumericJudgeResponse(score=0.0, reasoning="Minimum")
        NumericJudgeResponse(score=1.0, reasoning="Maximum")
        NumericJudgeResponse(score=0.5, reasoning="Middle")

        # Invalid scores
        with pytest.raises(ValueError):
            NumericJudgeResponse(score=-0.1, reasoning="Below minimum")

        with pytest.raises(ValueError):
            NumericJudgeResponse(score=1.1, reasoning="Above maximum")


class TestStructuredJudgeResponse:
    """Test structured judge response model."""

    def test_valid_structured_response(self):
        """Test valid structured response creation."""
        response = StructuredJudgeResponse(
            score=0.8,
            reasoning="Overall good performance",
            dimensions={"accuracy": 0.9, "clarity": 0.7, "completeness": 0.8},
            details={
                "strengths": ["accurate", "well-structured"],
                "improvements": ["could be clearer"],
            },
        )

        assert response.score == 0.8
        assert response.reasoning == "Overall good performance"
        assert response.dimensions["accuracy"] == 0.9
        assert "accurate" in response.details["strengths"]

    def test_dimension_score_validation(self):
        """Test dimension score validation."""
        # Valid dimensions
        StructuredJudgeResponse(score=0.8, reasoning="Good", dimensions={"test": 0.0})

        StructuredJudgeResponse(score=0.8, reasoning="Good", dimensions={"test": 1.0})

        # Invalid dimension scores
        with pytest.raises(ValueError, match="Dimension 'test' score must be 0.0-1.0"):
            StructuredJudgeResponse(
                score=0.8, reasoning="Good", dimensions={"test": -0.1}
            )

        with pytest.raises(ValueError, match="Dimension 'test' score must be 0.0-1.0"):
            StructuredJudgeResponse(
                score=0.8, reasoning="Good", dimensions={"test": 1.1}
            )

    def test_optional_details(self):
        """Test optional details field."""
        response = StructuredJudgeResponse(
            score=0.8, reasoning="Good", dimensions={"test": 0.8}
        )

        assert response.details is None


class TestJudgeResult:
    """Test judge result model."""

    def test_judge_result_creation(self):
        """Test judge result creation."""
        binary_response = BinaryJudgeResponse(score=1, reasoning="Pass")

        result = JudgeResult(
            response=binary_response,
            raw_response='{"score": 1, "reasoning": "Pass"}',
            judge_model="gpt-4",
            temperature=0.0,
            metadata={"parsing_method": "json"},
        )

        assert result.response == binary_response
        assert result.judge_model == "gpt-4"
        assert result.temperature == 0.0
        assert result.metadata["parsing_method"] == "json"

    def test_different_response_types(self):
        """Test judge result with different response types."""
        # Binary response
        binary_response = BinaryJudgeResponse(score=1, reasoning="Pass")
        binary_result = JudgeResult(
            response=binary_response,
            raw_response="raw",
            judge_model="gpt-4",
            temperature=0.0,
        )
        assert isinstance(binary_result.response, BinaryJudgeResponse)

        # Numeric response
        numeric_response = NumericJudgeResponse(score=0.8, reasoning="Good")
        numeric_result = JudgeResult(
            response=numeric_response,
            raw_response="raw",
            judge_model="gpt-4",
            temperature=0.0,
        )
        assert isinstance(numeric_result.response, NumericJudgeResponse)

        # Structured response
        structured_response = StructuredJudgeResponse(
            score=0.8, reasoning="Good", dimensions={"test": 0.8}
        )
        structured_result = JudgeResult(
            response=structured_response,
            raw_response="raw",
            judge_model="gpt-4",
            temperature=0.0,
        )
        assert isinstance(structured_result.response, StructuredJudgeResponse)


class TestConsistencyMeasurement:
    """Test consistency measurement model."""

    def test_consistency_measurement_creation(self):
        """Test consistency measurement creation."""
        measurement = ConsistencyMeasurement(
            judge_model="gpt-4",
            consistency_score=0.85,
            variance=0.02,
            agreement_rate=0.9,
            sample_size=10,
            metadata={"test_info": "value"},
        )

        assert measurement.judge_model == "gpt-4"
        assert measurement.consistency_score == 0.85
        assert measurement.variance == 0.02
        assert measurement.agreement_rate == 0.9
        assert measurement.sample_size == 10

    def test_optional_agreement_rate(self):
        """Test optional agreement rate."""
        measurement = ConsistencyMeasurement(
            judge_model="gpt-4", consistency_score=0.85, variance=0.02, sample_size=10
        )

        assert measurement.agreement_rate is None

    def test_score_bounds_validation(self):
        """Test score bounds validation."""
        # Valid scores
        ConsistencyMeasurement(
            judge_model="gpt-4", consistency_score=0.0, variance=0.1, sample_size=5
        )

        ConsistencyMeasurement(
            judge_model="gpt-4", consistency_score=1.0, variance=0.1, sample_size=5
        )

        # Invalid consistency scores
        with pytest.raises(ValueError):
            ConsistencyMeasurement(
                judge_model="gpt-4", consistency_score=-0.1, variance=0.1, sample_size=5
            )

        with pytest.raises(ValueError):
            ConsistencyMeasurement(
                judge_model="gpt-4", consistency_score=1.1, variance=0.1, sample_size=5
            )


class TestBiasDetectionResult:
    """Test bias detection result model."""

    def test_bias_detection_creation(self):
        """Test bias detection result creation."""
        result = BiasDetectionResult(
            bias_type="categorical_bias",
            severity=0.7,
            affected_categories=["category1", "category2"],
            evidence={"p_value": 0.01, "effect_size": 0.8},
            recommendations=["Review criteria", "Retrain model"],
        )

        assert result.bias_type == "categorical_bias"
        assert result.severity == 0.7
        assert "category1" in result.affected_categories
        assert result.evidence["p_value"] == 0.01
        assert "Review criteria" in result.recommendations

    def test_severity_bounds(self):
        """Test severity bounds validation."""
        # Valid severity
        BiasDetectionResult(
            bias_type="test",
            severity=0.0,
            affected_categories=["cat1"],
            evidence={},
            recommendations=[],
        )

        BiasDetectionResult(
            bias_type="test",
            severity=1.0,
            affected_categories=["cat1"],
            evidence={},
            recommendations=[],
        )

        # Invalid severity
        with pytest.raises(ValueError):
            BiasDetectionResult(
                bias_type="test",
                severity=-0.1,
                affected_categories=["cat1"],
                evidence={},
                recommendations=[],
            )

        with pytest.raises(ValueError):
            BiasDetectionResult(
                bias_type="test",
                severity=1.1,
                affected_categories=["cat1"],
                evidence={},
                recommendations=[],
            )


class TestCalibrationResult:
    """Test calibration result model."""

    def test_calibration_result_creation(self):
        """Test calibration result creation."""
        result = CalibrationResult(
            judge_model="gpt-4",
            calibration_score=0.85,
            correlation=0.92,
            sample_size=100,
            confidence_intervals={
                "correlation": (0.88, 0.95),
                "calibration": (0.8, 0.9),
            },
        )

        assert result.judge_model == "gpt-4"
        assert result.calibration_score == 0.85
        assert result.correlation == 0.92
        assert result.sample_size == 100
        assert result.confidence_intervals["correlation"] == (0.88, 0.95)

    def test_score_bounds_validation(self):
        """Test score bounds validation."""
        # Valid scores
        CalibrationResult(
            judge_model="gpt-4",
            calibration_score=0.0,
            correlation=-1.0,
            sample_size=10,
            confidence_intervals={},
        )

        CalibrationResult(
            judge_model="gpt-4",
            calibration_score=1.0,
            correlation=1.0,
            sample_size=10,
            confidence_intervals={},
        )

        # Invalid calibration scores
        with pytest.raises(ValueError):
            CalibrationResult(
                judge_model="gpt-4",
                calibration_score=-0.1,
                correlation=0.5,
                sample_size=10,
                confidence_intervals={},
            )

        with pytest.raises(ValueError):
            CalibrationResult(
                judge_model="gpt-4",
                calibration_score=1.1,
                correlation=0.5,
                sample_size=10,
                confidence_intervals={},
            )

        # Invalid correlation scores
        with pytest.raises(ValueError):
            CalibrationResult(
                judge_model="gpt-4",
                calibration_score=0.8,
                correlation=-1.1,
                sample_size=10,
                confidence_intervals={},
            )

        with pytest.raises(ValueError):
            CalibrationResult(
                judge_model="gpt-4",
                calibration_score=0.8,
                correlation=1.1,
                sample_size=10,
                confidence_intervals={},
            )

    def test_sample_size_validation(self):
        """Test sample size validation."""
        # Valid sample size
        CalibrationResult(
            judge_model="gpt-4",
            calibration_score=0.8,
            correlation=0.5,
            sample_size=1,
            confidence_intervals={},
        )

        # Invalid sample size
        with pytest.raises(ValueError):
            CalibrationResult(
                judge_model="gpt-4",
                calibration_score=0.8,
                correlation=0.5,
                sample_size=0,
                confidence_intervals={},
            )
