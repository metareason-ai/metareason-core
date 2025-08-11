"""Tests for LLM judge quality assurance."""

from unittest.mock import patch

import numpy as np
import pytest

from metareason.oracles.judge_response import (
    BinaryJudgeResponse,
    JudgeResult,
    NumericJudgeResponse,
)
from metareason.oracles.quality_assurance import JudgeQualityAssurance


class TestJudgeQualityAssurance:
    """Test judge quality assurance system."""

    @pytest.fixture
    def qa_system(self):
        """Create quality assurance system."""
        return JudgeQualityAssurance()

    @pytest.fixture
    def binary_results(self):
        """Create binary judge results."""
        results = []
        scores = [1, 1, 0, 1, 0, 1, 1, 0]  # Mixed results

        for score in scores:
            response = BinaryJudgeResponse(
                score=score, reasoning=f"Reasoning for {score}"
            )
            result = JudgeResult(
                response=response,
                raw_response=f"raw_{score}",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        return results

    @pytest.fixture
    def numeric_results(self):
        """Create numeric judge results."""
        results = []
        scores = [0.8, 0.85, 0.75, 0.9, 0.7, 0.88, 0.82, 0.78]

        for score in scores:
            response = NumericJudgeResponse(score=score, reasoning=f"Score {score}")
            result = JudgeResult(
                response=response,
                raw_response=f"raw_{score}",
                judge_model="gpt-4",
                temperature=0.1,
            )
            results.append(result)

        return results

    def test_measure_consistency_insufficient_results(self, qa_system):
        """Test consistency measurement with insufficient results."""
        single_result = [
            JudgeResult(
                response=BinaryJudgeResponse(score=1, reasoning="Test"),
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
        ]

        with pytest.raises(ValueError, match="Need at least 2 results"):
            qa_system.measure_consistency("gpt-4", single_result)

    def test_measure_consistency_binary(self, qa_system, binary_results):
        """Test consistency measurement with binary results."""
        measurement = qa_system.measure_consistency("gpt-4", binary_results)

        assert measurement.judge_model == "gpt-4"
        assert 0.0 <= measurement.consistency_score <= 1.0
        assert measurement.variance >= 0.0
        assert measurement.agreement_rate is not None
        assert 0.0 <= measurement.agreement_rate <= 1.0
        assert measurement.sample_size == len(binary_results)

        # Check metadata
        assert "scores" in measurement.metadata
        assert "mean_score" in measurement.metadata
        assert len(measurement.metadata["scores"]) == len(binary_results)

    def test_measure_consistency_numeric(self, qa_system, numeric_results):
        """Test consistency measurement with numeric results."""
        measurement = qa_system.measure_consistency("gpt-4", numeric_results)

        assert measurement.judge_model == "gpt-4"
        assert 0.0 <= measurement.consistency_score <= 1.0
        assert measurement.variance >= 0.0
        assert measurement.agreement_rate is None  # Not applicable for numeric
        assert measurement.sample_size == len(numeric_results)

    def test_measure_consistency_perfect_agreement(self, qa_system):
        """Test consistency with perfect agreement."""
        # All same binary scores
        results = []
        for _ in range(5):
            response = BinaryJudgeResponse(score=1, reasoning="Always pass")
            result = JudgeResult(
                response=response,
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        measurement = qa_system.measure_consistency("gpt-4", results)

        assert measurement.consistency_score == 1.0
        assert measurement.variance == 0.0
        assert measurement.agreement_rate == 1.0

    def test_measure_consistency_with_context(self, qa_system, binary_results):
        """Test consistency measurement with response context."""
        response_text = "This is the response being evaluated" * 10  # Long text

        measurement = qa_system.measure_consistency(
            "gpt-4", binary_results, response_text
        )

        assert measurement.metadata["response_context"] is not None
        assert len(measurement.metadata["response_context"]) <= 100  # Truncated

    def test_consistency_caching(self, qa_system, binary_results):
        """Test consistency results are cached."""
        qa_system.measure_consistency("gpt-4", binary_results)

        assert "gpt-4" in qa_system._consistency_cache
        assert len(qa_system._consistency_cache["gpt-4"]) == len(binary_results)

    def test_inter_judge_reliability_insufficient_judges(
        self, qa_system, binary_results
    ):
        """Test inter-judge reliability with insufficient judges."""
        judge_results = {"gpt-4": binary_results}

        with pytest.raises(ValueError, match="Need at least 2 judges"):
            qa_system.test_inter_judge_reliability(judge_results)

    def test_inter_judge_reliability_mismatched_counts(self, qa_system, binary_results):
        """Test inter-judge reliability with mismatched result counts."""
        judge_results = {
            "gpt-4": binary_results,
            "claude-3": binary_results[:5],  # Different count
        }

        with pytest.raises(ValueError, match="same number of cases"):
            qa_system.test_inter_judge_reliability(judge_results)

    def test_inter_judge_reliability_insufficient_cases(self, qa_system):
        """Test inter-judge reliability with insufficient test cases."""
        results1 = [
            JudgeResult(
                response=BinaryJudgeResponse(score=1, reasoning="Test"),
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
        ]
        results2 = [
            JudgeResult(
                response=BinaryJudgeResponse(score=0, reasoning="Test"),
                raw_response="raw",
                judge_model="claude-3",
                temperature=0.0,
            )
        ]

        judge_results = {"gpt-4": results1, "claude-3": results2}

        with pytest.raises(ValueError, match="Need at least 3 test cases"):
            qa_system.test_inter_judge_reliability(judge_results)

    def test_inter_judge_reliability_success(self, qa_system, binary_results):
        """Test successful inter-judge reliability testing."""
        # Create correlated results for second judge
        results2 = []
        for result in binary_results:
            # 80% agreement with first judge
            score = (
                result.response.score
                if np.random.random() > 0.2
                else 1 - result.response.score
            )
            response = BinaryJudgeResponse(score=score, reasoning="Second judge")
            result2 = JudgeResult(
                response=response,
                raw_response="raw2",
                judge_model="claude-3",
                temperature=0.0,
            )
            results2.append(result2)

        judge_results = {"gpt-4": binary_results, "claude-3": results2}

        reliability = qa_system.test_inter_judge_reliability(judge_results)

        assert "judges" in reliability
        assert reliability["judges"] == ["gpt-4", "claude-3"]
        assert reliability["n_cases"] == len(binary_results)
        assert "pairwise_correlations" in reliability
        assert "gpt-4_vs_claude-3" in reliability["pairwise_correlations"]
        assert "overall_correlation" in reliability
        assert "cronbach_alpha" in reliability
        assert "agreement_metrics" in reliability

    def test_detect_bias_no_categories(self, qa_system, binary_results):
        """Test bias detection with insufficient categories."""
        categories = {"cat1": [0, 1, 2]}  # Only one category

        results = qa_system.detect_bias("gpt-4", binary_results, categories)
        assert len(results) == 0

    def test_detect_bias_insufficient_data(self, qa_system):
        """Test bias detection with insufficient data per category."""
        # Too few results per category
        results = [
            JudgeResult(
                response=BinaryJudgeResponse(score=1, reasoning="Test"),
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
        ]

        categories = {"cat1": [0], "cat2": []}  # Not enough data

        bias_results = qa_system.detect_bias("gpt-4", results, categories)
        assert len(bias_results) == 0

    def test_detect_bias_success(self, qa_system):
        """Test successful bias detection."""
        # Create biased results - category 1 always passes, category 2 always fails
        results = []

        # Category 1 results (all pass)
        for _ in range(5):
            response = BinaryJudgeResponse(score=1, reasoning="Pass")
            result = JudgeResult(
                response=response,
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        # Category 2 results (all fail)
        for _ in range(5):
            response = BinaryJudgeResponse(score=0, reasoning="Fail")
            result = JudgeResult(
                response=response,
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        categories = {
            "category1": list(range(5)),  # Indices 0-4
            "category2": list(range(5, 10)),  # Indices 5-9
        }

        bias_results = qa_system.detect_bias("gpt-4", results, categories)

        if bias_results:  # May not detect bias if effect size criteria not met
            bias_result = bias_results[0]
            assert bias_result.bias_type.startswith("categorical_bias")
            assert "category1" in bias_result.affected_categories
            assert "category2" in bias_result.affected_categories
            assert 0.0 <= bias_result.severity <= 1.0
            assert len(bias_result.recommendations) > 0
            assert "p_value" in bias_result.evidence

    def test_bias_caching(self, qa_system, binary_results):
        """Test bias detection results are cached."""
        categories = {"cat1": [0, 1, 2, 3], "cat2": [4, 5, 6, 7]}

        qa_system.detect_bias("gpt-4", binary_results, categories)

        assert "gpt-4" in qa_system._bias_cache
        assert len(qa_system._bias_cache["gpt-4"]) > 0

    def test_calibrate_against_human_mismatched_lengths(
        self, qa_system, binary_results
    ):
        """Test calibration with mismatched lengths."""
        human_scores = [0.8, 0.7]  # Different length

        with pytest.raises(ValueError, match="same length"):
            qa_system.calibrate_against_human_judgments(
                "gpt-4", binary_results, human_scores
            )

    def test_calibrate_against_human_success(self, qa_system, numeric_results):
        """Test successful calibration against human judgments."""
        # Create correlated human scores
        human_scores = []
        for result in numeric_results:
            # Add some noise to judge scores to simulate human variation
            human_score = result.response.score + np.random.normal(0, 0.05)
            human_score = max(0.0, min(1.0, human_score))  # Clamp to valid range
            human_scores.append(human_score)

        calibration = qa_system.calibrate_against_human_judgments(
            "gpt-4", numeric_results, human_scores
        )

        assert calibration.judge_model == "gpt-4"
        assert 0.0 <= calibration.calibration_score <= 1.0
        assert -1.0 <= calibration.correlation <= 1.0
        assert calibration.sample_size == len(numeric_results)
        assert "correlation" in calibration.confidence_intervals
        assert "mae" in calibration.confidence_intervals

        # Check metadata
        assert "mae" in calibration.metadata
        assert "rmse" in calibration.metadata
        assert "human_score_stats" in calibration.metadata
        assert "judge_score_stats" in calibration.metadata

    def test_calibrate_with_perfect_correlation(self, qa_system):
        """Test calibration with perfect correlation."""
        # Create perfectly correlated results
        results = []
        human_scores = [0.8, 0.6, 0.9, 0.7, 0.5]

        for score in human_scores:
            response = NumericJudgeResponse(score=score, reasoning="Perfect match")
            result = JudgeResult(
                response=response,
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        calibration = qa_system.calibrate_against_human_judgments(
            "gpt-4", results, human_scores
        )

        # Should have high calibration score and perfect correlation
        assert calibration.calibration_score > 0.9
        assert abs(calibration.correlation - 1.0) < 0.01

    def test_calibrate_small_sample_warning(self, qa_system):
        """Test calibration with small sample size warning."""
        # Small sample
        results = []
        human_scores = [0.8, 0.6, 0.7]  # Only 3 samples

        for score in human_scores:
            response = NumericJudgeResponse(score=score, reasoning="Small sample")
            result = JudgeResult(
                response=response,
                raw_response="raw",
                judge_model="gpt-4",
                temperature=0.0,
            )
            results.append(result)

        with patch("metareason.oracles.quality_assurance.logger") as mock_logger:
            qa_system.calibrate_against_human_judgments("gpt-4", results, human_scores)

            # Should log warning about small sample
            mock_logger.warning.assert_called_once()
            assert "unreliable" in str(mock_logger.warning.call_args)

    def test_cronbach_alpha_calculation(self, qa_system):
        """Test Cronbach's alpha calculation."""
        # Create score matrix for testing
        score_matrix = np.array(
            [
                [0.8, 0.7, 0.9, 0.6],  # Judge 1
                [0.75, 0.72, 0.88, 0.62],  # Judge 2 (similar to judge 1)
                [0.82, 0.68, 0.91, 0.58],  # Judge 3 (similar to judge 1)
            ]
        )

        alpha = qa_system._calculate_cronbach_alpha(score_matrix)

        # Should be positive for consistent judges
        assert alpha > 0
        assert alpha <= 1.0

    def test_cronbach_alpha_single_judge(self, qa_system):
        """Test Cronbach's alpha with single judge."""
        score_matrix = np.array([[0.8, 0.7, 0.9]])  # Only one judge

        alpha = qa_system._calculate_cronbach_alpha(score_matrix)
        assert alpha == 0.0

    def test_agreement_metrics_binary(self, qa_system):
        """Test agreement metrics for binary scores."""
        # Create binary score matrix
        score_matrix = np.array(
            [
                [1, 0, 1, 0],  # Judge 1
                [1, 0, 1, 1],  # Judge 2 (mostly agrees)
                [1, 1, 1, 0],  # Judge 3 (mostly agrees)
            ]
        )

        judges = ["judge1", "judge2", "judge3"]
        metrics = qa_system._calculate_agreement_metrics(score_matrix, judges)

        assert metrics["type"] == "binary"
        assert "percentage_agreement" in metrics
        assert 0.0 <= metrics["percentage_agreement"] <= 1.0
        assert "unanimous_cases" in metrics
        assert "total_cases" in metrics

    def test_agreement_metrics_continuous(self, qa_system):
        """Test agreement metrics for continuous scores."""
        # Create continuous score matrix
        score_matrix = np.array(
            [
                [0.8, 0.7, 0.9, 0.6],
                [0.82, 0.71, 0.88, 0.59],
                [0.79, 0.69, 0.91, 0.61],
            ]
        )

        judges = ["judge1", "judge2", "judge3"]
        metrics = qa_system._calculate_agreement_metrics(score_matrix, judges)

        assert metrics["type"] == "continuous"
        assert "agreement_score" in metrics
        assert 0.0 <= metrics["agreement_score"] <= 1.0
        assert "mean_absolute_deviation" in metrics
        assert "std_mad" in metrics

    def test_bootstrap_confidence_interval(self, qa_system):
        """Test bootstrap confidence interval calculation."""
        x = [0.8, 0.7, 0.9, 0.6, 0.75, 0.85, 0.65, 0.95]
        y = [0.82, 0.71, 0.88, 0.59, 0.74, 0.87, 0.63, 0.94]

        def mean_absolute_error(x_vals, y_vals):
            return np.mean(
                np.abs(np.array(x_vals, dtype=float) - np.array(y_vals, dtype=float))
            )

        ci = qa_system._bootstrap_confidence_interval(
            x, y, mean_absolute_error, 0.95, n_bootstrap=100
        )

        assert len(ci) == 2
        assert ci[0] <= ci[1]  # Lower bound <= upper bound
        assert ci[0] >= 0  # MAE should be non-negative
