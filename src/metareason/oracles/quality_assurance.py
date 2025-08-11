"""Quality assurance components for LLM judges."""

import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .judge_response import (
    BiasDetectionResult,
    CalibrationResult,
    ConsistencyMeasurement,
    JudgeResult,
)

logger = logging.getLogger(__name__)


class JudgeQualityAssurance:
    """Quality assurance system for LLM judge evaluations."""

    def __init__(self):
        """Initialize quality assurance system."""
        self._consistency_cache: Dict[str, List[float]] = {}
        self._bias_cache: Dict[str, List[Dict[str, Any]]] = {}

    def measure_consistency(
        self,
        judge_model: str,
        results: List[JudgeResult],
        response_text: Optional[str] = None,
    ) -> ConsistencyMeasurement:
        """Measure judge consistency across multiple evaluations.

        Args:
            judge_model: Judge model identifier
            results: List of judge results for the same input
            response_text: Optional response text for context

        Returns:
            Consistency measurement
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 results to measure consistency")

        scores = [result.response.score for result in results]

        # Calculate variance and consistency score
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        consistency_score = 1.0 - min(1.0, score_variance * 4.0)  # Scale variance

        # Calculate agreement rate for binary judgments
        agreement_rate = None
        if all(isinstance(score, int) and score in (0, 1) for score in scores):
            unique_scores = set(scores)
            if len(unique_scores) == 1:
                agreement_rate = 1.0
            else:
                # Calculate mode agreement
                mode_score = statistics.mode(scores)
                agreement_rate = scores.count(mode_score) / len(scores)

        # Cache results for trend analysis
        if judge_model not in self._consistency_cache:
            self._consistency_cache[judge_model] = []
        self._consistency_cache[judge_model].extend(scores)

        return ConsistencyMeasurement(
            judge_model=judge_model,
            consistency_score=consistency_score,
            variance=score_variance,
            agreement_rate=agreement_rate,
            sample_size=len(results),
            metadata={
                "scores": scores,
                "mean_score": statistics.mean(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "response_context": response_text[:100] if response_text else None,
            },
        )

    def test_inter_judge_reliability(
        self,
        judge_results: Dict[str, List[JudgeResult]],
        test_cases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Test reliability between multiple judges.

        Args:
            judge_results: Dict mapping judge model to their results
            test_cases: Optional test case identifiers

        Returns:
            Inter-judge reliability metrics
        """
        judges = list(judge_results.keys())
        if len(judges) < 2:
            raise ValueError("Need at least 2 judges for reliability testing")

        # Ensure all judges have same number of results
        result_counts = [len(results) for results in judge_results.values()]
        if len(set(result_counts)) > 1:
            raise ValueError("All judges must evaluate same number of cases")

        n_cases = result_counts[0]
        if n_cases < 3:
            raise ValueError("Need at least 3 test cases for meaningful reliability")

        # Extract scores for correlation analysis
        score_matrix = []
        for judge in judges:
            scores = [result.response.score for result in judge_results[judge]]
            score_matrix.append(scores)

        score_matrix = np.array(score_matrix)

        # Calculate pairwise correlations
        correlations = {}
        for i, judge1 in enumerate(judges):
            for j, judge2 in enumerate(judges[i + 1 :], i + 1):
                corr, p_value = stats.pearsonr(score_matrix[i], score_matrix[j])
                correlations[f"{judge1}_vs_{judge2}"] = {
                    "correlation": float(corr),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }

        # Calculate overall reliability metrics
        overall_corr = np.mean([c["correlation"] for c in correlations.values()])

        # Calculate Cronbach's alpha for internal consistency
        cronbach_alpha = self._calculate_cronbach_alpha(score_matrix)

        # Calculate agreement rates for binary judgments
        agreement_metrics = self._calculate_agreement_metrics(score_matrix, judges)

        return {
            "judges": judges,
            "n_cases": n_cases,
            "pairwise_correlations": correlations,
            "overall_correlation": float(overall_corr),
            "cronbach_alpha": float(cronbach_alpha),
            "agreement_metrics": agreement_metrics,
            "metadata": {
                "score_matrix_shape": score_matrix.shape,
                "test_cases": test_cases or list(range(n_cases)),
            },
        }

    def detect_bias(
        self,
        judge_model: str,
        results: List[JudgeResult],
        categories: Dict[str, List[int]],
    ) -> List[BiasDetectionResult]:
        """Detect bias in judge responses across different categories.

        Args:
            judge_model: Judge model identifier
            results: Judge results to analyze
            categories: Dict mapping category names to result indices

        Returns:
            List of bias detection results
        """
        bias_results = []

        # Check each category pair for bias
        category_names = list(categories.keys())
        for i, cat1 in enumerate(category_names):
            for cat2 in category_names[i + 1 :]:
                bias_result = self._detect_pairwise_bias(
                    judge_model, results, cat1, categories[cat1], cat2, categories[cat2]
                )
                if bias_result:
                    bias_results.append(bias_result)

        # Cache results for trend analysis
        if judge_model not in self._bias_cache:
            self._bias_cache[judge_model] = []
        self._bias_cache[judge_model].extend(
            [{"categories": categories, "results": len(results)}]
        )

        return bias_results

    def calibrate_against_human_judgments(
        self,
        judge_model: str,
        judge_results: List[JudgeResult],
        human_scores: List[float],
        confidence_level: float = 0.95,
    ) -> CalibrationResult:
        """Calibrate judge against human judgments.

        Args:
            judge_model: Judge model identifier
            judge_results: Judge evaluation results
            human_scores: Corresponding human judgment scores
            confidence_level: Confidence level for intervals

        Returns:
            Calibration result
        """
        if len(judge_results) != len(human_scores):
            raise ValueError("Judge results and human scores must have same length")

        if len(judge_results) < 10:
            logger.warning("Calibration with < 10 samples may be unreliable")

        judge_scores = [result.response.score for result in judge_results]

        # Calculate correlation
        correlation, p_value = stats.pearsonr(judge_scores, human_scores)

        # Calculate calibration metrics
        mae = np.mean(np.abs(np.array(judge_scores) - np.array(human_scores)))
        rmse = np.sqrt(np.mean((np.array(judge_scores) - np.array(human_scores)) ** 2))

        # Calibration score (higher is better)
        calibration_score = max(0.0, 1.0 - (mae + rmse) / 2.0)

        # Calculate confidence intervals
        alpha = 1 - confidence_level

        # Correlation confidence interval
        n = len(judge_scores)
        r_z = np.arctanh(correlation)
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_z = r_z + np.array([-z_crit, z_crit]) * se_z
        correlation_ci = tuple(np.tanh(ci_z))

        # MAE confidence interval (bootstrap approximation)
        mae_ci = self._bootstrap_confidence_interval(
            judge_scores,
            human_scores,
            lambda x, y: np.mean(
                np.abs(np.array(x, dtype=float) - np.array(y, dtype=float))
            ),
            confidence_level,
        )

        confidence_intervals = {
            "correlation": correlation_ci,
            "mae": mae_ci,
            "calibration_score": (
                max(0, calibration_score - 0.1),
                min(1, calibration_score + 0.1),
            ),
        }

        return CalibrationResult(
            judge_model=judge_model,
            calibration_score=calibration_score,
            correlation=correlation,
            sample_size=len(judge_results),
            confidence_intervals=confidence_intervals,
            metadata={
                "mae": mae,
                "rmse": rmse,
                "p_value": p_value,
                "human_score_stats": {
                    "mean": statistics.mean(human_scores),
                    "std": (
                        statistics.stdev(human_scores) if len(human_scores) > 1 else 0.0
                    ),
                    "min": min(human_scores),
                    "max": max(human_scores),
                },
                "judge_score_stats": {
                    "mean": statistics.mean(judge_scores),
                    "std": (
                        statistics.stdev(judge_scores) if len(judge_scores) > 1 else 0.0
                    ),
                    "min": min(judge_scores),
                    "max": max(judge_scores),
                },
            },
        )

    def _detect_pairwise_bias(
        self,
        judge_model: str,
        results: List[JudgeResult],
        cat1_name: str,
        cat1_indices: List[int],
        cat2_name: str,
        cat2_indices: List[int],
    ) -> Optional[BiasDetectionResult]:
        """Detect bias between two categories."""
        # Extract scores for each category
        cat1_scores = [results[i].response.score for i in cat1_indices]
        cat2_scores = [results[i].response.score for i in cat2_indices]

        if len(cat1_scores) < 3 or len(cat2_scores) < 3:
            return None  # Not enough data for meaningful comparison

        # Statistical test for difference
        statistic, p_value = stats.mannwhitneyu(
            cat1_scores, cat2_scores, alternative="two-sided"
        )

        # Effect size (Cohen's d approximation)
        mean1, mean2 = np.mean(cat1_scores), np.mean(cat2_scores)
        pooled_std = np.sqrt(
            (
                (len(cat1_scores) - 1) * np.var(cat1_scores, ddof=1)
                + (len(cat2_scores) - 1) * np.var(cat2_scores, ddof=1)
            )
            / (len(cat1_scores) + len(cat2_scores) - 2)
        )

        effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Determine bias severity
        if p_value < 0.05 and effect_size > 0.5:
            severity = min(1.0, effect_size / 2.0)  # Scale effect size
        else:
            severity = 0.0

        # Only report significant bias
        if severity < 0.3:
            return None

        recommendations = [
            f"Review evaluation criteria for {cat1_name} vs {cat2_name}",
            "Consider retraining with balanced examples",
            "Use multiple judges to reduce individual bias",
        ]

        if effect_size > 0.8:
            recommendations.append(
                "Consider this a large effect requiring immediate attention"
            )

        return BiasDetectionResult(
            bias_type=f"categorical_bias_{cat1_name}_vs_{cat2_name}",
            severity=severity,
            affected_categories=[cat1_name, cat2_name],
            evidence={
                "p_value": p_value,
                "effect_size": effect_size,
                "mean_difference": mean1 - mean2,
                "cat1_mean": mean1,
                "cat2_mean": mean2,
                "cat1_n": len(cat1_scores),
                "cat2_n": len(cat2_scores),
                "statistic": float(statistic),
            },
            recommendations=recommendations,
        )

    def _calculate_cronbach_alpha(self, score_matrix: np.ndarray) -> float:
        """Calculate Cronbach's alpha for internal consistency."""
        n_judges, n_cases = score_matrix.shape

        if n_judges < 2:
            return 0.0

        # Calculate item variances and total variance
        item_variances = np.var(score_matrix, axis=1, ddof=1)
        total_scores = np.sum(score_matrix, axis=0)
        total_variance = np.var(total_scores, ddof=1)

        # Cronbach's alpha formula
        alpha = (n_judges / (n_judges - 1)) * (
            1 - np.sum(item_variances) / total_variance
        )

        return float(alpha)

    def _calculate_agreement_metrics(
        self, score_matrix: np.ndarray, judges: List[str]
    ) -> Dict[str, Any]:
        """Calculate agreement metrics for judges."""
        n_judges, n_cases = score_matrix.shape

        # Check if scores are binary
        is_binary = np.all(np.isin(score_matrix, [0, 1]))

        if is_binary:
            # Calculate percentage agreement
            agreements = []
            for case in range(n_cases):
                case_scores = score_matrix[:, case]
                unanimous = len(set(case_scores)) == 1
                agreements.append(unanimous)

            percentage_agreement = np.mean(agreements)

            return {
                "type": "binary",
                "percentage_agreement": float(percentage_agreement),
                "unanimous_cases": int(np.sum(agreements)),
                "total_cases": n_cases,
            }
        else:
            # Calculate mean absolute deviation for continuous scores
            mad_scores = []
            for case in range(n_cases):
                case_scores = score_matrix[:, case]
                mad = np.mean(np.abs(case_scores - np.mean(case_scores)))
                mad_scores.append(mad)

            mean_mad = np.mean(mad_scores)
            agreement_score = max(0.0, 1.0 - mean_mad * 4.0)  # Scale MAD

            return {
                "type": "continuous",
                "agreement_score": float(agreement_score),
                "mean_absolute_deviation": float(mean_mad),
                "std_mad": float(np.std(mad_scores)),
            }

    def _bootstrap_confidence_interval(
        self,
        x: List[float],
        y: List[float],
        statistic_func,
        confidence_level: float,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        n = len(x)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = [x[i] for i in indices]
            y_boot = [y[i] for i in indices]

            # Calculate statistic
            stat = statistic_func(x_boot, y_boot)
            bootstrap_stats.append(stat)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return (float(ci_lower), float(ci_upper))
