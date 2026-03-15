import numpy as np
import pytest

from metareason.analysis.agreement import (
    compute_agreement_summary,
    compute_krippendorff_alpha,
    compute_pairwise_correlations,
    extract_scores_by_oracle,
)
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.runner import SampleResult


def _make_sample_result(evaluations_dict):
    """Helper to build a SampleResult with the given oracle->score mapping."""
    evals = {
        name: EvaluationResult(score=score, explanation="ok")
        for name, score in evaluations_dict.items()
    }
    return SampleResult(
        sample_params={"p": 0},
        original_prompt="prompt",
        final_response="response",
        evaluations=evals,
    )


# ---------------------------------------------------------------------------
# compute_pairwise_correlations
# ---------------------------------------------------------------------------


class TestComputePairwiseCorrelations:
    def test_perfect_positive_correlation(self):
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        result = compute_pairwise_correlations(scores)

        assert ("judge_a", "judge_b") in result["pearson"]
        assert result["pearson"][("judge_a", "judge_b")] == pytest.approx(1.0)
        assert result["spearman"][("judge_a", "judge_b")] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
        result = compute_pairwise_correlations(scores)

        assert result["pearson"][("judge_a", "judge_b")] == pytest.approx(-1.0)
        assert result["spearman"][("judge_a", "judge_b")] == pytest.approx(-1.0)

    def test_no_correlation(self):
        # Orthogonal vectors have zero Pearson correlation
        scores = {
            "judge_a": [1.0, 0.0, -1.0, 0.0],
            "judge_b": [0.0, 1.0, 0.0, -1.0],
        }
        result = compute_pairwise_correlations(scores)

        assert result["pearson"][("judge_a", "judge_b")] == pytest.approx(0.0, abs=0.01)

    def test_with_none_values_still_computes(self):
        """None values are filtered; enough remaining pairs should compute."""
        scores = {
            "judge_a": [1.0, None, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, None, 5.0],
        }
        result = compute_pairwise_correlations(scores)

        # Paired non-None: indices 0, 2, 4 -> (1,1), (3,3), (5,5)
        assert result["pearson"][("judge_a", "judge_b")] == pytest.approx(1.0)
        assert result["spearman"][("judge_a", "judge_b")] == pytest.approx(1.0)

    def test_too_few_pairs_returns_none(self):
        """Fewer than 3 valid pairs should yield None."""
        scores = {
            "judge_a": [1.0, None, None, None],
            "judge_b": [None, 2.0, None, 4.0],
        }
        result = compute_pairwise_correlations(scores)

        assert result["pearson"][("judge_a", "judge_b")] is None
        assert result["spearman"][("judge_a", "judge_b")] is None

    def test_three_judges_produces_three_pairs(self):
        scores = {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],
            "c": [3.0, 2.0, 1.0],
        }
        result = compute_pairwise_correlations(scores)

        assert len(result["pearson"]) == 3
        assert len(result["spearman"]) == 3

    def test_two_judges_minimum(self):
        """Two judges is the minimum for pairwise correlation."""
        scores = {
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.0],
        }
        result = compute_pairwise_correlations(scores)

        assert len(result["pearson"]) == 1
        assert result["pearson"][("x", "y")] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_krippendorff_alpha
# ---------------------------------------------------------------------------


class TestComputeKrippendorffAlpha:
    def test_perfect_agreement_returns_one(self):
        scores = {
            "judge_a": [3.0, 3.0, 3.0, 3.0],
            "judge_b": [3.0, 3.0, 3.0, 3.0],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_perfect_agreement_varying_scores(self):
        """Judges agree perfectly but scores vary across units."""
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_no_agreement_alpha_below_perfect(self):
        """Independent random scores should give alpha well below 1."""
        rng = np.random.default_rng(42)
        n = 20
        scores = {
            "judge_a": rng.uniform(1.0, 5.0, n).tolist(),
            "judge_b": rng.uniform(1.0, 5.0, n).tolist(),
        }
        alpha = compute_krippendorff_alpha(scores)
        # Random scores -> alpha should be strictly less than 1.0
        assert alpha < 1.0

    def test_systematic_disagreement_lower_than_agreement(self):
        """Opposed scores should yield lower alpha than agreeing scores."""
        agreeing = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        disagreeing = {
            "judge_a": [1.0, 5.0, 1.0, 5.0, 1.0],
            "judge_b": [5.0, 1.0, 5.0, 1.0, 5.0],
        }
        alpha_agree = compute_krippendorff_alpha(agreeing)
        alpha_disagree = compute_krippendorff_alpha(disagreeing)
        assert alpha_agree > alpha_disagree

    def test_missing_data_handling(self):
        """None values should be treated as missing, not crash."""
        scores = {
            "judge_a": [1.0, 2.0, None, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, 4.0, None],
        }
        alpha = compute_krippendorff_alpha(scores)
        # Agreement is perfect on the overlapping set
        assert alpha == pytest.approx(1.0)

    def test_all_missing_returns_one(self):
        """If no valid pairs exist, function returns 1.0 by convention."""
        scores = {
            "judge_a": [None, None],
            "judge_b": [None, None],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_single_unit_returns_one(self):
        """Only one unit with both raters present -> d_o_den = 0 -> returns 1.0."""
        scores = {
            "judge_a": [3.0],
            "judge_b": [3.0],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_three_judges_agreement(self):
        scores = {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_two_judge_minimum(self):
        """Two judges is the minimum for Krippendorff's alpha."""
        scores = {
            "x": [1.0, 3.0, 5.0],
            "y": [1.0, 3.0, 5.0],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_ragged_data_different_lengths(self):
        """Handles oracles with different numbers of scores."""
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0],
        }
        # judge_b is shorter -> units 3,4 have only one rater -> skipped
        alpha = compute_krippendorff_alpha(scores)
        assert alpha == pytest.approx(1.0)

    def test_partial_agreement(self):
        """Partial agreement should yield alpha between 0 and 1."""
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.5, 2.5, 3.5, 4.5, 4.5],
        }
        alpha = compute_krippendorff_alpha(scores)
        assert 0.0 < alpha < 1.0


# ---------------------------------------------------------------------------
# compute_agreement_summary
# ---------------------------------------------------------------------------


class TestComputeAgreementSummary:
    def test_returns_all_expected_keys(self):
        scores = {
            "judge_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "judge_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        summary = compute_agreement_summary(scores)

        expected_keys = {
            "krippendorff_alpha",
            "pairwise_correlations",
            "mean_pearson",
            "mean_spearman",
            "oracle_stats",
            "n_judges",
        }
        assert set(summary.keys()) == expected_keys

    def test_n_judges_correct(self):
        scores = {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],
            "c": [1.0, 2.0, 3.0],
        }
        summary = compute_agreement_summary(scores)
        assert summary["n_judges"] == 3

    def test_oracle_stats_populated(self):
        scores = {
            "judge_a": [1.0, 2.0, 3.0],
            "judge_b": [4.0, 4.0, 4.0],
        }
        summary = compute_agreement_summary(scores)

        assert "judge_a" in summary["oracle_stats"]
        assert "judge_b" in summary["oracle_stats"]

        stats_a = summary["oracle_stats"]["judge_a"]
        assert stats_a["mean"] == pytest.approx(2.0)
        assert stats_a["n"] == 3
        assert stats_a["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))

        stats_b = summary["oracle_stats"]["judge_b"]
        assert stats_b["mean"] == pytest.approx(4.0)
        assert stats_b["std"] == pytest.approx(0.0)

    def test_pairwise_correlations_serialized_keys(self):
        """Pairwise correlation keys should be 'oracle_a_vs_oracle_b' strings."""
        scores = {
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "beta": [1.0, 2.0, 3.0, 4.0],
        }
        summary = compute_agreement_summary(scores)

        pearson_keys = list(summary["pairwise_correlations"]["pearson"].keys())
        spearman_keys = list(summary["pairwise_correlations"]["spearman"].keys())

        assert "alpha_vs_beta" in pearson_keys
        assert "alpha_vs_beta" in spearman_keys

    def test_mean_correlations_with_perfect_agreement(self):
        scores = {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
        summary = compute_agreement_summary(scores)

        assert summary["mean_pearson"] == pytest.approx(1.0)
        assert summary["mean_spearman"] == pytest.approx(1.0)

    def test_mean_correlations_none_when_insufficient_data(self):
        """If all pairs have too few data points, mean correlations are None."""
        scores = {
            "a": [1.0, None],
            "b": [None, 2.0],
        }
        summary = compute_agreement_summary(scores)

        assert summary["mean_pearson"] is None
        assert summary["mean_spearman"] is None

    def test_handles_none_scores_in_oracle_stats(self):
        scores = {
            "judge_a": [1.0, None, 3.0],
            "judge_b": [2.0, 2.0, 2.0],
        }
        summary = compute_agreement_summary(scores)

        # judge_a: non-None values are [1.0, 3.0]
        stats_a = summary["oracle_stats"]["judge_a"]
        assert stats_a["n"] == 2
        assert stats_a["mean"] == pytest.approx(2.0)

    def test_two_judges_summary(self):
        """Two judges minimum for a meaningful summary."""
        scores = {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
        }
        summary = compute_agreement_summary(scores)

        assert summary["n_judges"] == 2
        assert summary["krippendorff_alpha"] == pytest.approx(1.0)
        assert summary["mean_pearson"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_scores_by_oracle
# ---------------------------------------------------------------------------


class TestExtractScoresByOracle:
    def test_basic_extraction(self):
        results = [
            _make_sample_result({"oracle_a": 3.0, "oracle_b": 4.0}),
            _make_sample_result({"oracle_a": 2.0, "oracle_b": 5.0}),
        ]
        scores = extract_scores_by_oracle(results)

        assert set(scores.keys()) == {"oracle_a", "oracle_b"}
        assert scores["oracle_a"] == [3.0, 4.0] or scores["oracle_a"] == [3.0, 2.0]
        # Verify actual values
        assert scores["oracle_a"] == [3.0, 2.0]
        assert scores["oracle_b"] == [4.0, 5.0]

    def test_missing_evaluation_produces_none(self):
        """When an oracle is missing from a SampleResult, None is inserted."""
        results = [
            _make_sample_result({"oracle_a": 3.0, "oracle_b": 4.0}),
            _make_sample_result({"oracle_a": 2.0}),  # oracle_b missing
        ]
        scores = extract_scores_by_oracle(results)

        assert scores["oracle_a"] == [3.0, 2.0]
        assert scores["oracle_b"] == [4.0, None]

    def test_ragged_data_all_oracles_same_length(self):
        """Even with ragged input, all oracle lists should be the same length."""
        results = [
            _make_sample_result({"a": 1.0}),
            _make_sample_result({"b": 2.0}),
            _make_sample_result({"a": 3.0, "b": 4.0}),
        ]
        scores = extract_scores_by_oracle(results)

        assert len(scores["a"]) == 3
        assert len(scores["b"]) == 3
        assert scores["a"] == [1.0, None, 3.0]
        assert scores["b"] == [None, 2.0, 4.0]

    def test_empty_results_list(self):
        scores = extract_scores_by_oracle([])
        assert scores == {}

    def test_single_oracle(self):
        results = [
            _make_sample_result({"only_judge": 3.5}),
            _make_sample_result({"only_judge": 4.0}),
        ]
        scores = extract_scores_by_oracle(results)

        assert list(scores.keys()) == ["only_judge"]
        assert scores["only_judge"] == [3.5, 4.0]

    def test_oracle_names_sorted(self):
        results = [
            _make_sample_result({"zebra": 1.0, "alpha": 2.0, "middle": 3.0}),
        ]
        scores = extract_scores_by_oracle(results)

        assert list(scores.keys()) == ["alpha", "middle", "zebra"]

    def test_three_oracles_one_missing_per_sample(self):
        """Each sample is missing a different oracle."""
        results = [
            _make_sample_result({"a": 1.0, "b": 2.0}),  # c missing
            _make_sample_result({"a": 3.0, "c": 4.0}),  # b missing
            _make_sample_result({"b": 5.0, "c": 1.0}),  # a missing
        ]
        scores = extract_scores_by_oracle(results)

        assert scores["a"] == [1.0, 3.0, None]
        assert scores["b"] == [2.0, None, 5.0]
        assert scores["c"] == [None, 4.0, 1.0]
