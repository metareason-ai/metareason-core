"""Tests for Bayesian statistical analysis module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from metareason.analysis.bayesian import (
    BayesianAnalyzer,
    BayesianResult,
    MultiOracleAnalyzer,
    MultiOracleResult,
    analyze_oracle_comparison,
    quick_analysis,
)
from metareason.config.statistical import (
    InferenceConfig,
    OutputConfig,
    PriorConfig,
    StatisticalConfig,
)
from metareason.oracles.base import OracleResult


class TestBayesianResult:
    """Test BayesianResult dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate property calculation."""
        result = BayesianResult(
            oracle_name="test",
            posterior_mean=0.7,
            posterior_std=0.1,
            hdi_lower=0.5,
            hdi_upper=0.9,
            credible_interval=0.95,
            n_successes=7,
            n_trials=10,
            r_hat=1.01,
            effective_sample_size=1000,
            n_divergences=0,
            model_type="beta_binomial",
            inference_method="mcmc",
            n_chains=4,
            n_samples=4000,
            computation_time=1.5,
        )

        assert result.success_rate == 0.7

    def test_success_rate_zero_trials(self):
        """Test success rate with zero trials."""
        result = BayesianResult(
            oracle_name="test",
            posterior_mean=0.0,
            posterior_std=0.0,
            hdi_lower=0.0,
            hdi_upper=0.0,
            credible_interval=0.95,
            n_successes=0,
            n_trials=0,
            r_hat=1.0,
            effective_sample_size=1000,
            n_divergences=0,
            model_type="beta_binomial",
            inference_method="mcmc",
            n_chains=4,
            n_samples=4000,
            computation_time=0.1,
        )

        assert result.success_rate == 0.0

    def test_convergence_check(self):
        """Test convergence property."""
        # Good convergence
        result_good = BayesianResult(
            oracle_name="test",
            posterior_mean=0.7,
            posterior_std=0.1,
            hdi_lower=0.5,
            hdi_upper=0.9,
            credible_interval=0.95,
            n_successes=7,
            n_trials=10,
            r_hat=1.01,
            effective_sample_size=1000,
            n_divergences=0,
            model_type="beta_binomial",
            inference_method="mcmc",
            n_chains=4,
            n_samples=4000,
            computation_time=1.5,
        )
        assert result_good.converged

        # Poor R-hat
        result_bad_rhat = result_good.__class__(
            **{**result_good.__dict__, "r_hat": 1.1}
        )
        assert not result_bad_rhat.converged

        # Divergences
        result_divergent = result_good.__class__(
            **{**result_good.__dict__, "n_divergences": 5}
        )
        assert not result_divergent.converged

    def test_reliability_check(self):
        """Test reliability property."""
        # Reliable result
        result_reliable = BayesianResult(
            oracle_name="test",
            posterior_mean=0.7,
            posterior_std=0.1,
            hdi_lower=0.5,
            hdi_upper=0.9,
            credible_interval=0.95,
            n_successes=70,
            n_trials=100,
            r_hat=1.01,
            effective_sample_size=1000,
            n_divergences=0,
            model_type="beta_binomial",
            inference_method="mcmc",
            n_chains=4,
            n_samples=4000,
            computation_time=1.5,
        )
        assert result_reliable.reliable

        # Low ESS
        result_low_ess = result_reliable.__class__(
            **{**result_reliable.__dict__, "effective_sample_size": 100}
        )
        assert not result_low_ess.reliable

        # Small sample size
        result_small_n = result_reliable.__class__(
            **{**result_reliable.__dict__, "n_trials": 5, "n_successes": 3}
        )
        assert not result_small_n.reliable


class TestMultiOracleResult:
    """Test MultiOracleResult dataclass."""

    @pytest.fixture
    def sample_individual_results(self):
        """Create sample individual results."""
        return {
            "oracle_a": BayesianResult(
                oracle_name="oracle_a",
                posterior_mean=0.8,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.95,
                credible_interval=0.95,
                n_successes=80,
                n_trials=100,
                r_hat=1.01,
                effective_sample_size=1000,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.5,
            ),
            "oracle_b": BayesianResult(
                oracle_name="oracle_b",
                posterior_mean=0.7,
                posterior_std=0.12,
                hdi_lower=0.5,
                hdi_upper=0.9,
                credible_interval=0.95,
                n_successes=70,
                n_trials=100,
                r_hat=1.02,
                effective_sample_size=950,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.8,
            ),
        }

    def test_all_converged(self, sample_individual_results):
        """Test all_converged property."""
        result = MultiOracleResult(individual_results=sample_individual_results)
        assert result.all_converged

        # Make one not converged
        sample_individual_results["oracle_a"].r_hat = 1.1
        result_bad = MultiOracleResult(individual_results=sample_individual_results)
        assert not result_bad.all_converged

    def test_all_reliable(self, sample_individual_results):
        """Test all_reliable property."""
        result = MultiOracleResult(individual_results=sample_individual_results)
        assert result.all_reliable

        # Make one unreliable
        sample_individual_results["oracle_b"].effective_sample_size = 100
        result_unreliable = MultiOracleResult(
            individual_results=sample_individual_results
        )
        assert not result_unreliable.all_reliable


class TestBayesianAnalyzer:
    """Test BayesianAnalyzer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return StatisticalConfig(
            model="beta_binomial",
            prior=PriorConfig(alpha=1.0, beta=1.0),
            inference=InferenceConfig(
                method="mcmc",
                samples=1000,  # Reduced for faster testing
                chains=2,
                target_accept=0.8,
            ),
            output=OutputConfig(credible_interval=0.95, hdi_method="shortest"),
        )

    @pytest.fixture
    def analyzer(self, config):
        """Create test analyzer."""
        return BayesianAnalyzer(config)

    @pytest.fixture
    def sample_oracle_results(self):
        """Create sample oracle results."""
        # 8 successes out of 10 trials
        return [
            OracleResult(score=0.8, metadata={}),
            OracleResult(score=0.9, metadata={}),
            OracleResult(score=0.7, metadata={}),
            OracleResult(score=0.6, metadata={}),
            OracleResult(score=0.3, metadata={}),  # failure
            OracleResult(score=0.85, metadata={}),
            OracleResult(score=0.2, metadata={}),  # failure
            OracleResult(score=0.75, metadata={}),
            OracleResult(score=0.92, metadata={}),
            OracleResult(score=0.88, metadata={}),
        ]

    def test_analyze_oracle_results_basic(self, analyzer, sample_oracle_results):
        """Test basic oracle analysis."""
        with (
            patch("pymc.sample") as mock_sample,
            patch("arviz.hdi") as mock_hdi,
            patch("arviz.summary") as mock_summary,
        ):

            # Mock PyMC sampling results
            mock_trace = MagicMock()
            mock_trace.posterior = {"theta": MagicMock()}
            mock_trace.posterior["theta"].values.flatten.return_value = np.array(
                [0.7, 0.75, 0.8, 0.72, 0.78]
            )
            mock_trace.sample_stats = {"diverging": MagicMock()}
            mock_trace.sample_stats["diverging"].sum.return_value.item.return_value = 0
            mock_sample.return_value = mock_trace

            # Mock HDI calculation
            mock_hdi.return_value = {"theta": MagicMock()}
            mock_hdi.return_value["theta"].values = np.array([0.6, 0.85])

            # Mock summary statistics
            mock_summary_df = MagicMock()
            mock_summary_df.loc.__getitem__.side_effect = lambda key: {
                ("theta", "r_hat"): 1.01,
                ("theta", "ess_bulk"): 800,
            }[key]
            mock_summary.return_value = mock_summary_df

            result = analyzer.analyze_oracle_results(
                sample_oracle_results, "test_oracle", threshold=0.5
            )

            assert isinstance(result, BayesianResult)
            assert result.oracle_name == "test_oracle"
            assert result.n_successes == 8  # scores >= 0.5
            assert result.n_trials == 10
            assert result.model_type == "beta_binomial"
            assert result.r_hat == 1.01
            assert result.effective_sample_size == 800
            assert result.n_divergences == 0
            assert result.converged

    def test_empty_results_error(self, analyzer):
        """Test error handling for empty results."""
        with pytest.raises(ValueError, match="Cannot analyze empty oracle results"):
            analyzer.analyze_oracle_results([], "test_oracle")

    def test_caching_functionality(self, analyzer, sample_oracle_results):
        """Test result caching."""
        with (
            patch("pymc.sample") as mock_sample,
            patch("arviz.hdi") as mock_hdi,
            patch("arviz.summary") as mock_summary,
        ):

            # Setup mocks
            mock_trace = MagicMock()
            mock_trace.posterior = {"theta": MagicMock()}
            mock_trace.posterior["theta"].values.flatten.return_value = np.array([0.8])
            mock_trace.sample_stats = {"diverging": MagicMock()}
            mock_trace.sample_stats["diverging"].sum.return_value.item.return_value = 0
            mock_sample.return_value = mock_trace

            mock_hdi.return_value = {"theta": MagicMock()}
            mock_hdi.return_value["theta"].values = np.array([0.6, 0.9])

            mock_summary_df = MagicMock()
            mock_summary_df.loc.__getitem__.side_effect = lambda key: {
                ("theta", "r_hat"): 1.01,
                ("theta", "ess_bulk"): 800,
            }[key]
            mock_summary.return_value = mock_summary_df

            # First call should run PyMC
            result1 = analyzer.analyze_oracle_results(
                sample_oracle_results, "test_oracle"
            )
            assert mock_sample.call_count == 1

            # Second call should use cache
            result2 = analyzer.analyze_oracle_results(
                sample_oracle_results, "test_oracle"
            )
            assert mock_sample.call_count == 1  # No additional calls

            # Results should be identical
            assert result1.posterior_mean == result2.posterior_mean
            assert result1.oracle_name == result2.oracle_name


class TestMultiOracleAnalyzer:
    """Test MultiOracleAnalyzer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return StatisticalConfig(
            inference=InferenceConfig(samples=1000, chains=2)  # Minimum valid samples
        )

    @pytest.fixture
    def multi_analyzer(self, config):
        """Create test multi-oracle analyzer."""
        return MultiOracleAnalyzer(config)

    @pytest.fixture
    def sample_multi_oracle_results(self):
        """Create sample multi-oracle results."""
        return {
            "accuracy": [
                OracleResult(score=0.8, metadata={}),
                OracleResult(score=0.9, metadata={}),
                OracleResult(score=0.7, metadata={}),
                OracleResult(score=0.6, metadata={}),
                OracleResult(score=0.85, metadata={}),
            ],
            "clarity": [
                OracleResult(score=0.75, metadata={}),
                OracleResult(score=0.8, metadata={}),
                OracleResult(score=0.6, metadata={}),
                OracleResult(score=0.7, metadata={}),
                OracleResult(score=0.9, metadata={}),
            ],
        }

    def test_analyze_multiple_oracles(
        self, multi_analyzer, sample_multi_oracle_results
    ):
        """Test multi-oracle analysis."""
        with patch.object(
            multi_analyzer.single_analyzer, "analyze_oracle_results"
        ) as mock_analyze:
            # Mock individual analyses
            mock_result_accuracy = BayesianResult(
                oracle_name="accuracy",
                posterior_mean=0.8,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.95,
                credible_interval=0.95,
                n_successes=4,
                n_trials=5,
                r_hat=1.01,
                effective_sample_size=400,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=2,
                n_samples=500,
                computation_time=1.0,
            )

            mock_result_clarity = BayesianResult(
                oracle_name="clarity",
                posterior_mean=0.7,
                posterior_std=0.12,
                hdi_lower=0.5,
                hdi_upper=0.9,
                credible_interval=0.95,
                n_successes=4,
                n_trials=5,
                r_hat=1.02,
                effective_sample_size=420,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=2,
                n_samples=500,
                computation_time=1.1,
            )

            mock_analyze.side_effect = [mock_result_accuracy, mock_result_clarity]

            result = multi_analyzer.analyze_multiple_oracles(
                sample_multi_oracle_results
            )

            assert isinstance(result, MultiOracleResult)
            assert len(result.individual_results) == 2
            assert "accuracy" in result.individual_results
            assert "clarity" in result.individual_results
            assert result.joint_posterior_mean is not None
            assert result.oracle_correlations is not None
            assert result.combined_confidence is not None

    def test_empty_results_error(self, multi_analyzer):
        """Test error handling for empty results dictionary."""
        with pytest.raises(
            ValueError, match="Cannot analyze empty oracle results dictionary"
        ):
            multi_analyzer.analyze_multiple_oracles({})

    def test_joint_posterior_computation(self, multi_analyzer):
        """Test joint posterior computation."""
        individual_results = {
            "oracle_a": BayesianResult(
                oracle_name="oracle_a",
                posterior_mean=0.8,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.95,
                credible_interval=0.95,
                n_successes=8,
                n_trials=10,
                r_hat=1.01,
                effective_sample_size=400,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=2,
                n_samples=500,
                computation_time=1.0,
            ),
            "oracle_b": BayesianResult(
                oracle_name="oracle_b",
                posterior_mean=0.6,
                posterior_std=0.12,
                hdi_lower=0.4,
                hdi_upper=0.8,
                credible_interval=0.95,
                n_successes=6,
                n_trials=10,
                r_hat=1.02,
                effective_sample_size=420,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=2,
                n_samples=500,
                computation_time=1.1,
            ),
        }

        joint_mean, joint_lower, joint_upper = multi_analyzer._compute_joint_posterior(
            individual_results
        )

        # Should be geometric mean of individual means
        expected_mean = (0.8 * 0.6) ** 0.5
        assert abs(joint_mean - expected_mean) < 0.01

        # Should use conservative bounds
        assert joint_lower == min(0.6, 0.4)  # min of lower bounds
        assert joint_upper == min(0.95, 0.8)  # min of upper bounds

    def test_combined_confidence_computation(self, multi_analyzer):
        """Test combined confidence computation."""
        reliable_results = {
            "oracle_a": BayesianResult(
                oracle_name="oracle_a",
                posterior_mean=0.8,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.95,
                credible_interval=0.95,
                n_successes=80,
                n_trials=100,
                r_hat=1.01,
                effective_sample_size=1000,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.0,
            ),
            "oracle_b": BayesianResult(
                oracle_name="oracle_b",
                posterior_mean=0.6,
                posterior_std=0.12,
                hdi_lower=0.4,
                hdi_upper=0.8,
                credible_interval=0.95,
                n_successes=60,
                n_trials=100,
                r_hat=1.02,
                effective_sample_size=950,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.1,
            ),
        }

        combined = multi_analyzer._compute_combined_confidence(reliable_results)

        # Should be harmonic mean for conservative estimate
        expected = 2 / (1 / 0.8 + 1 / 0.6)
        assert abs(combined - expected) < 0.01

    def test_empty_combined_confidence(self, multi_analyzer):
        """Test combined confidence with no results."""
        assert multi_analyzer._compute_combined_confidence({}) == 0.0


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.fixture
    def sample_results(self):
        """Create sample oracle results."""
        return [
            OracleResult(score=0.8, metadata={}),
            OracleResult(score=0.6, metadata={}),
            OracleResult(score=0.4, metadata={}),  # failure at 0.5 threshold
            OracleResult(score=0.9, metadata={}),
            OracleResult(score=0.7, metadata={}),
        ]

    def test_quick_analysis(self, sample_results):
        """Test quick analysis utility function."""
        with patch(
            "metareason.analysis.bayesian.BayesianAnalyzer.analyze_oracle_results"
        ) as mock_analyze:
            mock_result = BayesianResult(
                oracle_name="oracle",
                posterior_mean=0.75,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.9,
                credible_interval=0.89,  # Custom interval
                n_successes=4,
                n_trials=5,
                r_hat=1.01,
                effective_sample_size=1000,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.0,
            )
            mock_analyze.return_value = mock_result

            result = quick_analysis(
                sample_results, oracle_name="test_oracle", credible_interval=0.89
            )

            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args

            # Check that custom credible interval was passed through config
            assert call_args[0][1] == "test_oracle"  # oracle_name
            assert call_args[0][2] == 0.5  # default threshold

            assert result.credible_interval == 0.89

    def test_analyze_oracle_comparison(self, sample_results):
        """Test oracle comparison utility function."""
        results_b = [
            OracleResult(score=0.5, metadata={}),
            OracleResult(score=0.3, metadata={}),  # failure
            OracleResult(score=0.8, metadata={}),
            OracleResult(score=0.7, metadata={}),
            OracleResult(score=0.6, metadata={}),
        ]

        with patch(
            "metareason.analysis.bayesian.BayesianAnalyzer.analyze_oracle_results"
        ) as mock_analyze:
            mock_result_a = BayesianResult(
                oracle_name="custom_a",
                posterior_mean=0.8,
                posterior_std=0.1,
                hdi_lower=0.6,
                hdi_upper=0.95,
                credible_interval=0.95,
                n_successes=4,
                n_trials=5,
                r_hat=1.01,
                effective_sample_size=1000,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.0,
            )

            mock_result_b = BayesianResult(
                oracle_name="custom_b",
                posterior_mean=0.6,
                posterior_std=0.12,
                hdi_lower=0.4,
                hdi_upper=0.8,
                credible_interval=0.95,
                n_successes=4,
                n_trials=5,
                r_hat=1.02,
                effective_sample_size=950,
                n_divergences=0,
                model_type="beta_binomial",
                inference_method="mcmc",
                n_chains=4,
                n_samples=4000,
                computation_time=1.1,
            )

            mock_analyze.side_effect = [mock_result_a, mock_result_b]

            result_a, result_b, prob_a_better = analyze_oracle_comparison(
                sample_results,
                results_b,
                oracle_a_name="custom_a",
                oracle_b_name="custom_b",
            )

            assert result_a.oracle_name == "custom_a"
            assert result_b.oracle_name == "custom_b"
            assert prob_a_better == 1.0  # A has higher mean
            assert mock_analyze.call_count == 2


@pytest.mark.integration
class TestBayesianIntegration:
    """Integration tests with actual PyMC (marked as slow)."""

    @pytest.mark.slow
    def test_real_pymc_analysis(self):
        """Test with actual PyMC sampling (slow test)."""
        # Create simple test data
        oracle_results = [
            OracleResult(score=0.8, metadata={}),
            OracleResult(score=0.9, metadata={}),
            OracleResult(score=0.7, metadata={}),
            OracleResult(score=0.85, metadata={}),
            OracleResult(score=0.3, metadata={}),  # failure
        ]

        # Use minimal sampling for speed
        config = StatisticalConfig(
            inference=InferenceConfig(
                samples=1000, chains=2  # Minimum valid for testing
            )
        )

        analyzer = BayesianAnalyzer(config)
        result = analyzer.analyze_oracle_results(oracle_results, "real_test")

        # Basic checks that it ran successfully
        assert isinstance(result, BayesianResult)
        assert result.oracle_name == "real_test"
        assert result.n_successes == 4
        assert result.n_trials == 5
        assert 0.0 <= result.posterior_mean <= 1.0
        assert result.hdi_lower <= result.posterior_mean <= result.hdi_upper
