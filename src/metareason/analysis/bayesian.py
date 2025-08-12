"""Bayesian statistical analysis using PyMC for confidence interval calculation.

This module provides robust Bayesian analysis of oracle evaluation results,
using MCMC sampling to produce statistically rigorous confidence intervals.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
from scipy import stats

from metareason.config.statistical import StatisticalConfig
from metareason.oracles.base import OracleResult

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Results from Bayesian analysis."""

    oracle_name: str
    posterior_mean: float
    posterior_std: float
    hdi_lower: float
    hdi_upper: float
    credible_interval: float
    n_successes: int
    n_trials: int

    # Convergence diagnostics
    r_hat: float
    effective_sample_size: float
    n_divergences: int

    # Metadata
    model_type: str
    inference_method: str
    n_chains: int
    n_samples: int
    computation_time: float

    @property
    def success_rate(self) -> float:
        """Observed success rate."""
        return self.n_successes / self.n_trials if self.n_trials > 0 else 0.0

    @property
    def converged(self) -> bool:
        """Check if MCMC sampling converged."""
        return self.r_hat <= 1.05 and self.n_divergences == 0

    @property
    def reliable(self) -> bool:
        """Check if results are statistically reliable."""
        return (
            self.converged
            and self.effective_sample_size >= 400  # Minimum ESS
            and self.n_trials >= 10  # Minimum sample size
        )


@dataclass
class MultiOracleResult:
    """Results from multi-oracle Bayesian analysis."""

    individual_results: Dict[str, BayesianResult]
    joint_posterior_mean: Optional[float] = None
    joint_hdi_lower: Optional[float] = None
    joint_hdi_upper: Optional[float] = None
    oracle_correlations: Optional[Dict[Tuple[str, str], float]] = None
    combined_confidence: Optional[float] = None

    @property
    def all_converged(self) -> bool:
        """Check if all oracle analyses converged."""
        return all(result.converged for result in self.individual_results.values())

    @property
    def all_reliable(self) -> bool:
        """Check if all oracle results are reliable."""
        return all(result.reliable for result in self.individual_results.values())


class BayesianAnalyzer:
    """Bayesian statistical analyzer for oracle evaluation results."""

    def __init__(self, config: StatisticalConfig):
        """Initialize analyzer with statistical configuration.

        Args:
            config: Statistical configuration including priors and inference settings
        """
        self.config = config
        self._cache = {}
        self._cache_ttl = timedelta(hours=1)

    def analyze_oracle_results(
        self,
        oracle_results: List[OracleResult],
        oracle_name: str,
        threshold: float = 0.5,
    ) -> BayesianResult:
        """Analyze results from a single oracle using Bayesian inference.

        Args:
            oracle_results: List of oracle evaluation results
            oracle_name: Name of the oracle for identification
            threshold: Threshold for binary success/failure classification

        Returns:
            BayesianResult with posterior statistics and diagnostics

        Raises:
            ValueError: If insufficient data or invalid configuration
        """
        if not oracle_results:
            raise ValueError("Cannot analyze empty oracle results")

        # Check cache
        cache_key = self._get_cache_key(oracle_results, oracle_name, threshold)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Returning cached result for {oracle_name}")
            return cached_result

        start_time = datetime.now()

        # Convert scores to binary outcomes
        scores = np.array([result.score for result in oracle_results])
        successes = np.sum(scores >= threshold)
        trials = len(scores)

        logger.info(
            f"Analyzing {oracle_name}: {successes}/{trials} successes "
            f"(threshold={threshold})"
        )

        # Build and sample from Bayesian model
        with pm.Model():
            # Beta prior for success probability
            theta = pm.Beta(
                "theta", alpha=self.config.prior.alpha, beta=self.config.prior.beta
            )

            # Binomial likelihood
            pm.Binomial("obs", n=trials, p=theta, observed=successes)

            # Sample posterior
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                if self.config.inference.method == "mcmc":
                    trace = pm.sample(
                        draws=self.config.inference.samples,
                        chains=self.config.inference.chains,
                        target_accept=self.config.inference.target_accept,
                        return_inferencedata=True,
                        progressbar=trials > 100,  # Show progress for large datasets
                        random_seed=42,
                    )
                else:
                    raise NotImplementedError(
                        f"Inference method {self.config.inference.method} not yet implemented"
                    )

        # Extract posterior statistics
        posterior_samples = trace.posterior["theta"].values.flatten()

        # Calculate HDI
        hdi = az.hdi(
            trace, hdi_prob=self.config.output.credible_interval, var_names=["theta"]
        )["theta"].values

        # Convergence diagnostics
        summary_stats = az.summary(trace, var_names=["theta"])
        r_hat = float(summary_stats.loc["theta", "r_hat"])
        ess = float(summary_stats.loc["theta", "ess_bulk"])

        # Count divergences
        divergences = trace.sample_stats["diverging"].sum().item()

        computation_time = (datetime.now() - start_time).total_seconds()

        result = BayesianResult(
            oracle_name=oracle_name,
            posterior_mean=float(np.mean(posterior_samples)),
            posterior_std=float(np.std(posterior_samples)),
            hdi_lower=float(hdi[0]),
            hdi_upper=float(hdi[1]),
            credible_interval=self.config.output.credible_interval,
            n_successes=int(successes),
            n_trials=int(trials),
            r_hat=r_hat,
            effective_sample_size=ess,
            n_divergences=int(divergences),
            model_type=self.config.model,
            inference_method=self.config.inference.method,
            n_chains=self.config.inference.chains,
            n_samples=self.config.inference.samples,
            computation_time=computation_time,
        )

        # Cache result
        self._cache_result(cache_key, result)

        # Log warnings for poor convergence
        if not result.converged:
            logger.warning(
                f"Poor convergence for {oracle_name}: R-hat={r_hat:.3f}, "
                f"divergences={divergences}"
            )

        if not result.reliable:
            logger.warning(
                f"Results may be unreliable for {oracle_name}: "
                f"ESS={ess:.0f}, n_trials={trials}"
            )

        return result

    def _get_cache_key(
        self, oracle_results: List[OracleResult], oracle_name: str, threshold: float
    ) -> str:
        """Generate cache key for oracle results."""
        # Use hash of scores and configuration for caching
        scores_hash = hash(tuple(result.score for result in oracle_results))
        config_hash = hash(
            (
                self.config.model,
                self.config.prior.alpha,
                self.config.prior.beta,
                self.config.inference.samples,
                self.config.inference.chains,
                self.config.output.credible_interval,
                threshold,
            )
        )
        return f"{oracle_name}_{scores_hash}_{config_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[BayesianResult]:
        """Retrieve result from cache if valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return result
            else:
                del self._cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: BayesianResult) -> None:
        """Cache analysis result with timestamp."""
        self._cache[cache_key] = (result, datetime.now())


class MultiOracleAnalyzer:
    """Analyzer for multi-oracle Bayesian analysis."""

    def __init__(self, config: StatisticalConfig):
        """Initialize multi-oracle analyzer.

        Args:
            config: Statistical configuration
        """
        self.config = config
        self.single_analyzer = BayesianAnalyzer(config)

    def analyze_multiple_oracles(
        self,
        oracle_results_dict: Dict[str, List[OracleResult]],
        thresholds: Optional[Dict[str, float]] = None,
        compute_joint: bool = True,
        compute_correlations: bool = True,
    ) -> MultiOracleResult:
        """Analyze results from multiple oracles.

        Args:
            oracle_results_dict: Dictionary mapping oracle names to result lists
            thresholds: Optional thresholds per oracle (default 0.5)
            compute_joint: Whether to compute joint posterior
            compute_correlations: Whether to compute cross-oracle correlations

        Returns:
            MultiOracleResult with individual and joint analysis
        """
        if not oracle_results_dict:
            raise ValueError("Cannot analyze empty oracle results dictionary")

        thresholds = thresholds or {}

        logger.info(f"Analyzing {len(oracle_results_dict)} oracles")

        # Analyze each oracle independently
        individual_results = {}
        for oracle_name, results in oracle_results_dict.items():
            threshold = thresholds.get(oracle_name, 0.5)
            individual_results[oracle_name] = (
                self.single_analyzer.analyze_oracle_results(
                    results, oracle_name, threshold
                )
            )

        # Initialize multi-oracle result
        multi_result = MultiOracleResult(individual_results=individual_results)

        # Compute joint posterior if requested
        if compute_joint and len(individual_results) >= 2:
            joint_mean, joint_lower, joint_upper = self._compute_joint_posterior(
                individual_results
            )
            multi_result.joint_posterior_mean = joint_mean
            multi_result.joint_hdi_lower = joint_lower
            multi_result.joint_hdi_upper = joint_upper

        # Compute correlations if requested
        if compute_correlations and len(individual_results) >= 2:
            multi_result.oracle_correlations = self._compute_oracle_correlations(
                oracle_results_dict, thresholds
            )

        # Compute combined confidence score
        multi_result.combined_confidence = self._compute_combined_confidence(
            individual_results
        )

        return multi_result

    def _compute_joint_posterior(
        self, individual_results: Dict[str, BayesianResult]
    ) -> Tuple[float, float, float]:
        """Compute joint posterior across oracles.

        This assumes independence between oracles and computes the product of posteriors.
        """
        logger.info("Computing joint posterior distribution")

        # For now, use simple geometric mean of posterior means
        # In future, could implement proper joint modeling
        means = [result.posterior_mean for result in individual_results.values()]
        joint_mean = stats.gmean(means)

        # Conservative approach: use minimum HDI bounds
        lower_bounds = [result.hdi_lower for result in individual_results.values()]
        upper_bounds = [result.hdi_upper for result in individual_results.values()]

        joint_lower = min(lower_bounds)
        joint_upper = min(upper_bounds)  # Conservative: use minimum upper bound

        return joint_mean, joint_lower, joint_upper

    def _compute_oracle_correlations(
        self,
        oracle_results_dict: Dict[str, List[OracleResult]],
        thresholds: Dict[str, float],
    ) -> Dict[Tuple[str, str], float]:
        """Compute correlations between oracle outcomes."""
        logger.info("Computing cross-oracle correlations")

        oracle_names = list(oracle_results_dict.keys())
        correlations = {}

        # Convert to binary outcomes
        oracle_outcomes = {}
        for name, results in oracle_results_dict.items():
            threshold = thresholds.get(name, 0.5)
            scores = np.array([r.score for r in results])
            oracle_outcomes[name] = (scores >= threshold).astype(int)

        # Compute pairwise correlations
        for i, name1 in enumerate(oracle_names):
            for name2 in oracle_names[i + 1 :]:
                # Ensure same length
                min_len = min(len(oracle_outcomes[name1]), len(oracle_outcomes[name2]))
                outcomes1 = oracle_outcomes[name1][:min_len]
                outcomes2 = oracle_outcomes[name2][:min_len]

                if min_len > 5:  # Minimum samples for correlation
                    correlation, _ = stats.pearsonr(outcomes1, outcomes2)
                    correlations[(name1, name2)] = correlation
                else:
                    correlations[(name1, name2)] = 0.0

        return correlations

    def _compute_combined_confidence(
        self, individual_results: Dict[str, BayesianResult]
    ) -> float:
        """Compute combined confidence score across oracles."""
        if not individual_results:
            return 0.0

        # Weight by reliability and combine using harmonic mean
        reliable_results = [r for r in individual_results.values() if r.reliable]

        if not reliable_results:
            logger.warning("No reliable oracle results for combined confidence")
            return 0.0

        # Use harmonic mean of posterior means for conservative estimate
        means = [result.posterior_mean for result in reliable_results]
        combined = stats.hmean(
            [max(mean, 0.001) for mean in means]
        )  # Avoid division by zero

        return float(combined)


# Utility functions for common analysis patterns


def quick_analysis(
    oracle_results: List[OracleResult],
    oracle_name: str = "oracle",
    threshold: float = 0.5,
    credible_interval: float = 0.95,
) -> BayesianResult:
    """Quick Bayesian analysis with default settings.

    Args:
        oracle_results: Oracle evaluation results
        oracle_name: Name for the oracle
        threshold: Success threshold
        credible_interval: Credible interval level

    Returns:
        BayesianResult with analysis
    """
    config = StatisticalConfig()
    config.output.credible_interval = credible_interval

    analyzer = BayesianAnalyzer(config)
    return analyzer.analyze_oracle_results(oracle_results, oracle_name, threshold)


def analyze_oracle_comparison(
    oracle_a_results: List[OracleResult],
    oracle_b_results: List[OracleResult],
    oracle_a_name: str = "oracle_a",
    oracle_b_name: str = "oracle_b",
    threshold: float = 0.5,
) -> Tuple[BayesianResult, BayesianResult, float]:
    """Compare two oracles using Bayesian analysis.

    Args:
        oracle_a_results: Results from first oracle
        oracle_b_results: Results from second oracle
        oracle_a_name: Name for first oracle
        oracle_b_name: Name for second oracle
        threshold: Success threshold

    Returns:
        Tuple of (result_a, result_b, probability_a_better_than_b)
    """
    config = StatisticalConfig()
    analyzer = BayesianAnalyzer(config)

    result_a = analyzer.analyze_oracle_results(
        oracle_a_results, oracle_a_name, threshold
    )
    result_b = analyzer.analyze_oracle_results(
        oracle_b_results, oracle_b_name, threshold
    )

    # Simple comparison: probability that A > B based on posterior means
    # In practice, would use proper posterior sampling for this
    prob_a_better = 1.0 if result_a.posterior_mean > result_b.posterior_mean else 0.0

    return result_a, result_b, prob_a_better
