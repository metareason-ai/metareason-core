from typing import List

import arviz as az
import numpy as np
import pymc as pm

from metareason.config.models import BayesianAnalysisConfig, SpecConfig
from metareason.pipeline.runner import SampleResult


class BayesianAnalyzer:
    """Performs Bayesian statistical analysis on LLM evaluation results.

    Uses PyMC to build probabilistic models that estimate true quality scores
    while accounting for oracle measurement error and uncertainty.

    Attributes:
        results: List of SampleResult objects from evaluation pipeline.
        spec_config: Specification configuration including analysis parameters.
        n_variants: Number of sample variants evaluated.
        analysis_config: Bayesian analysis configuration (uses defaults if not specified).
    """

    def __init__(self, results: List[SampleResult], spec_config: SpecConfig):
        """Initialize the Bayesian analyzer.

        Args:
            results: List of SampleResult objects from the evaluation pipeline.
            spec_config: Specification configuration that may include analysis parameters.
                If spec_config.analysis is None, default parameters are used.
        """
        self.results = results
        self.spec_config = spec_config
        self.n_variants = len(results)

        # Use provided analysis config or defaults
        self.analysis_config = spec_config.analysis or BayesianAnalysisConfig()

    def fit_calibration_model(self, oracle_name: str) -> az.InferenceData:
        """Estimate true quality for each variant using a Bayesian calibration model.

        This model accounts for oracle measurement noise and provides posterior
        distributions over the true quality of each variant, rather than point estimates.

        Model structure:
            true_quality[i] ~ Normal(μ, σ)  # Prior: latent true quality
            oracle_noise ~ HalfNormal(σ_n)  # Prior: oracle measurement error
            observed[i] ~ Normal(true_quality[i], oracle_noise)  # Likelihood

        The priors μ, σ, and σ_n are configured via BayesianAnalysisConfig.

        Args:
            oracle_name: Name of the oracle to analyze (must exist in results).

        Returns:
            ArviZ InferenceData object containing posterior samples for:
                - true_quality: Array of true quality estimates for each variant
                - oracle_noise: Estimated oracle measurement error

        Raises:
            KeyError: If oracle_name doesn't exist in evaluation results.
        """
        scores = np.array([r.evaluations[oracle_name].score for r in self.results])

        with pm.Model() as model:
            # Prior: True quality for each variant
            true_quality = pm.Normal(
                "true_quality",
                mu=self.analysis_config.prior_quality_mu,
                sigma=self.analysis_config.prior_quality_sigma,
                shape=self.n_variants,
            )

            # Prior: Oracle measurement noise
            oracle_noise = pm.HalfNormal(
                "oracle_noise", sigma=self.analysis_config.prior_noise_sigma
            )

            # Likelihood: Observed scores = true quality + noise
            observed = pm.Normal(
                "observed", mu=true_quality, sigma=oracle_noise, observed=scores
            )

            # Sample posterior using MCMC (NUTS sampler)
            trace = pm.sample(
                draws=self.analysis_config.mcmc_draws,
                tune=self.analysis_config.mcmc_tune,
                chains=self.analysis_config.mcmc_chains,
            )

        return trace

    def estimate_population_quality(
        self, oracle_name: str, hdi_prob: float = 0.94
    ) -> dict:
        """Estimate overall population quality with high-density credible intervals.

        This method provides a single population-level quality estimate by pooling
        all oracle scores together, rather than estimating per-variant quality.
        This answers: "What's the overall quality of LLM outputs in this evaluation?"

        Model structure:
            overall_quality ~ Normal(μ, σ)      # Prior: population mean quality
            oracle_noise ~ HalfNormal(σ_n)      # Prior: measurement error
            observed[i] ~ Normal(overall_quality, oracle_noise)  # Likelihood

        Args:
            oracle_name: Name of the oracle to analyze (must exist in results).
            hdi_prob: Probability mass for the credible interval (default: 0.94).
                Common values: 0.89, 0.94, 0.95.

        Returns:
            Dictionary containing:
                - 'population_mean': Posterior mean of overall quality
                - 'population_median': Posterior median of overall quality
                - 'hdi_lower': Lower bound of high-density credible interval
                - 'hdi_upper': Upper bound of high-density credible interval
                - 'hdi_prob': Probability mass of the interval
                - 'oracle_noise_mean': Posterior mean of oracle measurement error
                - 'oracle_noise_hdi': (lower, upper) bounds for oracle noise
                - 'n_samples': Number of scores analyzed

        Raises:
            KeyError: If oracle_name doesn't exist in evaluation results.

        Example:
            >>> result = analyzer.estimate_population_quality('coherence_judge', hdi_prob=0.94)
            >>> print(f"Overall quality: {result['population_mean']:.2f}")
            >>> print(f"94% credible interval: [{result['hdi_lower']:.2f}, {result['hdi_upper']:.2f}]")
        """
        scores = np.array([r.evaluations[oracle_name].score for r in self.results])

        with pm.Model() as model:
            # Prior: Overall population quality (single parameter for all variants)
            overall_quality = pm.Normal(
                "overall_quality",
                mu=self.analysis_config.prior_quality_mu,
                sigma=self.analysis_config.prior_quality_sigma,
            )

            # Prior: Oracle measurement noise
            oracle_noise = pm.HalfNormal(
                "oracle_noise", sigma=self.analysis_config.prior_noise_sigma
            )

            # Likelihood: All observed scores come from same quality level + noise
            observed = pm.Normal(
                "observed", mu=overall_quality, sigma=oracle_noise, observed=scores
            )

            # Sample posterior using MCMC (NUTS sampler)
            trace = pm.sample(
                draws=self.analysis_config.mcmc_draws,
                tune=self.analysis_config.mcmc_tune,
                chains=self.analysis_config.mcmc_chains,
            )

        # Extract posterior samples
        posterior = trace.posterior

        # Compute HDI for overall quality
        quality_hdi = az.hdi(trace, var_names=["overall_quality"], hdi_prob=hdi_prob)
        noise_hdi = az.hdi(trace, var_names=["oracle_noise"], hdi_prob=hdi_prob)

        # Compute summary statistics
        quality_samples = posterior["overall_quality"].values.flatten()
        noise_samples = posterior["oracle_noise"].values.flatten()

        return {
            "population_mean": float(quality_samples.mean()),
            "population_median": float(np.median(quality_samples)),
            "hdi_lower": float(quality_hdi["overall_quality"].values[0]),
            "hdi_upper": float(quality_hdi["overall_quality"].values[1]),
            "hdi_prob": hdi_prob,
            "oracle_noise_mean": float(noise_samples.mean()),
            "oracle_noise_hdi": (
                float(noise_hdi["oracle_noise"].values[0]),
                float(noise_hdi["oracle_noise"].values[1]),
            ),
            "n_samples": len(scores),
        }
