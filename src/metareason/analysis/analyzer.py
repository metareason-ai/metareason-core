from typing import List, Optional

import arviz as az
import numpy as np
import pymc as pm

from metareason.config.models import AxisConfig, BayesianAnalysisConfig
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

    def __init__(
        self, results: List[SampleResult], spec_config=None, analysis_config=None
    ):
        """Initialize the Bayesian analyzer.

        Args:
            results: List of SampleResult objects from the evaluation pipeline.
            spec_config: Specification configuration that may include analysis parameters.
                If spec_config.analysis is None, default parameters are used.
            analysis_config: Direct BayesianAnalysisConfig to use. Takes precedence
                over spec_config.analysis if both are provided.
        """
        self.results = results
        self.spec_config = spec_config
        self.n_variants = len(results)

        if analysis_config:
            self.analysis_config = analysis_config
        elif spec_config:
            self.analysis_config = spec_config.analysis or BayesianAnalysisConfig()
        else:
            self.analysis_config = BayesianAnalysisConfig()

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
            true_quality = pm.Normal(
                "true_quality",
                mu=self.analysis_config.prior_quality_mu,
                sigma=self.analysis_config.prior_quality_sigma,
                shape=self.n_variants,
            )

            oracle_noise = pm.HalfNormal(
                "oracle_noise", sigma=self.analysis_config.prior_noise_sigma
            )

            observed = pm.Normal(
                "observed", mu=true_quality, sigma=oracle_noise, observed=scores
            )

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
            "population_std": float(quality_samples.std()),
            "hdi_lower": float(quality_hdi["overall_quality"].values[0]),
            "hdi_upper": float(quality_hdi["overall_quality"].values[1]),
            "hdi_prob": hdi_prob,
            "oracle_noise_mean": float(noise_samples.mean()),
            "oracle_noise_hdi": (
                float(noise_hdi["oracle_noise"].values[0]),
                float(noise_hdi["oracle_noise"].values[1]),
            ),
            "n_samples": len(scores),
            "posterior_samples": quality_samples.tolist(),
            "noise_posterior_samples": noise_samples.tolist(),
        }

    def estimate_judge_calibration(
        self,
        oracle_name: str,
        expected_score: Optional[float] = None,
        hdi_prob: float = 0.94,
    ) -> dict:
        """Estimate judge bias and noise for calibration.

        With expected_score: estimates judge_bias and judge_noise, anchored to
        the known score. Without: estimates true_quality (as nuisance) and
        judge_noise; bias is not identifiable without a reference point.

        Args:
            oracle_name: Name of the oracle to analyze.
            expected_score: Known ground-truth score, if available.
            hdi_prob: Probability mass for credible intervals.

        Returns:
            Dict with judge-focused calibration results.
        """
        scores = np.array([r.evaluations[oracle_name].score for r in self.results])

        with pm.Model():
            if expected_score is not None:
                judge_bias = pm.Normal(
                    "judge_bias",
                    mu=0,
                    sigma=self.analysis_config.prior_bias_sigma,
                )
                judge_noise = pm.HalfNormal(
                    "judge_noise", sigma=self.analysis_config.prior_noise_sigma
                )
                pm.Normal(
                    "observed",
                    mu=expected_score + judge_bias,
                    sigma=judge_noise,
                    observed=scores,
                )
            else:
                true_quality = pm.Normal(
                    "true_quality",
                    mu=self.analysis_config.prior_quality_mu,
                    sigma=self.analysis_config.prior_quality_sigma,
                )
                judge_noise = pm.HalfNormal(
                    "judge_noise", sigma=self.analysis_config.prior_noise_sigma
                )
                pm.Normal(
                    "observed",
                    mu=true_quality,
                    sigma=judge_noise,
                    observed=scores,
                )

            trace = pm.sample(
                draws=self.analysis_config.mcmc_draws,
                tune=self.analysis_config.mcmc_tune,
                chains=self.analysis_config.mcmc_chains,
            )

        posterior = trace.posterior
        noise_samples = posterior["judge_noise"].values.flatten()
        noise_hdi = az.hdi(trace, var_names=["judge_noise"], hdi_prob=hdi_prob)

        result = {
            "noise_mean": float(noise_samples.mean()),
            "noise_hdi": (
                float(noise_hdi["judge_noise"].values[0]),
                float(noise_hdi["judge_noise"].values[1]),
            ),
            "n_samples": len(scores),
            "hdi_prob": hdi_prob,
            "raw_score_mean": float(scores.mean()),
            "raw_score_std": float(scores.std()),
            "noise_posterior_samples": noise_samples.tolist(),
        }

        if expected_score is not None:
            bias_samples = posterior["judge_bias"].values.flatten()
            bias_hdi = az.hdi(trace, var_names=["judge_bias"], hdi_prob=hdi_prob)
            result["expected_score"] = expected_score
            result["bias_mean"] = float(bias_samples.mean())
            result["bias_median"] = float(np.median(bias_samples))
            result["bias_hdi"] = (
                float(bias_hdi["judge_bias"].values[0]),
                float(bias_hdi["judge_bias"].values[1]),
            )
            result["bias_posterior_samples"] = bias_samples.tolist()
        else:
            quality_samples = posterior["true_quality"].values.flatten()
            quality_hdi = az.hdi(trace, var_names=["true_quality"], hdi_prob=hdi_prob)
            result["estimated_quality_mean"] = float(quality_samples.mean())
            result["estimated_quality_hdi"] = (
                float(quality_hdi["true_quality"].values[0]),
                float(quality_hdi["true_quality"].values[1]),
            )

        return result

    def _build_design_matrix(self, oracle_name: str, axes: List[AxisConfig]) -> tuple:
        """Build design matrix from sample params and axis configs.

        Continuous axes are z-standardized. Categorical axes use reference
        coding (first value in axis.values is the reference level).

        Returns:
            (X, col_info) where X is ndarray of shape (n_samples, n_predictors)
            and col_info is a list of dicts with column metadata.
        """
        columns = []
        col_info = []

        for axis in axes:
            values = [r.sample_params[axis.name] for r in self.results]

            if axis.type == "continuous":
                arr = np.array(values, dtype=float)
                std = arr.std(ddof=0)
                if std == 0:
                    continue  # Skip zero-variance axes
                mean = arr.mean()
                standardized = (arr - mean) / std
                columns.append(standardized)
                col_info.append(
                    {
                        "parameter": axis.name,
                        "type": "continuous",
                        "level": None,
                        "reference_level": None,
                        "standardization": {
                            "mean": float(mean),
                            "std": float(std),
                        },
                    }
                )

            elif axis.type == "categorical":
                reference = axis.values[0]
                non_ref_levels = axis.values[1:]
                for level in non_ref_levels:
                    dummy = np.array([1.0 if v == level else 0.0 for v in values])
                    columns.append(dummy)
                    col_info.append(
                        {
                            "parameter": axis.name,
                            "type": "categorical",
                            "level": str(level),
                            "reference_level": str(reference),
                            "standardization": None,
                        }
                    )

        if columns:
            X = np.column_stack(columns)
        else:
            X = np.empty((len(self.results), 0))

        return X, col_info

    def estimate_parameter_effects(
        self,
        oracle_name: str,
        axes: List[AxisConfig],
        hdi_prob: float = 0.94,
    ) -> dict:
        """Estimate per-parameter effect sizes using Bayesian linear regression.

        Args:
            oracle_name: Oracle to analyze.
            axes: List of AxisConfig defining parameter axes.
            hdi_prob: Probability mass for credible intervals.

        Returns:
            Dict with intercept, sorted effects list, noise, sample counts,
            HDI prob.

        Raises:
            ValueError: If axes is empty or no valid predictor columns.
        """
        if not axes:
            raise ValueError("axes must be non-empty for parameter effects analysis")

        scores = np.array([r.evaluations[oracle_name].score for r in self.results])
        X, col_info = self._build_design_matrix(oracle_name, axes)

        if X.shape[1] == 0:
            raise ValueError(
                "No valid predictor columns after filtering zero-variance axes"
            )

        with pm.Model():
            intercept = pm.Normal(
                "intercept",
                mu=self.analysis_config.prior_quality_mu,
                sigma=self.analysis_config.prior_quality_sigma,
            )
            beta = pm.Normal(
                "beta",
                mu=0,
                sigma=self.analysis_config.prior_effect_sigma,
                shape=X.shape[1],
            )
            noise = pm.HalfNormal(
                "oracle_noise",
                sigma=self.analysis_config.prior_noise_sigma,
            )
            mu = intercept + pm.math.dot(X, beta)
            pm.Normal("observed", mu=mu, sigma=noise, observed=scores)

            trace = pm.sample(
                draws=self.analysis_config.mcmc_draws,
                tune=self.analysis_config.mcmc_tune,
                chains=self.analysis_config.mcmc_chains,
            )

        posterior = trace.posterior
        intercept_samples = posterior["intercept"].values.flatten()
        beta_samples = posterior["beta"].values
        noise_samples = posterior["oracle_noise"].values.flatten()

        # Intercept stats
        intercept_hdi = az.hdi(trace, var_names=["intercept"], hdi_prob=hdi_prob)
        noise_hdi = az.hdi(trace, var_names=["oracle_noise"], hdi_prob=hdi_prob)

        # Per-predictor effects
        effects = []
        n_chains, n_draws = beta_samples.shape[0], beta_samples.shape[1]
        flat_beta = beta_samples.reshape(n_chains * n_draws, -1)

        beta_hdi = az.hdi(trace, var_names=["beta"], hdi_prob=hdi_prob)

        for k, info in enumerate(col_info):
            samples_k = flat_beta[:, k]
            hdi_vals = beta_hdi["beta"].values[k]

            effect_entry = {
                "parameter": info["parameter"],
                "type": info["type"],
                "level": info["level"],
                "reference_level": info["reference_level"],
                "effect_mean": float(samples_k.mean()),
                "effect_median": float(np.median(samples_k)),
                "hdi_lower": float(hdi_vals[0]),
                "hdi_upper": float(hdi_vals[1]),
                "prob_positive": float((samples_k > 0).mean()),
                "prob_negative": float((samples_k < 0).mean()),
                "standardization": info["standardization"],
            }
            effects.append(effect_entry)

        # Sort by absolute effect magnitude (descending)
        effects.sort(key=lambda e: abs(e["effect_mean"]), reverse=True)

        return {
            "intercept": {
                "mean": float(intercept_samples.mean()),
                "hdi_lower": float(intercept_hdi["intercept"].values[0]),
                "hdi_upper": float(intercept_hdi["intercept"].values[1]),
            },
            "effects": effects,
            "oracle_noise_mean": float(noise_samples.mean()),
            "oracle_noise_hdi": (
                float(noise_hdi["oracle_noise"].values[0]),
                float(noise_hdi["oracle_noise"].values[1]),
            ),
            "n_samples": len(scores),
            "n_predictors": X.shape[1],
            "hdi_prob": hdi_prob,
        }

    def estimate_multi_judge_quality(
        self,
        oracle_names: List[str],
        hdi_prob: float = 0.94,
        expected_score: Optional[float] = None,
    ) -> dict:
        """Estimate judge calibration from multiple judges using a hierarchical model.

        With expected_score: anchors to the known score and drops true_quality.
        Without: keeps true_quality as a nuisance parameter.

        In both cases, per-judge bias and noise are the primary outputs.

        Args:
            oracle_names: List of oracle names to include in the model.
            hdi_prob: Probability mass for credible intervals.
            expected_score: Known ground-truth score, if available.

        Returns:
            Dict with per-judge bias/noise/consistency_weight, and optionally
            true quality estimates (when expected_score is not provided).
        """
        # Build flat arrays with judge index mapping
        scores_flat = []
        judge_idx = []
        per_judge_scores = {}

        for j, name in enumerate(oracle_names):
            oracle_scores = [
                r.evaluations[name].score for r in self.results if name in r.evaluations
            ]
            per_judge_scores[name] = oracle_scores
            scores_flat.extend(oracle_scores)
            judge_idx.extend([j] * len(oracle_scores))

        scores_array = np.array(scores_flat)
        judge_idx_array = np.array(judge_idx, dtype=int)
        n_judges = len(oracle_names)

        with pm.Model():
            if expected_score is not None:
                anchor = expected_score
            else:
                true_quality = pm.Normal(
                    "true_quality",
                    mu=self.analysis_config.prior_quality_mu,
                    sigma=self.analysis_config.prior_quality_sigma,
                )
                anchor = true_quality

            bias = pm.Normal(
                "bias",
                mu=0,
                sigma=self.analysis_config.prior_bias_sigma,
                shape=n_judges,
            )

            noise = pm.HalfNormal(
                "noise",
                sigma=self.analysis_config.prior_noise_sigma,
                shape=n_judges,
            )

            mu = anchor + bias[judge_idx_array]
            sigma = noise[judge_idx_array]

            pm.Normal("observed", mu=mu, sigma=sigma, observed=scores_array)

            trace = pm.sample(
                draws=self.analysis_config.mcmc_draws,
                tune=self.analysis_config.mcmc_tune,
                chains=self.analysis_config.mcmc_chains,
            )

        posterior = trace.posterior

        # Per-judge stats
        bias_samples = posterior["bias"].values  # (chains, draws, n_judges)
        noise_samples = posterior["noise"].values

        n_chains, n_draws = bias_samples.shape[0], bias_samples.shape[1]
        flat_bias = bias_samples.reshape(n_chains * n_draws, n_judges)
        flat_noise = noise_samples.reshape(n_chains * n_draws, n_judges)

        bias_hdi = az.hdi(trace, var_names=["bias"], hdi_prob=hdi_prob)
        noise_hdi = az.hdi(trace, var_names=["noise"], hdi_prob=hdi_prob)

        judges = {}
        for j, name in enumerate(oracle_names):
            judge_bias = flat_bias[:, j]
            judge_noise = flat_noise[:, j]
            raw_scores = per_judge_scores[name]

            judges[name] = {
                "bias_mean": float(judge_bias.mean()),
                "bias_hdi": (
                    float(bias_hdi["bias"].values[j, 0]),
                    float(bias_hdi["bias"].values[j, 1]),
                ),
                "noise_mean": float(judge_noise.mean()),
                "noise_hdi": (
                    float(noise_hdi["noise"].values[j, 0]),
                    float(noise_hdi["noise"].values[j, 1]),
                ),
                "n_evaluations": len(raw_scores),
                "raw_score_mean": float(np.mean(raw_scores)),
                "raw_score_std": float(np.std(raw_scores)),
                "bias_posterior_samples": judge_bias.tolist(),
                "noise_posterior_samples": judge_noise.tolist(),
            }

        # Consistency weights: 1/noise^2, normalized
        # Use a floor to prevent near-zero noise from dominating
        noise_values = [info["noise_mean"] for info in judges.values()]
        non_tiny = [v for v in noise_values if v > 0.01]
        noise_floor = max(min(non_tiny) if non_tiny else 0.05, 0.01)
        weights = {}
        for name, info in judges.items():
            noise_val = max(info["noise_mean"], noise_floor)
            weights[name] = 1.0 / (noise_val**2)
        total_weight = sum(weights.values())
        for name in judges:
            judges[name]["consistency_weight"] = weights[name] / total_weight

        result = {
            "hdi_prob": hdi_prob,
            "judges": judges,
            "n_judges": n_judges,
            "n_total_evaluations": len(scores_flat),
        }

        if expected_score is not None:
            result["expected_score"] = expected_score
        else:
            # True quality stats (nuisance parameter, secondary output)
            quality_samples = posterior["true_quality"].values.flatten()
            quality_hdi = az.hdi(trace, var_names=["true_quality"], hdi_prob=hdi_prob)

            result["true_quality_mean"] = float(quality_samples.mean())
            result["true_quality_median"] = float(np.median(quality_samples))
            result["true_quality_std"] = float(quality_samples.std())
            result["hdi_lower"] = float(quality_hdi["true_quality"].values[0])
            result["hdi_upper"] = float(quality_hdi["true_quality"].values[1])

            # Bias-corrected weighted score
            bc_num = sum(
                judges[name]["consistency_weight"]
                * (judges[name]["raw_score_mean"] - judges[name]["bias_mean"])
                for name in oracle_names
            )
            result["bias_corrected_weighted_score"] = float(bc_num)

        return result
