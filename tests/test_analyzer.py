import numpy as np
import pytest

from metareason.analysis.analyzer import BayesianAnalyzer
from metareason.config.models import (
    AdapterConfig,
    AxisConfig,
    BayesianAnalysisConfig,
    OracleConfig,
    PipelineConfig,
    SamplingConfig,
    SpecConfig,
)
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.runner import SampleResult


def make_spec(**overrides):
    defaults = dict(
        spec_id="test",
        pipeline=[
            PipelineConfig(
                template="t",
                adapter=AdapterConfig(name="ollama"),
                model="m",
                temperature=0.7,
                top_p=0.9,
                max_tokens=100,
            )
        ],
        sampling=SamplingConfig(method="latin_hypercube", optimization="maximin"),
        oracles={
            "test_oracle": OracleConfig(
                type="llm_judge",
                model="m",
                adapter=AdapterConfig(name="ollama"),
                rubric="test",
            )
        },
    )
    defaults.update(overrides)
    return SpecConfig(**defaults)


def make_results(n=5, score=3.5):
    return [
        SampleResult(
            sample_params={"param": i},
            original_prompt="test prompt",
            final_response="test response",
            evaluations={
                "test_oracle": EvaluationResult(score=score, explanation="ok")
            },
        )
        for i in range(n)
    ]


class TestBayesianAnalyzerInit:
    def test_init_defaults(self):
        results = make_results()
        spec = make_spec()
        analyzer = BayesianAnalyzer(results, spec)

        assert analyzer.n_variants == 5
        assert analyzer.analysis_config.mcmc_draws == 2000
        assert analyzer.analysis_config.mcmc_tune == 1000
        assert analyzer.analysis_config.mcmc_chains == 4

    def test_init_custom_config(self):
        results = make_results()
        custom = BayesianAnalysisConfig(mcmc_draws=200, mcmc_tune=100, mcmc_chains=2)
        spec = make_spec(analysis=custom)
        analyzer = BayesianAnalyzer(results, spec)

        assert analyzer.analysis_config.mcmc_draws == 200
        assert analyzer.analysis_config.mcmc_tune == 100
        assert analyzer.analysis_config.mcmc_chains == 2

    def test_init_custom_effect_sigma(self):
        results = make_results()
        custom = BayesianAnalysisConfig(
            mcmc_draws=200,
            mcmc_tune=100,
            mcmc_chains=1,
            prior_effect_sigma=2.0,
        )
        spec = make_spec(analysis=custom)
        analyzer = BayesianAnalyzer(results, spec)
        assert analyzer.analysis_config.prior_effect_sigma == 2.0

    def test_init_default_effect_sigma(self):
        results = make_results()
        spec = make_spec()
        analyzer = BayesianAnalyzer(results, spec)
        assert analyzer.analysis_config.prior_effect_sigma == 1.0


@pytest.mark.slow
class TestBayesianAnalyzerSampling:
    def test_estimate_population_quality_returns_expected_keys(self):
        results = make_results(n=5, score=3.5)
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            )
        )
        analyzer = BayesianAnalyzer(results, spec)
        result = analyzer.estimate_population_quality("test_oracle")

        expected_keys = {
            "population_mean",
            "population_median",
            "population_std",
            "hdi_lower",
            "hdi_upper",
            "hdi_prob",
            "oracle_noise_mean",
            "oracle_noise_hdi",
            "n_samples",
            "posterior_samples",
            "noise_posterior_samples",
        }
        assert set(result.keys()) == expected_keys
        assert result["n_samples"] == 5
        assert result["hdi_prob"] == 0.94

    def test_estimate_population_quality_hdi_bounds(self):
        results = make_results(n=5, score=3.5)
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            )
        )
        analyzer = BayesianAnalyzer(results, spec)
        result = analyzer.estimate_population_quality("test_oracle")

        assert result["hdi_lower"] < result["hdi_upper"]
        assert result["hdi_lower"] > 0.0
        assert result["hdi_upper"] < 7.0


# --- Design matrix + parameter effects helpers ---


def make_axes():
    """Create a mixed set of axes for testing."""
    return [
        AxisConfig(
            name="temperature",
            type="continuous",
            distribution="uniform",
            params={"low": 0.0, "high": 1.0},
        ),
        AxisConfig(
            name="tone",
            type="categorical",
            values=["formal", "casual", "technical"],
        ),
    ]


def make_parametric_results(params_list, scores, oracle_name="test_oracle"):
    """Create SampleResults with specific params and scores."""
    return [
        SampleResult(
            sample_params=params,
            original_prompt="test prompt",
            final_response="test response",
            evaluations={oracle_name: EvaluationResult(score=score, explanation="ok")},
        )
        for params, score in zip(params_list, scores)
    ]


class TestBuildDesignMatrix:
    def test_continuous_only(self):
        axes = [
            AxisConfig(
                name="temp",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            ),
        ]
        params_list = [{"temp": 0.2}, {"temp": 0.4}, {"temp": 0.6}, {"temp": 0.8}]
        scores = [3.0, 3.5, 4.0, 4.5]
        results = make_parametric_results(params_list, scores)
        analyzer = BayesianAnalyzer(results, make_spec())

        X, col_info = analyzer._build_design_matrix("test_oracle", axes)

        assert X.shape == (4, 1)
        # Z-standardized: mean ~0, std ~1
        assert abs(X[:, 0].mean()) < 1e-10
        assert abs(X[:, 0].std(ddof=0) - 1.0) < 1e-10
        assert col_info[0]["parameter"] == "temp"
        assert col_info[0]["type"] == "continuous"
        assert col_info[0]["level"] is None

    def test_categorical_reference_coding(self):
        axes = [
            AxisConfig(
                name="tone",
                type="categorical",
                values=["formal", "casual", "technical"],
            ),
        ]
        params_list = [
            {"tone": "formal"},
            {"tone": "casual"},
            {"tone": "technical"},
            {"tone": "formal"},
        ]
        scores = [3.0, 3.5, 4.0, 4.5]
        results = make_parametric_results(params_list, scores)
        analyzer = BayesianAnalyzer(results, make_spec())

        X, col_info = analyzer._build_design_matrix("test_oracle", axes)

        # 3 categories -> 2 dummy columns (formal is reference)
        assert X.shape == (4, 2)
        # Row 0: formal -> [0, 0]
        assert list(X[0]) == [0.0, 0.0]
        # Row 1: casual -> [1, 0]
        assert list(X[1]) == [1.0, 0.0]
        # Row 2: technical -> [0, 1]
        assert list(X[2]) == [0.0, 1.0]
        assert col_info[0]["reference_level"] == "formal"
        assert col_info[0]["level"] == "casual"
        assert col_info[1]["level"] == "technical"

    def test_mixed_axes(self):
        axes = make_axes()  # temperature (continuous) + tone (categorical, 3 levels)
        params_list = [
            {"temperature": 0.2, "tone": "formal"},
            {"temperature": 0.8, "tone": "casual"},
        ]
        scores = [3.0, 4.0]
        results = make_parametric_results(params_list, scores)
        analyzer = BayesianAnalyzer(results, make_spec())

        X, col_info = analyzer._build_design_matrix("test_oracle", axes)

        # 1 continuous + 2 dummy = 3 columns
        assert X.shape == (2, 3)
        assert len(col_info) == 3

    def test_zero_variance_continuous_skipped(self):
        axes = [
            AxisConfig(
                name="temp",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            ),
        ]
        params_list = [{"temp": 0.5}, {"temp": 0.5}, {"temp": 0.5}]
        scores = [3.0, 3.5, 4.0]
        results = make_parametric_results(params_list, scores)
        analyzer = BayesianAnalyzer(results, make_spec())

        X, col_info = analyzer._build_design_matrix("test_oracle", axes)

        # Zero-variance axis should be skipped
        assert X.shape == (3, 0)
        assert len(col_info) == 0


@pytest.mark.slow
class TestParameterEffects:
    def _make_linear_data(self, n=20, true_effect=0.5, noise=0.1, seed=42):
        """Generate data where score = 3.0 + true_effect * z(x) + noise."""
        rng = np.random.default_rng(seed)
        x_values = np.linspace(0, 1, n)
        scores = (
            3.0
            + true_effect * (x_values - x_values.mean()) / x_values.std()
            + rng.normal(0, noise, n)
        )
        params_list = [{"x": float(v)} for v in x_values]
        return params_list, scores.tolist()

    def test_returns_expected_keys(self):
        params, scores = self._make_linear_data()
        results = make_parametric_results(params, scores)
        axes = [
            AxisConfig(
                name="x",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            )
        ]
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            )
        )
        analyzer = BayesianAnalyzer(results, spec)

        result = analyzer.estimate_parameter_effects("test_oracle", axes)

        assert "intercept" in result
        assert "effects" in result
        assert "oracle_noise_mean" in result
        assert "n_samples" in result
        assert "n_predictors" in result
        assert "hdi_prob" in result
        assert len(result["effects"]) == 1

    def test_continuous_effect_recovery(self):
        """Verify regression recovers a known positive effect."""
        params, scores = self._make_linear_data(n=30, true_effect=0.5, noise=0.1)
        results = make_parametric_results(params, scores)
        axes = [
            AxisConfig(
                name="x",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            )
        ]
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            )
        )
        analyzer = BayesianAnalyzer(results, spec)

        result = analyzer.estimate_parameter_effects("test_oracle", axes)
        effect = result["effects"][0]

        # Effect should be positive and HDI should not contain zero
        assert effect["effect_mean"] > 0
        assert effect["hdi_lower"] > 0
        assert effect["prob_positive"] > 0.9

    def test_categorical_effect_recovery(self):
        """Verify regression detects a categorical effect."""
        rng = np.random.default_rng(42)
        params_list = [{"tone": "formal"}] * 15 + [{"tone": "casual"}] * 15
        scores = [4.0 + rng.normal(0, 0.1) for _ in range(15)] + [
            2.0 + rng.normal(0, 0.1) for _ in range(15)
        ]
        results = make_parametric_results(params_list, scores)
        axes = [
            AxisConfig(name="tone", type="categorical", values=["formal", "casual"])
        ]
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            )
        )
        analyzer = BayesianAnalyzer(results, spec)

        result = analyzer.estimate_parameter_effects("test_oracle", axes)
        effect = result["effects"][0]

        # casual vs formal: expect negative effect ~-2.0
        assert effect["effect_mean"] < 0
        assert effect["prob_negative"] > 0.9

    def test_no_axes_raises(self):
        results = make_results()
        analyzer = BayesianAnalyzer(results, make_spec())
        with pytest.raises(ValueError, match="axes"):
            analyzer.estimate_parameter_effects("test_oracle", [])

    def test_effects_sorted_by_magnitude(self):
        """Two axes: one with large effect, one with small."""
        rng = np.random.default_rng(42)
        n = 30
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        rng.shuffle(x2)
        # x1 has large effect (1.0), x2 has small effect (0.1)
        scores = (
            3.0
            + 1.0 * (x1 - x1.mean()) / x1.std()
            + 0.1 * (x2 - x2.mean()) / x2.std()
            + rng.normal(0, 0.1, n)
        )
        params_list = [{"big": float(a), "small": float(b)} for a, b in zip(x1, x2)]
        results = make_parametric_results(params_list, scores.tolist())
        axes = [
            AxisConfig(
                name="big",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            ),
            AxisConfig(
                name="small",
                type="continuous",
                distribution="uniform",
                params={"low": 0, "high": 1},
            ),
        ]
        spec = make_spec(
            analysis=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            )
        )
        analyzer = BayesianAnalyzer(results, spec)

        result = analyzer.estimate_parameter_effects("test_oracle", axes)

        assert abs(result["effects"][0]["effect_mean"]) >= abs(
            result["effects"][1]["effect_mean"]
        )


# --- Multi-judge hierarchical model ---


def make_multi_judge_results(n=20, oracle_scores=None, oracle_names=None, seed=42):
    """Create SampleResults with multiple oracle evaluations."""
    rng = np.random.default_rng(seed)
    if oracle_names is None:
        oracle_names = ["judge_a", "judge_b"]
    if oracle_scores is None:
        oracle_scores = {
            name: [3.5 + rng.normal(0, 0.3) for _ in range(n)] for name in oracle_names
        }

    results = []
    for i in range(n):
        evaluations = {}
        for name in oracle_names:
            if i < len(oracle_scores[name]):
                score = max(1.0, min(5.0, oracle_scores[name][i]))
                evaluations[name] = EvaluationResult(score=score, explanation="ok")
        results.append(
            SampleResult(
                sample_params={"param": i},
                original_prompt="test prompt",
                final_response="test response",
                evaluations=evaluations,
            )
        )
    return results


@pytest.mark.slow
class TestMultiJudgeQuality:
    def test_returns_expected_keys_without_expected_score(self):
        results = make_multi_judge_results(n=10)
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            ),
        )
        result = analyzer.estimate_multi_judge_quality(["judge_a", "judge_b"])

        expected_keys = {
            "true_quality_mean",
            "true_quality_median",
            "true_quality_std",
            "hdi_lower",
            "hdi_upper",
            "hdi_prob",
            "judges",
            "bias_corrected_weighted_score",
            "n_judges",
            "n_total_evaluations",
        }
        assert set(result.keys()) == expected_keys
        assert result["n_judges"] == 2
        assert result["n_total_evaluations"] == 20
        assert result["hdi_prob"] == 0.94

        # Check per-judge keys
        judge_keys = {
            "bias_mean",
            "bias_hdi",
            "noise_mean",
            "noise_hdi",
            "n_evaluations",
            "raw_score_mean",
            "raw_score_std",
            "consistency_weight",
            "bias_posterior_samples",
            "noise_posterior_samples",
        }
        for name in ["judge_a", "judge_b"]:
            assert name in result["judges"]
            assert set(result["judges"][name].keys()) == judge_keys

    def test_returns_expected_keys_with_expected_score(self):
        results = make_multi_judge_results(n=10)
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            ),
        )
        result = analyzer.estimate_multi_judge_quality(
            ["judge_a", "judge_b"], expected_score=3.5
        )

        expected_keys = {
            "expected_score",
            "hdi_prob",
            "judges",
            "n_judges",
            "n_total_evaluations",
        }
        assert set(result.keys()) == expected_keys
        assert result["expected_score"] == 3.5
        assert "true_quality_mean" not in result

    def test_detects_bias(self):
        """Judge with +1.0 offset should have positive bias."""
        rng = np.random.default_rng(42)
        n = 20
        oracle_scores = {
            "unbiased": [3.0 + rng.normal(0, 0.2) for _ in range(n)],
            "biased": [4.0 + rng.normal(0, 0.2) for _ in range(n)],
        }
        results = make_multi_judge_results(
            n=n,
            oracle_scores=oracle_scores,
            oracle_names=["unbiased", "biased"],
        )
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            ),
        )
        result = analyzer.estimate_multi_judge_quality(["unbiased", "biased"])

        # Biased judge should have higher bias than unbiased
        assert (
            result["judges"]["biased"]["bias_mean"]
            > result["judges"]["unbiased"]["bias_mean"]
        )

    def test_noisy_judge_higher_noise(self):
        """Noisy judge gets higher noise estimate."""
        rng = np.random.default_rng(42)
        n = 30
        oracle_scores = {
            "precise": [3.5 + rng.normal(0, 0.1) for _ in range(n)],
            "noisy": [3.5 + rng.normal(0, 0.8) for _ in range(n)],
        }
        results = make_multi_judge_results(
            n=n,
            oracle_scores=oracle_scores,
            oracle_names=["precise", "noisy"],
        )
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=200, mcmc_tune=200, mcmc_chains=2
            ),
        )
        result = analyzer.estimate_multi_judge_quality(["precise", "noisy"])

        assert (
            result["judges"]["noisy"]["noise_mean"]
            > result["judges"]["precise"]["noise_mean"]
        )

    def test_consistency_weights_sum_to_one(self):
        results = make_multi_judge_results(n=15)
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            ),
        )
        result = analyzer.estimate_multi_judge_quality(["judge_a", "judge_b"])

        total = sum(info["consistency_weight"] for info in result["judges"].values())
        assert abs(total - 1.0) < 1e-6

    def test_handles_ragged_data(self):
        """Different eval counts per judge."""
        rng = np.random.default_rng(42)
        # Judge A has 20 evals, Judge B has 15
        oracle_scores = {
            "judge_a": [3.5 + rng.normal(0, 0.3) for _ in range(20)],
            "judge_b": [3.5 + rng.normal(0, 0.3) for _ in range(15)],
        }
        results = make_multi_judge_results(
            n=20,
            oracle_scores=oracle_scores,
            oracle_names=["judge_a", "judge_b"],
        )
        analyzer = BayesianAnalyzer(
            results,
            analysis_config=BayesianAnalysisConfig(
                mcmc_draws=100, mcmc_tune=100, mcmc_chains=1
            ),
        )
        result = analyzer.estimate_multi_judge_quality(["judge_a", "judge_b"])

        assert result["judges"]["judge_a"]["n_evaluations"] == 20
        assert result["judges"]["judge_b"]["n_evaluations"] == 15
        assert result["n_total_evaluations"] == 35
