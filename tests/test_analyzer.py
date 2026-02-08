import pytest

from metareason.analysis.analyzer import BayesianAnalyzer
from metareason.config.models import (
    AdapterConfig,
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
