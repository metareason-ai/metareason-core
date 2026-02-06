import pytest
from pydantic import ValidationError

from metareason.config.models import (
    AdapterConfig,
    AxisConfig,
    BayesianAnalysisConfig,
    OracleConfig,
    PipelineConfig,
    SamplingConfig,
    SpecConfig,
)

# --- Helpers ---


def make_pipeline(**overrides):
    defaults = dict(
        template="Hello {{ name }}",
        adapter=AdapterConfig(name="ollama"),
        model="llama2",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def make_oracle(**overrides):
    defaults = dict(
        type="llm_judge",
        model="llama2",
        adapter=AdapterConfig(name="ollama"),
        rubric="Score 1-5",
    )
    defaults.update(overrides)
    return OracleConfig(**defaults)


def make_sampling(**overrides):
    defaults = dict(method="latin_hypercube", optimization="maximin")
    defaults.update(overrides)
    return SamplingConfig(**defaults)


# --- AdapterConfig ---


class TestAdapterConfig:
    def test_valid_adapters(self):
        for name in ("ollama", "google", "openai", "anthropic"):
            cfg = AdapterConfig(name=name)
            assert cfg.name == name
            assert cfg.params == {}

    def test_with_params(self):
        cfg = AdapterConfig(name="google", params={"api_key": "xxx"})
        assert cfg.params["api_key"] == "xxx"

    def test_invalid_adapter_name(self):
        with pytest.raises(ValidationError):
            AdapterConfig(name="invalid")


# --- AxisConfig ---


class TestAxisConfig:
    def test_categorical_axis(self):
        axis = AxisConfig(name="tone", type="categorical", values=["formal", "casual"])
        assert axis.name == "tone"
        assert axis.type == "categorical"
        assert axis.values == ["formal", "casual"]

    def test_continuous_axis(self):
        axis = AxisConfig(
            name="temp",
            type="continuous",
            distribution="uniform",
            params={"min": 0.0, "max": 1.0},
        )
        assert axis.distribution == "uniform"
        assert axis.params["min"] == 0.0

    def test_defaults(self):
        axis = AxisConfig(name="x", type="categorical")
        assert axis.values == []
        assert axis.weights == []
        assert axis.distribution is None
        assert axis.params == {}

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            AxisConfig(name="x", type="discrete")

    def test_invalid_distribution(self):
        with pytest.raises(ValidationError):
            AxisConfig(name="x", type="continuous", distribution="exponential")


# --- PipelineConfig ---


class TestPipelineConfig:
    def test_valid_pipeline(self):
        pipe = make_pipeline()
        assert pipe.template == "Hello {{ name }}"
        assert pipe.temperature == 0.7

    def test_temperature_too_high(self):
        with pytest.raises(ValidationError):
            make_pipeline(temperature=5.0)

    def test_temperature_negative(self):
        with pytest.raises(ValidationError):
            make_pipeline(temperature=-1.0)

    def test_top_p_zero(self):
        with pytest.raises(ValidationError):
            make_pipeline(top_p=0.0)

    def test_top_p_above_one(self):
        with pytest.raises(ValidationError):
            make_pipeline(top_p=1.5)

    def test_boundary_values(self):
        pipe = make_pipeline(temperature=0.0, top_p=1.0)
        assert pipe.temperature == 0.0
        assert pipe.top_p == 1.0


# --- SamplingConfig ---


class TestSamplingConfig:
    def test_valid_sampling(self):
        cfg = make_sampling(random_seed=42)
        assert cfg.method == "latin_hypercube"
        assert cfg.random_seed == 42

    def test_defaults(self):
        cfg = make_sampling()
        assert cfg.random_seed is None

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            SamplingConfig(method="random", optimization="maximin")

    def test_invalid_optimization(self):
        with pytest.raises(ValidationError):
            SamplingConfig(method="latin_hypercube", optimization="random")


# --- OracleConfig ---


class TestOracleConfig:
    def test_valid_oracle(self):
        cfg = make_oracle()
        assert cfg.type == "llm_judge"
        assert cfg.max_tokens == 2000
        assert cfg.temperature == 1

    def test_defaults(self):
        cfg = make_oracle()
        assert cfg.max_tokens == 2000
        assert cfg.temperature == 1

    def test_rubric_optional(self):
        cfg = OracleConfig(
            type="llm_judge",
            model="m",
            adapter=AdapterConfig(name="ollama"),
        )
        assert cfg.rubric is None

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            OracleConfig(
                type="regex",
                model="m",
                adapter=AdapterConfig(name="ollama"),
            )


# --- BayesianAnalysisConfig ---


class TestBayesianAnalysisConfig:
    def test_defaults(self):
        cfg = BayesianAnalysisConfig()
        assert cfg.mcmc_draws == 2000
        assert cfg.mcmc_tune == 1000
        assert cfg.mcmc_chains == 4
        assert cfg.prior_quality_mu == 3.0
        assert cfg.prior_quality_sigma == 1.0
        assert cfg.prior_noise_sigma == 0.5
        assert cfg.hdi_probability == 0.94

    def test_custom_values(self):
        cfg = BayesianAnalysisConfig(
            mcmc_draws=500, mcmc_tune=200, mcmc_chains=2, hdi_probability=0.89
        )
        assert cfg.mcmc_draws == 500
        assert cfg.hdi_probability == 0.89

    def test_draws_too_low(self):
        with pytest.raises(ValidationError):
            BayesianAnalysisConfig(mcmc_draws=50)

    def test_tune_too_low(self):
        with pytest.raises(ValidationError):
            BayesianAnalysisConfig(mcmc_tune=50)

    def test_chains_zero(self):
        with pytest.raises(ValidationError):
            BayesianAnalysisConfig(mcmc_chains=0)

    def test_hdi_out_of_range(self):
        with pytest.raises(ValidationError):
            BayesianAnalysisConfig(hdi_probability=1.0)
        with pytest.raises(ValidationError):
            BayesianAnalysisConfig(hdi_probability=0.0)


# --- SpecConfig ---


class TestSpecConfig:
    def test_valid_spec(self):
        spec = SpecConfig(
            spec_id="test-1",
            pipeline=[make_pipeline()],
            sampling=make_sampling(),
            n_variants=5,
            oracles={"judge": make_oracle()},
        )
        assert spec.spec_id == "test-1"
        assert spec.n_variants == 5
        assert len(spec.pipeline) == 1
        assert "judge" in spec.oracles

    def test_defaults(self):
        spec = SpecConfig(
            spec_id="test",
            pipeline=[make_pipeline()],
            sampling=make_sampling(),
            oracles={"j": make_oracle()},
        )
        assert spec.n_variants == 1
        assert spec.axes == []
        assert spec.analysis is None

    def test_empty_pipeline_rejected(self):
        with pytest.raises(ValidationError):
            SpecConfig(
                spec_id="test",
                pipeline=[],
                sampling=make_sampling(),
                oracles={"j": make_oracle()},
            )

    def test_empty_oracles_rejected(self):
        with pytest.raises(ValidationError):
            SpecConfig(
                spec_id="test",
                pipeline=[make_pipeline()],
                sampling=make_sampling(),
                oracles={},
            )

    def test_with_axes_and_analysis(self):
        spec = SpecConfig(
            spec_id="full",
            pipeline=[make_pipeline()],
            sampling=make_sampling(),
            oracles={"j": make_oracle()},
            axes=[AxisConfig(name="tone", type="categorical", values=["a", "b"])],
            analysis=BayesianAnalysisConfig(mcmc_draws=500),
        )
        assert len(spec.axes) == 1
        assert spec.analysis.mcmc_draws == 500
