from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AxisConfig(BaseModel):
    """Configuration for a parameter axis in the sampling space.

    Defines a single dimension of the parameter space to be explored via sampling.
    Axes can be either categorical (discrete values) or continuous (numeric ranges).

    Attributes:
        name: Name of the parameter (used in template interpolation).
        type: Type of axis - "categorical" for discrete values or "continuous" for numeric.
        values: List of allowed values for categorical axes. Empty for continuous.
        weights: Sampling weights for categorical values. Must sum to 1.0 if provided.
        distribution: Probability distribution for continuous axes.
            Options: "uniform", "normal", "truncnorm", "beta". Required for continuous.
        params: Distribution parameters (e.g., {"low": 0, "high": 10} for uniform).
            Required parameters depend on the distribution type.

    Examples:
        Categorical axis:
            AxisConfig(
                name="tone",
                type="categorical",
                values=["formal", "casual"],
                weights=[0.5, 0.5]
            )

        Continuous uniform axis:
            AxisConfig(
                name="temperature",
                type="continuous",
                distribution="uniform",
                params={"low": 0.0, "high": 1.0}
            )
    """

    name: str
    type: Literal["categorical", "continuous"]
    values: List[Any] = []
    weights: List[float] = []
    distribution: Optional[Literal["uniform", "normal", "truncnorm", "beta"]] = None
    params: Dict[str, float] = {}


class PipelineConfig(BaseModel):
    """Configuration for a single stage in the LLM evaluation pipeline.

    Each pipeline stage defines a template to render with sample parameters,
    and the LLM settings to use for generation. Multiple stages can be chained
    where each stage uses the previous stage's output as input.

    Attributes:
        template: Jinja2 template string for the prompt. Can reference axis parameters
            using {{ parameter_name }} syntax.
        adapter: Name of the LLM adapter to use (e.g., "ollama", "openai").
        model: Model identifier for the adapter (e.g., "gpt-4", "llama2").
        temperature: Sampling temperature for generation. Range: [0.0, 2.0].
            Lower values = more deterministic, higher values = more random.
        top_p: Nucleus sampling parameter. Range: (0.0, 1.0].
            Smaller values = more focused, larger values = more diverse.
        max_tokens: Maximum number of tokens to generate in the response.

    Example:
        PipelineConfig(
            template="Explain {{ topic }} in a {{ tone }} tone.",
            adapter="ollama",
            model="llama2",
            temperature=0.7,
            top_p=0.9,
            max_tokens=500
        )
    """

    template: str
    adapter: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1)
    max_tokens: int


class SamplingConfig(BaseModel):
    """Configuration for parameter space sampling strategy.

    Defines how to sample points from the parameter space defined by the axes.
    Latin Hypercube Sampling ensures good coverage of the parameter space.

    Attributes:
        method: Sampling method to use. Currently only "latin_hypercube" is supported.
        optimization: Optimization criterion for Latin Hypercube Sampling.
            "maximin" maximizes the minimum distance between samples for better coverage.
        random_seed: Random seed for reproducible sampling. If None, sampling is non-deterministic.

    Example:
        SamplingConfig(
            method="latin_hypercube",
            optimization="maximin",
            random_seed=42  # For reproducibility
        )
    """

    method: Literal["latin_hypercube"]
    optimization: Literal["maximin"]
    random_seed: Optional[int] = None


class OracleConfig(BaseModel):
    """Configuration for an evaluation oracle.

    Oracles evaluate LLM responses and provide scores. Currently supports
    LLM Judge oracles that use another LLM to judge response quality.

    Attributes:
        type: Type of oracle. Currently only "llm_judge" is supported.
        model: Model identifier for the judge LLM (e.g., "gpt-4", "llama2").
        adapter: Name of the LLM adapter to use for the judge.
        max_tokens: Maximum tokens for the judge's evaluation response (default: 2000).
        temperature: Sampling temperature for the judge LLM (default: 1).
        rubric: Evaluation rubric or criteria for the judge. Should instruct the judge
            to return JSON with "score" (1-5) and "explanation" fields.

    Example:
        OracleConfig(
            type="llm_judge",
            model="gpt-4",
            adapter="openai",
            rubric='''
                Evaluate coherence on a 1-5 scale:
                1 = Incoherent, 5 = Perfectly coherent
                Return: {"score": X, "explanation": "..."}
            '''
        )
    """

    type: Literal["llm_judge"]
    model: str
    adapter: str
    max_tokens: int = 2000
    temperature: Optional[int] = 1
    rubric: Optional[str] = None


class BayesianAnalysisConfig(BaseModel):
    """Configuration for Bayesian analysis parameters.

    This configuration controls the PyMC Bayesian modeling and MCMC sampling
    parameters used in posterior inference.

    Attributes:
        mcmc_draws: Number of MCMC samples to draw per chain (default: 2000).
            More draws = better posterior approximation but slower sampling.
        mcmc_tune: Number of tuning/warmup samples (default: 1000).
            Tuning samples are discarded and used to adapt the sampler.
        mcmc_chains: Number of independent MCMC chains to run (default: 4).
            Multiple chains help detect convergence issues.
        prior_quality_mu: Prior mean for true quality scores (default: 3.0).
            Represents our belief about average quality on the 1-5 scale.
        prior_quality_sigma: Prior standard deviation for true quality (default: 1.0).
            Controls how much we allow quality to vary from the mean a priori.
        prior_noise_sigma: Prior standard deviation for oracle noise (default: 0.5).
            Represents our belief about oracle measurement error.
        hdi_probability: Probability mass for credible intervals (default: 0.94).
            Common values: 0.89, 0.94, 0.95. This determines the width of
            the credible interval in statements like "94% confident quality is between X and Y".
    """

    # MCMC sampling parameters
    mcmc_draws: int = Field(default=2000, ge=100)
    mcmc_tune: int = Field(default=1000, ge=100)
    mcmc_chains: int = Field(default=4, ge=1)

    # Prior parameters for calibration model
    prior_quality_mu: float = Field(default=3.0, ge=0.0, le=5.0)
    prior_quality_sigma: float = Field(default=1.0, gt=0.0)
    prior_noise_sigma: float = Field(default=0.5, gt=0.0)

    # High-Density Interval (HDI) configuration
    hdi_probability: float = Field(default=0.94, gt=0.0, lt=1.0)


class SpecConfig(BaseModel):
    """Complete specification for an LLM evaluation experiment.

    This is the top-level configuration that defines an entire evaluation run,
    including the parameter space to explore, the LLM pipeline to execute,
    the oracles to use for evaluation, and optional Bayesian analysis settings.

    Attributes:
        spec_id: Unique identifier for this specification.
        pipeline: List of pipeline stages to execute sequentially. At least one required.
            First stage uses template with sample parameters. Subsequent stages use
            previous stage's output as input.
        sampling: Configuration for parameter space sampling strategy.
        n_variants: Number of parameter variants to generate and evaluate (default: 1).
        oracles: Dictionary mapping oracle names to their configurations. At least one required.
            Oracle names are used to reference evaluation results.
        axes: List of parameter axes defining the exploration space. Can be empty for
            evaluations that don't vary parameters.
        analysis: Optional Bayesian analysis configuration. If None, no Bayesian analysis
            is performed. If provided, enables posterior inference on evaluation results.

    Example:
        SpecConfig(
            spec_id="coherence_test_v1",
            pipeline=[PipelineConfig(...)],
            sampling=SamplingConfig(method="latin_hypercube", ...),
            n_variants=10,
            oracles={"coherence": OracleConfig(...), "accuracy": OracleConfig(...)},
            axes=[AxisConfig(name="tone", ...), AxisConfig(name="complexity", ...)],
            analysis=BayesianAnalysisConfig(mcmc_draws=2000, ...)
        )
    """

    spec_id: str
    pipeline: List[PipelineConfig] = Field(..., min_length=1)
    sampling: SamplingConfig
    n_variants: int = 1
    oracles: Dict[str, OracleConfig] = Field(..., min_length=1)
    axes: List[AxisConfig] = []
    analysis: Optional[BayesianAnalysisConfig] = None
