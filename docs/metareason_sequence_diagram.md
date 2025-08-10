# MetaReason Core: Architecture & Sequence Diagram

## Overview
MetaReason is a statistical evaluation framework for Large Language Models that combines Latin Hypercube Sampling, template-based prompt generation, and multi-oracle evaluation to provide rigorous statistical confidence intervals.

## Core Architecture Flow

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant CM as cli.main
    participant CC as cli.config
    participant CL as config.loader
    participant CI as config.includes
    participant CE as config.environment
    participant CCH as config.cache
    participant M as config.models
    participant SF as sampling.factory
    participant LHS as sampling.lhs
    participant TI as templates.integration
    participant TE as templates.engine
    participant TR as templates.renderer
    participant AF as adapters.factory
    participant AR as adapters.registry
    participant OA as adapters.openai
    participant AA as adapters.anthropic
    participant OE as oracles.evaluation
    participant SA as analysis.statistical

    Note over U,SA: 1. Configuration Loading & Validation
    U->>CM: metareason run config.yaml
    CM->>CC: config_group.validate()
    CC->>CL: load_yaml_config(file_path)

    CL->>CCH: get_global_cache().get(path)
    alt Cache Hit
        CCH-->>CL: cached EvaluationConfig
    else Cache Miss
        CL->>CI: process_includes_and_inheritance(path)
        CI->>CE: substitute_environment_variables(data)
        CE-->>CI: processed data
        CI-->>CL: merged YAML data
        LHS->>M: EvaluationConfig.model_validate(data)
        M-->>CL: validated config
        CL->>CCH: cache.set(path, config)
    end
    CL-->>CC: EvaluationConfig
    CC-->>CM: validation result

    Note over U,SA: 2. Sampling Strategy Initialization
    CM->>SF: create_sampler(config.axes, config.sampling)
    SF->>LHS: LatinHypercubeSampler(axes, n_samples, optimization)
    LHS-->>SF: sampler instance
    SF-->>CM: BaseSampler

    Note over U,SA: 3. Sample Generation
    CM->>LHS: generate_samples()
    LHS->>LHS: _optimize_design(criterion)
    LHS->>LHS: _apply_scrambling()
    LHS->>LHS: _transform_to_axes()
    LHS-->>CM: SampleResult(samples, metrics, metadata)

    Note over U,SA: 4. Template Processing & Prompt Generation
    CM->>TI: PromptGenerator(config)
    TI->>TE: TemplateEngine()
    TI->>TR: BatchRenderer(engine, batch_size)
    CM->>TI: generate_from_samples(sample_result)

    TI->>TI: validate_template()
    loop For each sample batch
        TI->>TR: render_batch(template, contexts)
        TR->>TE: render(template, context)
        TE-->>TR: rendered_prompt
        TR-->>TI: RenderResult
    end
    TI-->>CM: PromptGenerationResult(prompts, contexts)

    Note over U,SA: 5. Adapter Setup & LLM Interaction
    CM->>AF: AdapterFactory.create(config.adapters)
    AF->>AR: get_adapter_class(config.type)

    alt OpenAI Adapter
        AR-->>AF: OpenAIAdapter class
        AF->>OA: OpenAIAdapter(config)
        OA-->>AF: adapter instance
    else Anthropic Adapter
        AR-->>AF: AnthropicAdapter class
        AF->>AA: AnthropicAdapter(config)
        AA-->>AF: adapter instance
    end
    AF-->>CM: LLMAdapter

    Note over U,SA: 6. Batch LLM Processing
    loop For each prompt batch
        CM->>OA: complete_async(CompletionRequest)
        OA->>OA: _handle_rate_limits()
        OA->>OA: _retry_with_backoff()
        OA-->>CM: CompletionResponse[]
    end

    Note over U,SA: 7. Oracle Evaluation
    loop For each oracle (accuracy, explainability, calibration)
        CM->>OE: evaluate_responses(responses, oracle_config)

        alt Embedding Similarity Oracle
            OE->>OE: compute_embeddings(responses)
            OE->>OE: cosine_similarity(canonical_embedding)
        else LLM Judge Oracle
            OE->>AF: create_judge_adapter(oracle.judge_model)
            OE->>OA: complete_async(judge_prompt)
            OA-->>OE: judgment_response
        else Statistical Calibration Oracle
            OE->>OE: analyze_confidence_calibration()
        end
        OE-->>CM: OracleResult(scores, metadata)
    end

    Note over U,SA: 8. Statistical Analysis & Confidence Intervals
    CM->>SA: BayesianAnalyzer(config.statistical_config)
    CM->>SA: analyze_results(oracle_results, sample_metadata)
    SA->>SA: fit_beta_binomial_model()
    SA->>SA: mcmc_sampling(chains=4, samples=4000)
    SA->>SA: compute_hdi(credible_interval=0.95)
    SA-->>CM: AnalysisResult(posteriors, confidence_intervals)

    Note over U,SA: 9. Results Output & Reporting
    CM->>CM: format_results(analysis_result, output_format)
    CM-->>U: Evaluation Results with Confidence Intervals
```

## Key Components & Classes

### 1. Configuration System (`metareason.config`)
- **`EvaluationConfig`**: Main configuration model with full validation
- **`ConfigLoader.load_yaml_config()`**: Entry point with caching and includes
- **`IncludeProcessor.process_includes_and_inheritance()`**: YAML composition
- **`EnvironmentSubstitutionEngine.substitute_variables()`**: Env var handling
- **`GlobalCache.get()/set()`**: Configuration caching with TTL

### 2. Sampling System (`metareason.sampling`)
- **`SamplerFactory.create_sampler()`**: Factory for different sampling strategies
- **`LatinHypercubeSampler`**: Core LHS implementation with optimization
  - `generate_samples() -> SampleResult`
  - `_optimize_design(criterion: str)` (maximin, centermaximin, correlation)
  - `_apply_scrambling()` for enhanced space-filling
- **`SampleResult`**: Container for samples + metadata + quality metrics

### 3. Template System (`metareason.templates`)
- **`PromptGenerator`**: High-level orchestrator
- **`TemplateEngine`**: Jinja2 wrapper with custom filters
- **`BatchRenderer`**: Parallel template rendering
- **`TemplateValidator`**: Template validation with variable checking
- **`RenderResult`**: Batch rendering results with success rates

### 4. Adapter System (`metareason.adapters`)
- **`AdapterRegistry`**: Plugin registry with lazy loading
- **`AdapterFactory.create()`**: Factory with configuration mapping
- **`LLMAdapter`** (abstract): Base interface
  - `complete_async(request: CompletionRequest) -> CompletionResponse`
  - `stream_async()` for streaming responses
- **`BaseHTTPAdapter`**: HTTP client with rate limiting and retries
- **`OpenAIAdapter`**: OpenAI API implementation
- **`AnthropicAdapter`**: Anthropic Claude API implementation

### 5. Oracle System (`metareason.oracles`)
- **`OracleEvaluator`**: Multi-oracle coordinator
- **`EmbeddingSimilarityOracle`**: Semantic similarity scoring
- **`LLMJudgeOracle`**: LLM-as-judge evaluation
- **`StatisticalCalibrationOracle`**: Confidence calibration analysis

### 6. Statistical Analysis (`metareason.analysis`)
- **`BayesianAnalyzer`**: PyMC-based Bayesian inference
- **`BetaBinomialModel`**: Success rate modeling with uncertainty
- **`MCMCSampler`**: Posterior sampling with diagnostics
- **`ConfidenceIntervalCalculator`**: HDI and credible intervals

## Key Design Patterns

### 1. **Plugin Architecture**
```python
# Adapter registration and discovery
@register_adapter("custom_provider")
class CustomAdapter(LLMAdapter):
    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        # Custom implementation
```

### 2. **Factory Pattern**
```python
# Flexible object creation based on configuration
sampler = SamplerFactory.create(axes, sampling_config)
adapter = AdapterFactory.create(adapter_config)
```

### 3. **Async-First Design**
```python
# All LLM operations are async for performance
responses = await asyncio.gather(*[
    adapter.complete_async(request) for request in batch
])
```

### 4. **Configuration-Driven Workflow**
```python
# Everything configured via YAML with Pydantic validation
config = load_yaml_config("evaluation.yaml")
# Automatic validation, includes, environment substitution
```

### 5. **Caching Strategy**
```python
# Multi-level caching for performance
config = cache.get(path) or load_and_cache(path)
samples = cache.get(sample_key) or generate_and_cache(axes)
```

## Data Flow Summary

1. **Configuration** → YAML → includes/env vars → validation → `EvaluationConfig`
2. **Sampling** → axes definition → LHS → optimization → `SampleResult`
3. **Templates** → Jinja2 + samples → batch rendering → prompts
4. **LLM Calls** → adapters + rate limiting → async batching → responses
5. **Oracles** → multiple evaluation criteria → scoring → `OracleResult[]`
6. **Analysis** → Bayesian modeling → MCMC → confidence intervals
7. **Output** → structured results → console/JSON/YAML

## Advanced Architecture Insights

### Concurrency & Performance Patterns

**Async Batch Processing**
```python
# Parallel LLM calls with rate limiting
async with adapter.rate_limiter:
    tasks = [adapter.complete_async(req) for req in batch]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
```

**Template Rendering Pipeline**
```python
# Multi-threaded template rendering
renderer = BatchRenderer(engine, batch_size=100, max_workers=4)
render_result = await renderer.render_batch(template, contexts)
```

### Error Handling & Resilience

**Retry with Exponential Backoff**
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
async def _make_request(self, request: CompletionRequest):
    # HTTP request with automatic retry
```

**Graceful Degradation**
```python
# Fallback mechanisms at multiple levels
try:
    data = process_includes_and_inheritance(path)
except Exception:
    data = standard_yaml_load(path)  # Fallback
```

### Memory & Resource Management

**Streaming for Large Datasets**
```python
async def stream_completions(self, requests: Iterator[CompletionRequest]):
    async for request in requests:
        yield await self.complete_async(request)
```

**Smart Caching with TTL**
```python
@lru_cache(maxsize=256, ttl=3600)  # 1-hour TTL
def cached_embedding(text: str) -> np.ndarray:
    return compute_embedding(text)
```

### Configuration Composition Patterns

**Include System with Inheritance**
```yaml
# base_config.yaml
base_sampling: &default_sampling
  method: latin_hypercube
  optimization_criterion: maximin

# child_config.yaml
sampling:
  <<: *default_sampling
  n_samples: 1000
```

**Environment Variable Integration**
```yaml
adapters:
  primary:
    api_key: ${OPENAI_API_KEY}
    base_url: ${API_BASE_URL:-https://api.openai.com}
```

### Quality Assurance Patterns

**Template Validation Pipeline**
```python
validator = TemplateValidator(level=ValidationLevel.STRICT)
result = validator.validate(template, expected_variables=axes.keys())
if not result.is_valid:
    raise TemplateValidationError(result.errors)
```

**Statistical Quality Metrics**
```python
# Sample quality assessment
metrics = lhs_sampler.compute_quality_metrics()
if metrics.space_filling_efficiency < 0.8:
    logger.warning("Poor space-filling detected")
```

### Extensibility Hooks

**Oracle Plugin System**
```python
@register_oracle("custom_metric")
class CustomOracle(BaseOracle):
    def evaluate(self, responses: List[str]) -> OracleResult:
        # Custom evaluation logic
```

**Custom Filter Registration**
```python
@register_filter("format_scientific")
def scientific_notation(value: float) -> str:
    return f"{value:.2e}"
```

## Key Architectural Strengths

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Components accept interfaces, enabling easy testing and swapping
3. **Configuration-Driven**: Behavior controlled through declarative YAML, not code changes
4. **Async-First**: Built for high-throughput LLM operations from the ground up
5. **Statistical Rigor**: Proper uncertainty quantification with Bayesian methods
6. **Plugin Architecture**: Extensible without modifying core codebase
7. **Comprehensive Error Handling**: Graceful failure modes with detailed diagnostics
8. **Performance Optimized**: Multi-level caching, batching, and parallel processing

This architecture enables reproducible, statistically rigorous LLM evaluation with comprehensive configuration management and extensible plugin systems.
