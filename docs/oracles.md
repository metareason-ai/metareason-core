# MetaReason Oracle System Documentation

## Overview

The MetaReason oracle system provides multiple evaluation methods for assessing LLM responses. Oracles are pluggable evaluation components that score responses based on different criteria like accuracy, explainability, and confidence calibration.

## Oracle Types

### 1. LLM-as-Judge Oracle

The LLM-as-Judge oracle uses a language model to evaluate responses according to custom rubrics. This is particularly valuable for subjective evaluations that require domain expertise or complex reasoning.

#### Key Features

- **Flexible Rubrics**: Define custom evaluation criteria in natural language
- **Multiple Output Formats**: Binary, score, or structured evaluations
- **Judge Rotation**: Support for multiple judge models to reduce bias
- **Chain-of-Thought**: Automatic reasoning enhancement for complex evaluations
- **Robust Parsing**: Handles malformed judge responses with fallback mechanisms
- **Few-Shot Learning**: Built-in examples for consistency

#### Configuration

```yaml
oracles:
  explainability:
    type: "llm_judge"
    rubric: |
      Evaluate the response on these criteria:
      1. Directly answers the question asked
      2. Provides clear, logical reasoning
      3. Uses appropriate terminology
      4. Addresses edge cases or limitations

      Score 1 if ALL criteria are met, 0 otherwise.
    judge_model: "gpt-4"
    temperature: 0.0
    output_format: "binary"
```

#### Output Formats

**Binary Format**: Simple pass/fail evaluation
- Score: 0 (fail) or 1 (pass)
- Use for: Clear-cut criteria, compliance checks

**Score Format**: Continuous scoring
- Score: Float between 0.0 and 1.0
- Use for: Quality assessments, ranking responses

**Structured Format**: Detailed evaluation
- Overall score plus dimension-specific scores
- Use for: Complex evaluations, detailed feedback

#### Implementation Details

The LLM judge oracle (`src/metareason/oracles/llm_judge.py`) provides:

1. **Robust JSON Parsing**: Extracts structured responses from LLM output
2. **Fallback Mechanisms**: Handles malformed responses gracefully
3. **Context Management**: Includes original prompts and evaluation context
4. **Error Handling**: Comprehensive error reporting and recovery

#### Best Practices

1. **Rubric Design**:
   - Use numbered criteria for clarity
   - Include specific, measurable requirements
   - Provide examples of good/bad responses
   - Keep criteria independent and non-overlapping

2. **Judge Selection**:
   - **GPT-4**: Best for complex, nuanced evaluations
   - **GPT-3.5-turbo**: Cost-effective for simple binary decisions
   - **Claude-3-opus**: Alternative high-quality judge
   - **Local models**: For privacy-sensitive evaluations

3. **Temperature Settings**:
   - **0.0**: Maximum consistency (recommended)
   - **0.1-0.3**: Slight variation while maintaining reliability
   - **Higher**: Use cautiously, may reduce consistency

4. **Bias Mitigation**:
   - Use multiple judges for critical evaluations
   - Rotate judge models to reduce systematic bias
   - Include diverse examples in rubrics
   - Monitor judge agreement statistics

#### Example Usage

```python
from metareason.oracles.llm_judge import LLMJudgeOracle
from metareason.config.oracles import LLMJudgeConfig

# Configure LLM judge
config = LLMJudgeConfig(
    rubric="Evaluate response quality on accuracy and clarity",
    judge_model="gpt-4",
    temperature=0.0,
    output_format="score"
)

# Initialize oracle
oracle = LLMJudgeOracle(config, adapter_config)
await oracle.initialize()

# Evaluate response
result = await oracle.evaluate(
    response="The capital of France is Paris.",
    context={"original_prompt": "What is the capital of France?"}
)

print(f"Score: {result.score}")
print(f"Reasoning: {result.metadata['reasoning']}")
```

### 2. Embedding Similarity Oracle

The embedding similarity oracle evaluates response accuracy using semantic similarity between responses and canonical answers. It converts text into high-dimensional vector representations (embeddings) and computes similarity scores using various mathematical methods.

#### Key Features

- **Multiple Similarity Metrics**: Four different similarity calculation methods
- **High Performance**: Vectorized calculations with NumPy optimization
- **Batch Processing**: Efficient handling of multiple responses
- **Parallel Execution**: Configurable worker threads for performance
- **Distribution Analysis**: Optional detailed similarity insights
- **Flexible Scoring**: Binary threshold or continuous confidence scores
- **Adapter Integration**: Works with any configured LLM provider for embeddings

#### Configuration Options

```yaml
oracles:
  accuracy:
    type: "embedding_similarity"
    canonical_answer: |
      Comprehensive canonical answer covering all expected response aspects.
      Should be detailed and include key concepts for accurate evaluation.
    method: "cosine_similarity"          # Required: similarity calculation method
    threshold: 0.85                      # Required: similarity threshold (0.0-1.0)
    embedding_model: "text-embedding-3-small"  # Optional: embedding model
    embedding_adapter: "openai"          # Optional: LLM adapter for embeddings
    batch_size: 32                       # Optional: batch processing size (1-1000)
    use_vectorized: true                 # Optional: enable NumPy optimization
    parallel_workers: 4                  # Optional: parallel processing threads
    confidence_passthrough: false       # Optional: raw scores vs binary threshold
    distribution_analysis: false        # Optional: detailed similarity analysis
```

#### Similarity Methods

**1. Cosine Similarity** (default)
- Measures angle between normalized vectors
- Range: -1 to 1 (typically 0-1 for positive embeddings)
- Best for: General semantic similarity, most common choice
- Formula: `cos(θ) = (A·B) / (||A|| × ||B||)`

**2. Euclidean Distance**
- Measures geometric distance between vectors
- Converted to similarity: `1 - (distance / max_distance)`
- Range: 0 to 1
- Best for: When magnitude differences matter

**3. Dot Product**
- Simple vector multiplication sum
- Range: Unbounded (depends on vector magnitudes)
- Best for: When both similarity and magnitude are important

**4. Semantic Entropy**
- Complex similarity using KL divergence approximation
- Range: 0 to 1
- Best for: Capturing distributional semantic differences
- Most computationally expensive but potentially most nuanced

#### Performance Configuration

**Batch Processing:**
```yaml
batch_size: 64              # Process 64 embeddings at once
parallel_workers: 8         # Use 8 threads for similarity calculations
use_vectorized: true        # Enable NumPy optimization (recommended)
```

**Memory vs Speed Trade-offs:**
- **Small batches (1-16)**: Lower memory, good for constrained environments
- **Medium batches (32-64)**: Balanced performance and memory
- **Large batches (100+)**: Higher throughput, requires more memory

**Parallel Processing:**
- **None**: Sequential processing (default)
- **2-4 workers**: Good for most use cases
- **8+ workers**: High-performance scenarios with many embeddings

#### Advanced Features

**Confidence Pass-through:**
```yaml
confidence_passthrough: true  # Return raw similarity scores (0.0-1.0)
# vs
confidence_passthrough: false # Apply binary threshold (0.0 or 1.0)
```

**Distribution Analysis:**
```yaml
distribution_analysis: true
```
Provides detailed metadata including:
- Component-wise similarity statistics (mean, std, min, max)
- Vector magnitude comparisons
- Confidence estimates based on vector properties
- Diagnostic information for debugging similarity calculations

#### Implementation Details

The embedding similarity oracle (`src/metareason/oracles/embedding_similarity.py`) provides:

1. **Mock Embeddings**: Deterministic embeddings for testing and development
2. **Vectorized Calculations**: NumPy-based optimization with sequential fallback
3. **Dimension Validation**: Automatic checking for embedding compatibility
4. **Error Recovery**: Graceful handling of embedding generation failures
5. **Resource Management**: Proper cleanup of thread pools and adapters

#### Usage Examples

**Basic Accuracy Evaluation:**
```python
from metareason.oracles.embedding_similarity import EmbeddingSimilarityOracle
from metareason.config.oracles import EmbeddingSimilarityConfig

config = EmbeddingSimilarityConfig(
    canonical_answer="Paris is the capital and largest city of France.",
    method="cosine_similarity",
    threshold=0.85
)

oracle = EmbeddingSimilarityOracle(config, primary_adapter=adapter)
await oracle.initialize()

result = await oracle.evaluate(
    response="The capital of France is Paris.",
    context={}
)

print(f"Similarity Score: {result.metadata['similarity_score']:.3f}")
print(f"Pass/Fail: {'PASS' if result.score == 1.0 else 'FAIL'}")
```

**High-Performance Batch Evaluation:**
```python
config = EmbeddingSimilarityConfig(
    canonical_answer="Detailed canonical answer...",
    method="cosine_similarity",
    threshold=0.80,
    batch_size=64,
    parallel_workers=8,
    use_vectorized=True
)

oracle = EmbeddingSimilarityOracle(config, primary_adapter=adapter)
await oracle.initialize()

responses = ["Response 1", "Response 2", "Response 3", ...]
results = await oracle.batch_evaluate(responses, {})

for i, result in enumerate(results):
    print(f"Response {i+1}: {result.score:.3f}")
```

**Detailed Analysis Mode:**
```python
config = EmbeddingSimilarityConfig(
    canonical_answer="Comprehensive answer...",
    method="cosine_similarity",
    threshold=0.85,
    confidence_passthrough=True,      # Get raw scores
    distribution_analysis=True        # Get detailed analysis
)

result = await oracle.evaluate(response, {})

analysis = result.metadata["distribution_analysis"]
print(f"Component mean: {analysis['component_mean']:.3f}")
print(f"Component std: {analysis['component_std']:.3f}")
print(f"Confidence estimate: {analysis['confidence_estimate']:.3f}")
```

#### Best Practices

1. **Canonical Answer Design**:
   - Include all key concepts expected in good responses
   - Use clear, comprehensive language
   - Cover edge cases and alternative phrasings
   - Aim for 2-5 sentences for most use cases

2. **Method Selection**:
   - **Cosine similarity**: Default choice for most applications
   - **Euclidean**: When response length/verbosity matters
   - **Dot product**: When both similarity and emphasis matter
   - **Semantic entropy**: For research or highly nuanced evaluation

3. **Threshold Tuning**:
   - **0.70-0.80**: Lenient, good for diverse valid responses
   - **0.85-0.90**: Standard range for most applications
   - **0.90-0.95**: Strict, for precise factual questions
   - **Above 0.95**: Very strict, mainly for exact matches

4. **Performance Optimization**:
   - Use vectorized calculations when NumPy is available
   - Batch responses for better throughput
   - Use parallel workers for CPU-intensive similarity calculations
   - Consider embedding adapter choice for cost optimization

5. **Local vs Cloud Embeddings**:
   ```yaml
   # Cost-optimized with local embeddings
   embedding_adapter: "ollama"
   embedding_model: "nomic-embed-text"

   # High-quality cloud embeddings
   embedding_adapter: "openai"
   embedding_model: "text-embedding-3-large"
   ```

#### Error Handling

Common scenarios and recovery mechanisms:
- **Embedding generation failures**: Automatic retry and error reporting
- **Dimension mismatches**: Clear error messages with dimension details
- **Adapter unavailability**: Graceful fallback to primary adapter
- **Memory limitations**: Automatic batch size adjustment recommendations

### 3. Statistical Calibration Oracle

Assesses confidence calibration of model responses.

#### Configuration

```yaml
oracles:
  confidence_calibration:
    type: "statistical_calibration"
    expected_confidence: 0.85
    tolerance: 0.10
    calibration_method: "platt_scaling"
```

#### Features

- **Calibration Methods**: Platt scaling, isotonic regression
- **Confidence Assessment**: Measures alignment between stated and actual confidence
- **Statistical Rigor**: Quantitative calibration metrics

### 4. Custom Oracle

Enables domain-specific evaluation implementations.

#### Configuration

```yaml
oracles:
  regulatory_compliance:
    type: "custom"
    module: "metareason.oracles.compliance"
    class_name: "RegulatoryComplianceOracle"
    config:
      frameworks: ["SOX", "GDPR", "MiFID II"]
      risk_threshold: 0.05
```

#### Implementation Pattern

```python
from metareason.oracles.base import BaseOracle, OracleResult

class CustomOracle(BaseOracle):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

    async def evaluate(self, response: str, context: Dict[str, Any]) -> OracleResult:
        # Custom evaluation logic
        score = self._custom_scoring_logic(response, context)

        return OracleResult(
            score=score,
            metadata={"custom_metric": "value"}
        )
```

## Oracle Architecture

### Base Oracle Interface

All oracles implement the `BaseOracle` interface:

```python
class BaseOracle(ABC):
    @abstractmethod
    async def evaluate(self, response: str, context: Dict[str, Any]) -> OracleResult:
        """Evaluate response and return scored result."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return oracle identifier."""
        pass
```

### Oracle Result Structure

```python
class OracleResult:
    score: float  # Primary evaluation score (0.0-1.0)
    metadata: Dict[str, Any]  # Additional evaluation details
    confidence: Optional[float] = None  # Oracle's confidence in score
    dimensions: Optional[Dict[str, float]] = None  # Multi-dimensional scores
```

### Integration with Evaluation Pipeline

Oracles are integrated into the evaluation pipeline through:

1. **Configuration**: YAML-based oracle definitions
2. **Factory Pattern**: Automatic oracle instantiation
3. **Async Execution**: Concurrent evaluation processing
4. **Result Aggregation**: Multi-oracle scoring combination

## Performance Considerations

### LLM Judge Oracle

- **Latency**: Judge evaluation adds ~1-3 seconds per response
- **Cost**: Additional API calls for each evaluation
- **Concurrency**: Supports batch evaluation for efficiency
- **Caching**: Results can be cached for repeated evaluations

### Optimization Strategies

1. **Batch Processing**: Evaluate multiple responses simultaneously
2. **Judge Pooling**: Rotate between multiple judge models
3. **Result Caching**: Cache evaluations for identical responses
4. **Parallel Execution**: Run multiple oracles concurrently

## Error Handling

### Common Error Scenarios

1. **Judge Response Parsing**: Malformed JSON from LLM judges
2. **API Failures**: Network or rate limit issues
3. **Configuration Errors**: Invalid oracle configurations
4. **Context Missing**: Required evaluation context not provided

### Recovery Mechanisms

1. **Fallback Parsing**: Extract scores from unstructured responses
2. **Retry Logic**: Automatic retry with exponential backoff
3. **Default Scores**: Safe fallback scores for failed evaluations
4. **Graceful Degradation**: Continue evaluation with partial oracle results

## Monitoring and Observability

### Metrics Tracked

- Oracle evaluation latency
- Judge agreement statistics
- Parsing success rates
- Error frequencies by type
- Score distributions by oracle

### Logging

Comprehensive logging includes:
- Oracle configuration details
- Evaluation context and results
- Error conditions and recovery actions
- Performance metrics and timing

## Future Extensions

### Planned Features

1. **Multi-Judge Consensus**: Automatic agreement analysis
2. **Active Learning**: Adaptive rubric improvement
3. **Human-in-the-Loop**: Hybrid human-AI evaluation
4. **Federated Oracles**: Distributed evaluation networks
5. **Explainable Scoring**: Detailed score attribution
