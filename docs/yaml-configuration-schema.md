# MetaReason YAML Schema Specification v1.0

## Overview

The MetaReason YAML Schema defines a declarative format for specifying Large Language Model (LLM) evaluation configurations. Each YAML file represents a single "prompt family" that generates multiple variants for statistical confidence scoring.

## Design Principles

1. **Version Control Native**: One file per prompt family, designed for Git versioning
2. **Statistical Rigor**: Support for both categorical and continuous parameter distributions
3. **Reproducibility**: All randomness is controlled and seedable
4. **Extensibility**: Schema supports custom axes types and oracle definitions
5. **Industry Alignment**: Built-in support for compliance and regulatory mappings

## Schema Structure

### Top-Level Fields

| Field             | Type    | Required | Description                                                              |
| ----------------- | ------- | -------- | ------------------------------------------------------------------------ |
| `prompt_id`       | string  | Yes      | Unique identifier for the prompt family. Use lowercase with underscores. |
| `prompt_template` | string  | Yes      | Jinja2-compatible template with variable placeholders.                   |
| `schema`          | object  | Yes      | Defines all variable axes for prompt generation.                         |
| `sampling`        | object  | No       | Configuration for sampling strategy (defaults to Latin Hypercube).       |
| `n_variants`      | integer | No       | Number of prompt variants to generate (default: 1000).                   |
| `oracles`         | object  | Yes      | Defines evaluation criteria and scoring methods.                         |
| `domain_context`  | object  | No       | Industry-specific context and compliance mappings.                       |
| `metadata`        | object  | No       | Tracking information for governance and audit trails.                    |

### Schema Object

The `schema` object defines all variable axes for prompt generation. Each axis has a name (key) and configuration (value).

#### Categorical Axes

```yaml
axis_name:
  type: categorical
  values: ["option1", "option2", "option3"]
  weights: [0.5, 0.3, 0.2]  # Optional: probability weights (must sum to 1.0)
```

#### Continuous Axes

##### Truncated Normal Distribution
```yaml
axis_name:
  type: truncated_normal
  mu: 0.7         # Mean
  sigma: 0.1      # Standard deviation
  min: 0.0        # Minimum value (truncation)
  max: 1.0        # Maximum value (truncation)
```

##### Uniform Distribution
```yaml
axis_name:
  type: uniform
  min: 0.0
  max: 1.0
```

##### Beta Distribution
```yaml
axis_name:
  type: beta
  alpha: 2.0
  beta: 5.0
```

### Sampling Object

```yaml
sampling:
  method: "latin_hypercube"  # Options: latin_hypercube, random, sobol
  optimization_criterion: "maximin"  # For LHS: maximin, correlation, esi
  random_seed: 42  # For reproducibility
  stratified_by: ["persona_clause"]  # Optional: ensure balanced categorical sampling
```

### Oracles Object

Oracles define the evaluation criteria. Multiple oracle types can be specified:

#### Accuracy Oracle
```yaml
oracles:
  accuracy:
    type: "embedding_similarity"
    canonical_answer: |
      The expected correct answer to the prompt.
    method: "cosine_similarity"  # Options: cosine_similarity, semantic_entropy, euclidean
    threshold: 0.90
    embedding_model: "text-embedding-3-small"  # Optional: specify embedding model
```

#### LLM-as-Judge Oracle

The LLM-as-judge oracle uses a language model to evaluate responses according to a specific rubric. This is particularly useful for subjective evaluations like explainability, helpfulness, or domain-specific quality criteria.

```yaml
oracles:
  explainability:
    type: "llm_judge"
    rubric: |
      Rate the response on the following criteria:
      1. Directly answers the question asked
      2. Provides clear, step-by-step reasoning
      3. Uses appropriate domain terminology
      4. Avoids unnecessary complexity or jargon
      5. Addresses potential edge cases or limitations

      Score each criterion as Yes (1) or No (0), then provide an overall assessment.
    judge_model: "gpt-4"  # Options: gpt-4, gpt-3.5-turbo, claude-3-opus, llama3, etc.
    temperature: 0.0  # Use 0.0 for consistent judging, higher for creative evaluation
    output_format: "structured"  # Options: binary, score, structured
```

**Configuration Options:**

- `rubric` (required): Detailed evaluation criteria. Should include:
  - Specific, measurable criteria
  - Clear scoring instructions
  - Examples of good/bad responses (optional)
  - Domain-specific requirements

- `judge_model` (default: "gpt-4"): LLM model used as judge. Consider:
  - **gpt-4**: Best for complex, nuanced evaluation
  - **gpt-3.5-turbo**: Cost-effective for simpler criteria
  - **claude-3-opus**: Alternative high-quality judge
  - **llama3**: For local/private deployments

- `temperature` (default: 0.0): Controls judge consistency
  - **0.0**: Maximum consistency, recommended for most use cases
  - **0.1-0.3**: Slight variation while maintaining reliability
  - **Higher values**: Use cautiously, may reduce judgment reliability

- `output_format`: Response format expected from judge
  - **binary**: Simple pass/fail (1/0)
  - **score**: Numerical score (0-10 scale)
  - **structured**: Detailed JSON with reasoning and scores

**Output Format Examples:**

*Binary Format:*
```json
{
  "score": 1,
  "reasoning": "Response directly answers the question with clear reasoning."
}
```

*Score Format:*
```json
{
  "score": 8,
  "reasoning": "Strong response with minor clarity issues in technical explanation."
}
```

*Structured Format:*
```json
{
  "score": 0.8,
  "reasoning": "Comprehensive answer with good reasoning but could improve domain terminology.",
  "criteria_scores": {
    "directly_answers": 1,
    "clear_reasoning": 1,
    "domain_terminology": 0,
    "appropriate_complexity": 1,
    "addresses_limitations": 1
  }
}
```

**Best Practices:**

1. **Rubric Design**: Make criteria specific and measurable
2. **Judge Selection**: Use high-quality models for complex evaluations
3. **Temperature**: Keep low (0.0-0.1) for consistency
4. **Bias Mitigation**: Consider multiple judges for critical evaluations
5. **Cost Management**: Balance judge quality with evaluation budget

#### Confidence Calibration Oracle
```yaml
oracles:
  confidence_calibration:
    type: "statistical_calibration"
    expected_confidence: 0.85
    tolerance: 0.10
    calibration_method: "platt_scaling"  # Options: platt_scaling, isotonic_regression
```

#### Custom Oracle
```yaml
oracles:
  custom_metric:
    type: "custom"
    module: "metareason.oracles.finance"
    class: "RegulatoryComplianceOracle"
    config:
      frameworks: ["SOX", "GDPR", "MiFID II"]
```

### Domain Context Object

```yaml
domain_context:
  industry: "financial_services"  # Options: financial_services, healthcare, insurance, general
  regulatory_frameworks: ["ISO 42001", "EU AI Act", "SR 11-7"]
  risk_category: "high"  # Per EU AI Act: minimal, limited, high, unacceptable
  use_case: "credit_decisioning"
  data_sensitivity: "pii"  # Options: public, internal, confidential, pii
```

### Metadata Object

```yaml
metadata:
  version: "1.0.0"  # Schema version
  created_by: "jeff@metareason.ai"
  created_date: "2024-07-09"
  last_modified: "2024-07-09"
  review_cycle: "quarterly"  # Options: monthly, quarterly, annual
  compliance_mappings:
    - "SOC2-CC6.1"
    - "ISO27001-A.12.1"
    - "NIST-AI-100-1"
  tags: ["production", "high-risk", "customer-facing"]
  deprecation_date: null  # ISO 8601 date when this evaluation expires
```

## Complete Example

```yaml
# iso_compliance_evaluation.yaml
prompt_id: "iso_42001_compliance_check"
prompt_template: |
  {{verb}} the implications of ISO 42001 on {{domain}} governance{{persona_clause}},
  focusing on {{focus_area}}{{structure_clause}}.

schema:
  # Categorical axes
  verb:
    type: categorical
    values: ["analyze", "evaluate", "assess", "review"]
    weights: [0.3, 0.3, 0.2, 0.2]

  domain:
    type: categorical
    values: ["AI model", "data", "algorithm", "ML pipeline"]

  persona_clause:
    type: categorical
    values:
      - " from a compliance officer perspective"
      - " from a technical implementation view"
      - " for executive stakeholders"
      - ""

  focus_area:
    type: categorical
    values: ["risk assessment", "documentation requirements", "audit procedures"]

  structure_clause:
    type: categorical
    values:
      - ""
      - ". Provide a structured list"
      - ". Include specific examples"

  # Continuous axes
  temperature:
    type: truncated_normal
    mu: 0.7
    sigma: 0.15
    min: 0.3
    max: 0.95

  top_p:
    type: beta
    alpha: 9.0
    beta: 1.0  # Skewed towards higher values

sampling:
  method: "latin_hypercube"
  optimization_criterion: "maximin"
  random_seed: 42
  stratified_by: ["domain", "focus_area"]

n_variants: 2000

oracles:
  accuracy:
    type: "embedding_similarity"
    canonical_answer: |
      ISO 42001 requires organizations to establish, implement, maintain and continually
      improve an AI management system. Key requirements include: documented AI policy,
      risk assessment procedures, data governance controls, algorithm transparency measures,
      performance monitoring, and regular audits. Organizations must demonstrate accountability
      through clear roles, responsibilities, and decision-making processes for AI systems.
    method: "cosine_similarity"
    threshold: 0.88
    embedding_model: "text-embedding-3-small"

  explainability:
    type: "llm_judge"
    rubric: |
      Score 1 if the response meets ALL criteria:
      1. Directly addresses ISO 42001 implications
      2. Provides actionable guidance for the specified domain
      3. Appropriate for the target persona
      4. Clear structure without excessive detail
      5. Includes at least one specific requirement or example
    judge_model: "gpt-4"
    temperature: 0.0
    output_format: "binary"

  regulatory_alignment:
    type: "custom"
    module: "metareason.oracles.compliance"
    class: "ISOComplianceOracle"
    config:
      standard: "ISO42001:2023"
      check_clauses: ["5.1", "6.1", "7.3", "8.1", "9.1"]

domain_context:
  industry: "financial_services"
  regulatory_frameworks: ["ISO 42001", "EU AI Act", "Basel III"]
  risk_category: "high"
  use_case: "credit_risk_assessment"
  data_sensitivity: "pii"

metadata:
  version: "1.0.0"
  created_by: "jeff@metareason.ai"
  created_date: "2024-07-09"
  last_modified: "2024-07-09"
  review_cycle: "quarterly"
  compliance_mappings:
    - "ISO42001-5.1"
    - "EU-AI-Act-Article-9"
    - "SR-11-7-Model-Risk"
  tags: ["production", "high-risk", "credit-decisioning"]
  deprecation_date: "2025-07-09"
```

## Statistical Configuration

When processing evaluations, MetaReason applies the following statistical model:

```yaml
statistical_config:
  model: "beta_binomial"
  prior:
    alpha: 1.0  # Uniform prior default
    beta: 1.0
  inference:
    method: "mcmc"
    samples: 4000
    chains: 4
    target_accept: 0.8
  output:
    credible_interval: 0.95
    hdi_method: "shortest"  # Options: shortest, central
```

## Best Practices

1. **Prompt ID Naming**: Use descriptive, lowercase identifiers with underscores (e.g., `iso_42001_risk_assessment`)

2. **Template Variables**: All variables in `prompt_template` must have corresponding entries in `schema`

3. **Categorical Balance**: For categorical axes, aim for 3-7 options to maintain statistical power

4. **Continuous Distributions**: Choose distributions that match the parameter's natural behavior:
   - Temperature/top_p: Beta or truncated normal
   - Lengths/counts: Poisson or negative binomial
   - Proportions: Beta

5. **Oracle Design**:
   - Accuracy oracles should have clear, comprehensive canonical answers
   - Explainability rubrics should be specific and measurable
   - Use multiple oracles to capture different quality dimensions

6. **Version Control**:
   - Commit YAML files with meaningful messages
   - Tag releases that correspond to production evaluations
   - Use branches for experimental prompt variations

## Validation Rules

1. **Required Fields**: `prompt_id`, `prompt_template`, `schema`, and `oracles` must be present

2. **Template Variables**: All `{{variables}}` in prompt_template must exist in schema

3. **Probability Weights**: If specified, categorical weights must sum to 1.0 (±0.001)

4. **Distribution Parameters**:
   - Normal distributions: sigma > 0
   - Beta distributions: alpha, beta > 0
   - Truncated distributions: min < max

5. **Oracle Thresholds**: Must be between 0.0 and 1.0

6. **Sampling Constraints**:
   - n_variants ≥ 100 (for statistical significance)
   - n_variants ≤ 10000 (for computational efficiency)

## Future Extensions (v2.0)

- **Conditional Dependencies**: Axes that depend on other axis values
- **Multi-Modal Inputs**: Support for image/document references
- **Hierarchical Prompts**: Parent-child prompt relationships
- **A/B Test Integration**: Built-in experiment tracking
- **Cost Optimization**: Token usage estimates and limits
