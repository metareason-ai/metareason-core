"""Test using the complete example from the YAML specification."""

import pytest

from metareason.config import load_yaml_config, validate_yaml_string


def test_complete_spec_example():
    """Test the complete example from yaml-configuration-schema.md."""
    yaml_content = """
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
    class_name: "ISOComplianceOracle"
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
"""
    
    # Parse and validate the configuration
    config = validate_yaml_string(yaml_content)
    
    # Test main fields
    assert config.prompt_id == "iso_42001_compliance_check"
    assert config.n_variants == 2000
    
    # Test axes
    assert len(config.axes) == 7
    assert config.axes["verb"].type == "categorical"
    assert config.axes["verb"].values == ["analyze", "evaluate", "assess", "review"]
    assert config.axes["verb"].weights == [0.3, 0.3, 0.2, 0.2]
    
    assert config.axes["temperature"].type == "truncated_normal"
    assert config.axes["temperature"].mu == 0.7
    assert config.axes["temperature"].sigma == 0.15
    
    assert config.axes["top_p"].type == "beta"
    assert config.axes["top_p"].alpha == 9.0
    assert config.axes["top_p"].beta == 1.0
    
    # Test sampling
    assert config.sampling.method == "latin_hypercube"
    assert config.sampling.optimization_criterion == "maximin"
    assert config.sampling.random_seed == 42
    assert config.sampling.stratified_by == ["domain", "focus_area"]
    
    # Test oracles
    assert config.oracles.accuracy is not None
    assert config.oracles.accuracy.type == "embedding_similarity"
    assert config.oracles.accuracy.threshold == 0.88
    assert config.oracles.accuracy.method == "cosine_similarity"
    
    assert config.oracles.explainability is not None
    assert config.oracles.explainability.type == "llm_judge"
    assert config.oracles.explainability.temperature == 0.0
    
    # Test custom oracle
    assert config.oracles.custom_oracles is not None
    assert "regulatory_alignment" in config.oracles.custom_oracles
    custom_oracle = config.oracles.custom_oracles["regulatory_alignment"]
    assert custom_oracle.module == "metareason.oracles.compliance"
    assert custom_oracle.class_name == "ISOComplianceOracle"
    
    # Test domain context
    assert config.domain_context is not None
    assert config.domain_context.industry == "financial_services"
    assert config.domain_context.risk_category == "high"
    assert config.domain_context.data_sensitivity == "pii"
    
    # Test metadata
    assert config.metadata is not None
    assert config.metadata.version == "1.0.0"
    assert config.metadata.created_by == "jeff@metareason.ai"
    assert config.metadata.review_cycle == "quarterly"
    assert len(config.metadata.compliance_mappings) == 3
    assert "production" in config.metadata.tags


def test_statistical_config_from_spec():
    """Test the statistical configuration example from the spec."""
    yaml_content = """
prompt_id: test_statistical
prompt_template: "Test {{param}}"
schema:
  param:
    type: categorical
    values: ["a", "b", "c"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Expected answer with sufficient detail for validation"
    threshold: 0.9
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
"""
    
    config = validate_yaml_string(yaml_content)
    
    assert config.statistical_config is not None
    assert config.statistical_config.model == "beta_binomial"
    
    # Test prior
    assert config.statistical_config.prior.alpha == 1.0
    assert config.statistical_config.prior.beta == 1.0
    
    # Test inference
    assert config.statistical_config.inference.method == "mcmc"
    assert config.statistical_config.inference.samples == 4000
    assert config.statistical_config.inference.chains == 4
    assert config.statistical_config.inference.target_accept == 0.8
    
    # Test output
    assert config.statistical_config.output.credible_interval == 0.95
    assert config.statistical_config.output.hdi_method == "shortest"