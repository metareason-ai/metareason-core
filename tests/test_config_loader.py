"""Tests for YAML configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from metareason.config import (
    EvaluationConfig,
    load_yaml_config,
    load_yaml_configs,
    validate_yaml_file,
    validate_yaml_string,
)


def test_load_valid_yaml_config(tmp_path):
    """Test loading a valid YAML configuration."""
    yaml_content = """
prompt_id: test_evaluation
prompt_template: "Analyze {{topic}} with {{style}}"
schema:
  topic:
    type: categorical
    values: ["AI", "ML", "DL"]
  style:
    type: categorical
    values: ["formal", "casual"]
sampling:
  method: latin_hypercube
  random_seed: 42
n_variants: 1000
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is the expected answer"
    threshold: 0.85
"""

    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = load_yaml_config(yaml_file)

    assert config.prompt_id == "test_evaluation"
    assert config.n_variants == 1000
    assert "topic" in config.axes
    assert "style" in config.axes
    assert config.oracles.accuracy.threshold == 0.85


def test_load_yaml_with_statistical_config(tmp_path):
    """Test loading YAML with statistical configuration."""
    yaml_content = """
prompt_id: test_with_stats
prompt_template: "Test {{param}}"
schema:
  param:
    type: uniform
    min: 0.0
    max: 1.0
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Expected answer with sufficient detail for validation"
    threshold: 0.9
statistical_config:
  model: beta_binomial
  prior:
    alpha: 2.0
    beta: 2.0
  inference:
    method: mcmc
    samples: 4000
    chains: 4
"""

    yaml_file = tmp_path / "test_stats.yaml"
    yaml_file.write_text(yaml_content)

    config = load_yaml_config(yaml_file)

    assert config.statistical_config is not None
    assert config.statistical_config.model == "beta_binomial"
    assert config.statistical_config.prior.alpha == 2.0
    assert config.statistical_config.inference.samples == 4000


def test_validate_yaml_string():
    """Test validating YAML from a string."""
    yaml_content = """
prompt_id: string_test
prompt_template: "Do {{action}}"
schema:
  action:
    type: categorical
    values: ["analyze", "evaluate"]
oracles:
  explainability:
    type: llm_judge
    rubric: |
      1. Clear
      2. Concise
      3. Correct
"""

    config = validate_yaml_string(yaml_content)

    assert config.prompt_id == "string_test"
    assert config.oracles.explainability.rubric.strip().startswith("1. Clear")


def test_load_yaml_configs_directory(tmp_path):
    """Test loading multiple YAML configs from a directory."""
    # Create multiple YAML files
    configs_data = {
        "config1.yaml": """
prompt_id: eval_1
prompt_template: "Test {{x}}"
schema:
  x:
    type: categorical
    values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Expected answer with sufficient detail"
    threshold: 0.8
""",
        "config2.yml": """
prompt_id: eval_2
prompt_template: "Check {{y}}"
schema:
  y:
    type: beta
    alpha: 2.0
    beta: 5.0
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Result with sufficient detail for evaluation"
    threshold: 0.9
""",
    }

    for filename, content in configs_data.items():
        (tmp_path / filename).write_text(content)

    configs = load_yaml_configs(tmp_path)

    assert len(configs) == 2
    assert "config1" in configs
    assert "config2" in configs
    assert configs["config1"].prompt_id == "eval_1"
    assert configs["config2"].prompt_id == "eval_2"


def test_validation_report(tmp_path):
    """Test validation report generation."""
    yaml_content = """
prompt_id: test_validation
prompt_template: "Analyze {{topic}}"
schema:
  topic:
    type: categorical
    values: ["AI", "ML"]
  unused_axis:
    type: categorical
    values: ["x", "y", "z"]
n_variants: 100
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Expected answer"
    threshold: 0.85
"""

    yaml_file = tmp_path / "test_validation.yaml"
    yaml_file.write_text(yaml_content)

    config, report = validate_yaml_file(yaml_file)

    assert config is not None
    assert report.is_valid
    assert len(report.warnings) > 0  # Should warn about unused axis
    assert any("unused_axis" in w for w in report.warnings)


def test_invalid_yaml_syntax(tmp_path):
    """Test handling of invalid YAML syntax."""
    yaml_content = """
prompt_id: invalid
prompt_template: "Test"
schema:
  - invalid list instead of dict
"""

    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(yaml_content)

    config, report = validate_yaml_file(yaml_file)

    assert config is None
    assert not report.is_valid
    assert len(report.errors) > 0


def test_missing_required_fields(tmp_path):
    """Test validation of missing required fields."""
    yaml_content = """
prompt_template: "Test {{x}}"
schema:
  x:
    type: categorical
    values: ["a", "b"]
"""
    # Missing prompt_id and oracles

    yaml_file = tmp_path / "missing_fields.yaml"
    yaml_file.write_text(yaml_content)

    with pytest.raises(Exception):  # Should raise validation error
        load_yaml_config(yaml_file)


def test_all_distribution_types(tmp_path):
    """Test all supported distribution types."""
    yaml_content = """
prompt_id: all_distributions
prompt_template: "Test {{cat}} {{truncnorm}} {{beta}} {{uniform}}"
schema:
  cat:
    type: categorical
    values: ["a", "b", "c"]
    weights: [0.5, 0.3, 0.2]
  truncnorm:
    type: truncated_normal
    mu: 0.5
    sigma: 0.1
    min: 0.0
    max: 1.0
  beta:
    type: beta
    alpha: 2.0
    beta: 5.0
  uniform:
    type: uniform
    min: -1.0
    max: 1.0
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a test canonical answer for validation"
    threshold: 0.8
"""

    yaml_file = tmp_path / "all_dist.yaml"
    yaml_file.write_text(yaml_content)

    config = load_yaml_config(yaml_file)

    assert len(config.axes) == 4
    assert config.axes["cat"].type == "categorical"
    assert config.axes["truncnorm"].type == "truncated_normal"
    assert config.axes["beta"].type == "beta"
    assert config.axes["uniform"].type == "uniform"


def test_file_not_found():
    """Test handling of non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("nonexistent.yaml")


def test_invalid_file_extension(tmp_path):
    """Test handling of invalid file extension."""
    txt_file = tmp_path / "config.txt"
    txt_file.write_text("not yaml")

    with pytest.raises(ValueError, match="Invalid file extension"):
        load_yaml_config(txt_file)
