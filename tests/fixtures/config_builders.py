"""Configuration builders for test data generation.

This module provides fluent builder classes for creating test configurations
without hardcoded YAML strings, making tests more maintainable and reducing
duplication across the test suite.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

from metareason.config.models import EvaluationConfig


class AxisBuilder:
    """Builder for creating axis configurations."""

    def __init__(self, name: str):
        self.name = name
        self._config: Dict[str, Any] = {}

    def categorical(
        self, values: List[str], weights: Optional[List[float]] = None
    ) -> "AxisBuilder":
        """Create a categorical axis."""
        self._config = {"type": "categorical", "values": values}
        if weights:
            self._config["weights"] = weights
        return self

    def uniform(self, min_val: float, max_val: float) -> "AxisBuilder":
        """Create a uniform distribution axis."""
        self._config = {"type": "uniform", "min": min_val, "max": max_val}
        return self

    def truncated_normal(
        self, mu: float, sigma: float, min_val: float, max_val: float
    ) -> "AxisBuilder":
        """Create a truncated normal distribution axis."""
        self._config = {
            "type": "truncated_normal",
            "mu": mu,
            "sigma": sigma,
            "min": min_val,
            "max": max_val,
        }
        return self

    def beta(self, alpha: float, beta: float) -> "AxisBuilder":
        """Create a beta distribution axis."""
        self._config = {"type": "beta", "alpha": alpha, "beta": beta}
        return self

    def build(self) -> Dict[str, Any]:
        """Build the axis configuration."""
        return deepcopy(self._config)


class OracleBuilder:
    """Builder for creating oracle configurations."""

    def __init__(self, name: str):
        self.name = name
        self._config: Dict[str, Any] = {}

    def embedding_similarity(
        self,
        canonical_answer: str,
        threshold: float = 0.8,
        method: str = "cosine_similarity",
        embedding_model: str = "text-embedding-3-small",
    ) -> "OracleBuilder":
        """Create an embedding similarity oracle."""
        self._config = {
            "type": "embedding_similarity",
            "canonical_answer": canonical_answer,
            "threshold": threshold,
            "method": method,
            "embedding_model": embedding_model,
        }
        return self

    def llm_judge(
        self,
        rubric: str,
        judge_model: str = "gpt-4",
        temperature: float = 0.0,
        output_format: str = "binary",
    ) -> "OracleBuilder":
        """Create an LLM judge oracle."""
        self._config = {
            "type": "llm_judge",
            "rubric": rubric,
            "judge_model": judge_model,
            "temperature": temperature,
            "output_format": output_format,
        }
        return self

    def quality_assurance(
        self, checks: List[str], severity: str = "error"
    ) -> "OracleBuilder":
        """Create a quality assurance oracle."""
        self._config = {
            "type": "quality_assurance",
            "checks": checks,
            "severity": severity,
        }
        return self

    def custom(self, custom_config: Dict[str, Any]) -> "OracleBuilder":
        """Create a custom oracle."""
        self._config = {"type": "custom", **custom_config}
        return self

    def build(self) -> Dict[str, Any]:
        """Build the oracle configuration."""
        return deepcopy(self._config)


class PrimaryModelBuilder:
    """Builder for primary model configurations."""

    def __init__(self):
        self._config = {"adapter": "openai", "model": "gpt-3.5-turbo"}

    def openai(self, model: str = "gpt-3.5-turbo") -> "PrimaryModelBuilder":
        """Configure OpenAI adapter."""
        self._config.update({"adapter": "openai", "model": model})
        return self

    def anthropic(
        self, model: str = "claude-3-sonnet-20240229"
    ) -> "PrimaryModelBuilder":
        """Configure Anthropic adapter."""
        self._config.update({"adapter": "anthropic", "model": model})
        return self

    def google(self, model: str = "gemini-2.0-flash-001") -> "PrimaryModelBuilder":
        """Configure Google adapter."""
        self._config.update({"adapter": "google", "model": model})
        return self

    def ollama(self, model: str = "llama3") -> "PrimaryModelBuilder":
        """Configure Ollama adapter."""
        self._config.update({"adapter": "ollama", "model": model})
        return self

    def with_temperature(self, temperature: float) -> "PrimaryModelBuilder":
        """Set temperature parameter."""
        self._config["temperature"] = temperature
        return self

    def with_max_tokens(self, max_tokens: int) -> "PrimaryModelBuilder":
        """Set max_tokens parameter."""
        self._config["max_tokens"] = max_tokens
        return self

    def with_json_schema(self, schema_path: str) -> "PrimaryModelBuilder":
        """Set JSON schema path."""
        self._config["json_schema"] = schema_path
        return self

    def with_params(self, **params) -> "PrimaryModelBuilder":
        """Set additional parameters."""
        self._config.update(params)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the primary model configuration."""
        return deepcopy(self._config)


class ConfigBuilder:
    """Main builder for creating evaluation configurations.

    This class provides a fluent API for building test configurations
    without hardcoded YAML strings, making tests more maintainable.

    Example:
        config = (ConfigBuilder()
            .test_id("my_test")
            .prompt_template("Hello {{name}}")
            .with_axis("name", lambda a: a.categorical(["Alice", "Bob"]))
            .with_oracle("accuracy", lambda o: o.embedding_similarity("Test answer"))
            .build())
    """

    def __init__(self):
        self._config: Dict[str, Any] = {
            "prompt_id": "test_config",
            "prompt_template": "Hello {{name}}, this is a test template",
            "primary_model": PrimaryModelBuilder().build(),
            "axes": {},
            "oracles": {},
        }

    def test_id(self, test_id: str) -> "ConfigBuilder":
        """Set the test ID (currently prompt_id)."""
        self._config["prompt_id"] = test_id
        return self

    def prompt_template(self, template: str) -> "ConfigBuilder":
        """Set the prompt template."""
        self._config["prompt_template"] = template
        return self

    def primary_model(self, builder_func) -> "ConfigBuilder":
        """Configure primary model using a builder function."""
        builder = PrimaryModelBuilder()
        self._config["primary_model"] = builder_func(builder).build()
        return self

    def with_axis(self, name: str, builder_func) -> "ConfigBuilder":
        """Add an axis using a builder function."""
        builder = AxisBuilder(name)
        self._config["axes"][name] = builder_func(builder).build()
        return self

    def with_axes(self, **axes_configs) -> "ConfigBuilder":
        """Add multiple axes with simple configurations."""
        for name, config in axes_configs.items():
            if isinstance(config, dict):
                self._config["axes"][name] = config
            else:
                # Assume it's a list for categorical values
                self._config["axes"][name] = {"type": "categorical", "values": config}
        return self

    def with_oracle(self, name: str, builder_func) -> "ConfigBuilder":
        """Add an oracle using a builder function."""
        builder = OracleBuilder(name)
        self._config["oracles"][name] = builder_func(builder).build()
        return self

    def with_oracles(self, **oracle_configs) -> "ConfigBuilder":
        """Add multiple oracles with simple configurations."""
        for name, config in oracle_configs.items():
            self._config["oracles"][name] = config
        return self

    def with_variants(self, n_variants: int) -> "ConfigBuilder":
        """Set number of variants."""
        self._config["n_variants"] = n_variants
        return self

    def with_sampling(self, **sampling_config) -> "ConfigBuilder":
        """Configure sampling parameters."""
        self._config["sampling"] = sampling_config
        return self

    def with_statistical_config(self, **stats_config) -> "ConfigBuilder":
        """Configure statistical analysis parameters."""
        self._config["statistical_config"] = stats_config
        return self

    def with_metadata(self, **metadata) -> "ConfigBuilder":
        """Add metadata to the configuration."""
        self._config["metadata"] = metadata
        return self

    def with_params(self, **params) -> "ConfigBuilder":
        """Add arbitrary parameters to the configuration."""
        self._config.update(params)
        return self

    def minimal(self) -> "ConfigBuilder":
        """Create a minimal valid configuration."""
        return (
            self.test_id("minimal_test")
            .prompt_template("Test {{param}}")
            .with_axis("param", lambda a: a.categorical(["value1", "value2"]))
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "This is a comprehensive test answer for validation"
                ),
            )
        )

    def comprehensive(self) -> "ConfigBuilder":
        """Create a comprehensive configuration for complex testing."""
        return (
            self.test_id("comprehensive_test")
            .prompt_template("Analyze {{topic}} with {{approach}} methodology")
            .primary_model(
                lambda p: p.openai("gpt-4").with_temperature(0.7).with_max_tokens(1000)
            )
            .with_axis(
                "topic", lambda a: a.categorical(["AI", "ML", "DL"], [0.4, 0.4, 0.2])
            )
            .with_axis(
                "approach",
                lambda a: a.categorical(["technical", "business", "academic"]),
            )
            .with_axis("temperature", lambda a: a.truncated_normal(0.7, 0.1, 0.3, 0.9))
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "A comprehensive analysis covering key concepts and practical applications",
                    threshold=0.85,
                ),
            )
            .with_oracle(
                "explainability",
                lambda o: o.llm_judge(
                    (
                        "Rate response quality from 1-5 based on: "
                        "1. Clarity and coherence of explanation, "
                        "2. Technical accuracy of content, "
                        "3. Completeness of analysis"
                    )
                ),
            )
            .with_variants(1000)
            .with_sampling(
                method="latin_hypercube",
                optimization_criterion="maximin",
                random_seed=42,
            )
        )

    def legacy_format(self) -> "ConfigBuilder":
        """Create configuration in legacy format for migration testing."""
        # This will be useful when we transition to pipeline format
        return self.minimal()

    def invalid(self, error_type: str = "missing_field") -> "ConfigBuilder":
        """Create an invalid configuration for error testing."""
        if error_type == "missing_field":
            # Remove required field
            if "oracles" in self._config:
                del self._config["oracles"]
        elif error_type == "empty_prompt_id":
            self._config["prompt_id"] = ""
        elif error_type == "invalid_axis_type":
            self._config["axes"]["invalid"] = {"type": "invalid_type"}
        elif error_type == "missing_primary_model":
            if "primary_model" in self._config:
                del self._config["primary_model"]
        return self

    def build(self) -> EvaluationConfig:
        """Build the final EvaluationConfig object."""
        return EvaluationConfig(**deepcopy(self._config))

    def build_dict(self) -> Dict[str, Any]:
        """Build the configuration as a dictionary."""
        return deepcopy(self._config)

    def to_yaml(self, **yaml_kwargs) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.build_dict(), default_flow_style=False, **yaml_kwargs)


class YamlTemplate:
    """Template strings for common YAML configurations."""

    MINIMAL = """
prompt_id: {prompt_id}
prompt_template: "{prompt_template}"
primary_model:
  adapter: {adapter}
  model: {model}
axes:
  {axis_name}:
    type: categorical
    values: {axis_values}
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "{canonical_answer}"
    threshold: {threshold}
"""

    WITH_VARIANTS = """
prompt_id: {prompt_id}
prompt_template: "{prompt_template}"
primary_model:
  adapter: {adapter}
  model: {model}
n_variants: {n_variants}
axes:
  {axis_name}:
    type: categorical
    values: {axis_values}
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "{canonical_answer}"
    threshold: {threshold}
"""

    COMPREHENSIVE = """
prompt_id: {prompt_id}
prompt_template: "{prompt_template}"
primary_model:
  adapter: {adapter}
  model: {model}
  temperature: {temperature}
  max_tokens: {max_tokens}
axes:
  param1:
    type: categorical
    values: {param1_values}
  param2:
    type: categorical
    values: {param2_values}
n_variants: {n_variants}
sampling:
  method: latin_hypercube
  optimization_criterion: maximin
  random_seed: 42
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "{canonical_answer}"
    threshold: {threshold}
  quality:
    type: llm_judge
    rubric: "{rubric}"
    judge_model: gpt-4
    temperature: 0.0
    output_format: binary
"""

    @classmethod
    def render(cls, template: str, **kwargs) -> str:
        """Render a template with provided parameters."""
        defaults = {
            "prompt_id": "test_config",
            "prompt_template": "Hello {{name}}, this is a test template",
            "adapter": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "axis_name": "name",
            "axis_values": '["Alice", "Bob"]',
            "param1_values": '["A", "B"]',
            "param2_values": '["X", "Y"]',
            "n_variants": 1000,
            "canonical_answer": "This is a comprehensive test answer for validation",
            "threshold": 0.8,
            "rubric": "Rate the response quality from 1-5",
        }
        defaults.update(kwargs)
        return template.format(**defaults)
