"""Factory classes for creating test evaluation configurations.

This module provides factory classes that encapsulate common patterns
for creating test configurations, reducing code duplication and improving
test maintainability.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

from metareason.config.models import EvaluationConfig

from ..fixtures.config_builders import ConfigBuilder


class EvaluationFactory:
    """Factory for creating evaluation configurations with common patterns."""

    @classmethod
    def minimal(cls, **overrides) -> EvaluationConfig:
        """Create a minimal valid evaluation configuration.

        Args:
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Valid minimal configuration
        """
        builder = ConfigBuilder().minimal()
        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def with_single_axis(
        cls, axis_name: str = "param", axis_values: List[str] = None, **overrides
    ) -> EvaluationConfig:
        """Create configuration with single categorical axis.

        Args:
            axis_name: Name of the axis
            axis_values: Values for the categorical axis
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Configuration with single axis
        """
        if axis_values is None:
            axis_values = ["value1", "value2"]

        builder = (
            ConfigBuilder()
            .test_id("single_axis_test")
            .prompt_template(f"Test {{{{{axis_name}}}}}")
            .with_axis(axis_name, lambda a: a.categorical(axis_values))
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "This is a comprehensive test answer for validation"
                ),
            )
        )

        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def with_multiple_axes(
        cls, axes_config: Dict[str, List[str]], **overrides
    ) -> EvaluationConfig:
        """Create configuration with multiple categorical axes.

        Args:
            axes_config: Dictionary of axis_name -> values
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Configuration with multiple axes
        """
        template_vars = " ".join([f"{{{{{name}}}}}" for name in axes_config.keys()])
        builder = (
            ConfigBuilder()
            .test_id("multi_axis_test")
            .prompt_template(f"Test {template_vars}")
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "This is a comprehensive test answer for validation"
                ),
            )
        )

        for axis_name, values in axes_config.items():
            builder = builder.with_axis(axis_name, lambda a, v=values: a.categorical(v))

        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def with_variants(cls, n_variants: int, **overrides) -> EvaluationConfig:
        """Create configuration with specified number of variants.

        Args:
            n_variants: Number of variants to generate
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Configuration with variants
        """
        builder = ConfigBuilder().minimal().with_variants(n_variants)

        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def with_sampling_config(
        cls, sampling_method: str = "latin_hypercube", **sampling_params
    ) -> EvaluationConfig:
        """Create configuration with sampling parameters.

        Args:
            sampling_method: Sampling method to use
            **sampling_params: Additional sampling parameters

        Returns:
            EvaluationConfig: Configuration with sampling
        """
        sampling_config = {"method": sampling_method, **sampling_params}
        return ConfigBuilder().minimal().with_sampling(**sampling_config).build()

    @classmethod
    def with_multiple_oracles(
        cls, oracle_configs: Dict[str, Dict[str, Any]], **overrides
    ) -> EvaluationConfig:
        """Create configuration with multiple oracles.

        Args:
            oracle_configs: Dictionary of oracle_name -> oracle_config
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Configuration with multiple oracles
        """
        builder = (
            ConfigBuilder()
            .test_id("multi_oracle_test")
            .prompt_template("Test {{param}}")
            .with_axis("param", lambda a: a.categorical(["A", "B"]))
            .with_oracles(**oracle_configs)
        )

        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def with_primary_model(
        cls, adapter: str, model: str, **model_params
    ) -> EvaluationConfig:
        """Create configuration with specific primary model.

        Args:
            adapter: Adapter name (openai, anthropic, google, etc.)
            model: Model name
            **model_params: Additional model parameters

        Returns:
            EvaluationConfig: Configuration with specified model
        """

        def configure_model(builder):
            if adapter == "openai":
                return builder.openai(model).with_params(**model_params)
            elif adapter == "anthropic":
                return builder.anthropic(model).with_params(**model_params)
            elif adapter == "google":
                return builder.google(model).with_params(**model_params)
            elif adapter == "ollama":
                return builder.ollama(model).with_params(**model_params)
            else:
                return builder.with_params(adapter=adapter, model=model, **model_params)

        return ConfigBuilder().minimal().primary_model(configure_model).build()

    @classmethod
    def with_json_schema(
        cls, schema_path: str = "schemas/test.json", **overrides
    ) -> EvaluationConfig:
        """Create configuration with JSON schema.

        Args:
            schema_path: Path to JSON schema file
            **overrides: Any configuration values to override

        Returns:
            EvaluationConfig: Configuration with JSON schema
        """
        builder = (
            ConfigBuilder()
            .minimal()
            .primary_model(lambda p: p.with_json_schema(schema_path))
        )

        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build()

    @classmethod
    def invalid_missing_field(cls, missing_field: str) -> Dict[str, Any]:
        """Create invalid configuration missing a required field.

        Args:
            missing_field: Name of field to remove

        Returns:
            Dict: Invalid configuration data (not EvaluationConfig object)
        """
        config_dict = ConfigBuilder().minimal().build_dict()
        if missing_field in config_dict:
            del config_dict[missing_field]
        return config_dict

    @classmethod
    def invalid_empty_field(cls, field_name: str) -> Dict[str, Any]:
        """Create invalid configuration with empty required field.

        Args:
            field_name: Name of field to make empty

        Returns:
            Dict: Invalid configuration data (not EvaluationConfig object)
        """
        config_dict = ConfigBuilder().minimal().build_dict()
        if field_name in config_dict:
            if isinstance(config_dict[field_name], str):
                config_dict[field_name] = ""
            elif isinstance(config_dict[field_name], (list, dict)):
                config_dict[field_name] = type(config_dict[field_name])()
        return config_dict

    @classmethod
    def legacy_format(cls, **overrides) -> Dict[str, Any]:
        """Create configuration in legacy format for migration testing.

        Args:
            **overrides: Any configuration values to override

        Returns:
            Dict: Configuration in legacy format
        """
        builder = ConfigBuilder().legacy_format()
        if overrides:
            builder = builder.with_params(**overrides)
        return builder.build_dict()


class YamlFileFactory:
    """Factory for creating temporary YAML configuration files."""

    @classmethod
    def create_temp_file(
        cls, config: Union[EvaluationConfig, Dict[str, Any]], suffix: str = ".yaml"
    ) -> Path:
        """Create a temporary YAML file with configuration.

        Args:
            config: Configuration object or dictionary
            suffix: File suffix (default: .yaml)

        Returns:
            Path: Path to created temporary file
        """
        if isinstance(config, EvaluationConfig):
            # Use model_dump with exclude_none to avoid null values
            config_dict = config.model_dump(exclude_none=True)
            yaml_content = ConfigBuilder().with_params(**config_dict).to_yaml()
        elif isinstance(config, dict):
            # Filter out None values from dictionary
            config_dict = {k: v for k, v in config.items() if v is not None}
            yaml_content = ConfigBuilder().with_params(**config_dict).to_yaml()
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        temp_file.write(yaml_content)
        temp_file.close()

        return Path(temp_file.name)

    @classmethod
    def create_minimal_file(cls, **overrides) -> Path:
        """Create temporary file with minimal configuration.

        Args:
            **overrides: Any configuration values to override

        Returns:
            Path: Path to created temporary file
        """
        config = EvaluationFactory.minimal(**overrides)
        return cls.create_temp_file(config)

    @classmethod
    def create_invalid_file(cls, error_type: str = "missing_field") -> Path:
        """Create temporary file with invalid configuration.

        Args:
            error_type: Type of error to introduce

        Returns:
            Path: Path to created temporary file
        """
        if error_type == "missing_field":
            config = EvaluationFactory.invalid_missing_field("oracles")
        elif error_type == "empty_prompt_id":
            config = EvaluationFactory.invalid_empty_field("prompt_id")
        elif error_type == "missing_primary_model":
            config = EvaluationFactory.invalid_missing_field("primary_model")
        else:
            raise ValueError(f"Unknown error type: {error_type}")

        return cls.create_temp_file(config)


class TestDataFactory:
    """Factory for creating common test data patterns."""

    COMMON_AXES = {
        "name": ["Alice", "Bob", "Charlie"],
        "topic": ["AI", "ML", "DL"],
        "style": ["technical", "simple", "academic"],
        "approach": ["detailed", "brief", "comprehensive"],
    }

    COMMON_ORACLES = {
        "accuracy": {
            "type": "embedding_similarity",
            "canonical_answer": "This is a comprehensive test answer for validation",
            "threshold": 0.8,
        },
        "explainability": {
            "type": "llm_judge",
            "rubric": (
                "Rate the response quality from 1-5 based on: "
                "1. Clarity and coherence of explanation, "
                "2. Technical accuracy of content, "
                "3. Completeness of analysis"
            ),
            "judge_model": "gpt-4",
            "temperature": 0.0,
            "output_format": "binary",
        },
    }

    @classmethod
    def get_common_axis_config(cls, axis_name: str) -> Dict[str, Any]:
        """Get configuration for a common axis.

        Args:
            axis_name: Name of the axis

        Returns:
            Dict: Axis configuration
        """
        if axis_name not in cls.COMMON_AXES:
            raise ValueError(
                f"Unknown axis: {axis_name}. Available: {list(cls.COMMON_AXES.keys())}"
            )

        return {"type": "categorical", "values": cls.COMMON_AXES[axis_name]}

    @classmethod
    def get_common_oracle_config(cls, oracle_name: str) -> Dict[str, Any]:
        """Get configuration for a common oracle.

        Args:
            oracle_name: Name of the oracle

        Returns:
            Dict: Oracle configuration
        """
        if oracle_name not in cls.COMMON_ORACLES:
            raise ValueError(
                f"Unknown oracle: {oracle_name}. Available: {list(cls.COMMON_ORACLES.keys())}"
            )

        return cls.COMMON_ORACLES[oracle_name].copy()
