"""Shared pytest fixtures and configuration."""

import tempfile
from pathlib import Path
from typing import Any

from pytest import fixture

from .factories.evaluation_factory import (
    EvaluationFactory,
    TestDataFactory,
    YamlFileFactory,
)
from .fixtures.config_builders import ConfigBuilder, YamlTemplate


@fixture  # type: ignore[misc]
def sample_config() -> dict[str, Any]:
    """Legacy sample configuration for testing."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "sampling_strategies": ["temperature", "top_p"],
    }


@fixture
def config_builder():
    """Provide a ConfigBuilder instance for test configuration creation."""
    return ConfigBuilder()


@fixture
def yaml_template():
    """Provide YamlTemplate class for template-based configuration creation."""
    return YamlTemplate


@fixture
def evaluation_factory():
    """Provide EvaluationFactory for common configuration patterns."""
    return EvaluationFactory


@fixture
def test_data_factory():
    """Provide TestDataFactory for common test data patterns."""
    return TestDataFactory


@fixture
def minimal_config(config_builder):
    """Provide a minimal valid evaluation configuration."""
    return config_builder.minimal().build()


@fixture
def comprehensive_config(config_builder):
    """Provide a comprehensive evaluation configuration."""
    return config_builder.comprehensive().build()


@fixture
def temp_yaml_file():
    """Create a temporary YAML file and clean it up after test."""
    temp_files = []

    def _create_temp_yaml(content: str, suffix: str = ".yaml") -> Path:
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        temp_path = Path(temp_file.name)
        temp_files.append(temp_path)
        return temp_path

    yield _create_temp_yaml

    # Cleanup
    for temp_path in temp_files:
        if temp_path.exists():
            temp_path.unlink()


@fixture
def temp_config_file(config_builder):
    """Create a temporary configuration file with minimal config."""
    temp_files = []

    def _create_config_file(builder_func=None, **overrides):
        if builder_func:
            builder = builder_func(config_builder)
        else:
            builder = config_builder.minimal()

        if overrides:
            builder = builder.with_params(**overrides)

        temp_path = YamlFileFactory.create_temp_file(builder.build())
        temp_files.append(temp_path)
        return temp_path

    yield _create_config_file

    # Cleanup
    for temp_path in temp_files:
        if temp_path.exists():
            temp_path.unlink()


@fixture
def common_axes():
    """Provide common axis configurations for testing."""
    return TestDataFactory.COMMON_AXES


@fixture
def common_oracles():
    """Provide common oracle configurations for testing."""
    return TestDataFactory.COMMON_ORACLES
