"""Shared pytest fixtures and configuration."""

from typing import Any

from pytest import fixture


@fixture  # type: ignore[misc]
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "sampling_strategies": ["temperature", "top_p"],
    }
