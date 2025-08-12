"""Schema migration utilities for test configurations.

This module provides utilities for migrating test configurations between
different schema versions, supporting the evolution from the current format
to the new pipeline-based format.
"""

from copy import deepcopy
from typing import Any, Dict, List


def migrate_config_v1_to_v2(config_v1: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate configuration from v1 (current) to v2 (pipeline) format.

    Args:
        config_v1: Configuration in v1 format with prompt_id and prompt_template

    Returns:
        Configuration in v2 format with test_id and pipeline

    Example:
        v1 = {
            "prompt_id": "test_analysis",
            "prompt_template": "Analyze {{topic}} with {{method}}",
            "primary_model": {...},
            "axes": {...},
            "oracles": {...}
        }

        v2 = migrate_config_v1_to_v2(v1)
        # Results in:
        # {
        #     "test_id": "test_analysis",
        #     "pipeline": [
        #         {
        #             "type": "template",
        #             "template": "Analyze {{topic}} with {{method}}"
        #         }
        #     ],
        #     "primary_model": {...},
        #     "axes": {...},
        #     "oracles": {...}
        # }
    """
    config_v2 = deepcopy(config_v1)

    # 1. Rename prompt_id to test_id
    if "prompt_id" in config_v2:
        config_v2["test_id"] = config_v2.pop("prompt_id")

    # 2. Move prompt_template into pipeline structure
    if "prompt_template" in config_v2:
        template = config_v2.pop("prompt_template")
        config_v2["pipeline"] = [{"type": "template", "template": template}]

    return config_v2


def migrate_config_v2_to_v1(config_v2: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate configuration from v2 (pipeline) to v1 (current) format.

    Args:
        config_v2: Configuration in v2 format with test_id and pipeline

    Returns:
        Configuration in v1 format with prompt_id and prompt_template

    This is useful for backward compatibility testing.
    """
    config_v1 = deepcopy(config_v2)

    # 1. Rename test_id to prompt_id
    if "test_id" in config_v1:
        config_v1["prompt_id"] = config_v1.pop("test_id")

    # 2. Extract prompt_template from pipeline structure
    if "pipeline" in config_v1:
        pipeline = config_v1.pop("pipeline")
        # Find the first template step
        for step in pipeline:
            if step.get("type") == "template" and "template" in step:
                config_v1["prompt_template"] = step["template"]
                break
        else:
            # No template step found, create a default
            config_v1["prompt_template"] = "Default template"

    return config_v1


def extract_template_variables(template: str) -> List[str]:
    """Extract variable names from a Jinja2 template string.

    Args:
        template: Jinja2 template string like "Hello {{name}} from {{location}}"

    Returns:
        List of variable names like ["name", "location"]
    """
    import re

    # Find all {{variable}} patterns
    pattern = r"\{\{\s*(\w+)\s*\}\}"
    variables = re.findall(pattern, template)
    return list(set(variables))  # Remove duplicates


def validate_axes_match_template(config: Dict[str, Any]) -> List[str]:
    """Validate that all template variables have corresponding axes.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Extract template from either v1 or v2 format
    template = None
    if "prompt_template" in config:
        template = config["prompt_template"]
    elif "pipeline" in config:
        for step in config["pipeline"]:
            if step.get("type") == "template" and "template" in step:
                template = step["template"]
                break

    if not template:
        errors.append("No template found in configuration")
        return errors

    # Get template variables
    template_vars = extract_template_variables(template)

    # Get axes
    axes = config.get("axes", {})
    axis_names = set(axes.keys())

    # Check for missing axes
    missing_axes = set(template_vars) - axis_names
    if missing_axes:
        errors.append(
            f"Template variables without axes: {', '.join(sorted(missing_axes))}"
        )

    # Check for unused axes (warning, not error)
    unused_axes = axis_names - set(template_vars)
    if unused_axes:
        errors.append(f"Axes not used in template: {', '.join(sorted(unused_axes))}")

    return errors


def create_migration_test_cases() -> Dict[str, Dict[str, Any]]:
    """Create test cases for schema migration testing.

    Returns:
        Dictionary of test case name -> config data
    """
    test_cases = {
        "minimal_v1": {
            "prompt_id": "minimal_test",
            "prompt_template": "Test {{param}}",
            "primary_model": {"adapter": "openai", "model": "gpt-3.5-turbo"},
            "axes": {"param": {"type": "categorical", "values": ["A", "B"]}},
            "oracles": {
                "accuracy": {
                    "type": "embedding_similarity",
                    "canonical_answer": "Test answer",
                    "threshold": 0.8,
                }
            },
        },
        "comprehensive_v1": {
            "prompt_id": "comprehensive_test",
            "prompt_template": "Analyze {{topic}} using {{method}} approach",
            "primary_model": {
                "adapter": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "axes": {
                "topic": {
                    "type": "categorical",
                    "values": ["AI", "ML", "DL"],
                    "weights": [0.4, 0.4, 0.2],
                },
                "method": {"type": "categorical", "values": ["technical", "business"]},
            },
            "n_variants": 1000,
            "sampling": {
                "method": "latin_hypercube",
                "optimization_criterion": "maximin",
                "random_seed": 42,
            },
            "oracles": {
                "accuracy": {
                    "type": "embedding_similarity",
                    "canonical_answer": "Comprehensive analysis covering key concepts",
                    "threshold": 0.85,
                },
                "quality": {
                    "type": "llm_judge",
                    "rubric": "Rate quality from 1-5",
                    "judge_model": "gpt-4",
                },
            },
        },
    }

    return test_cases


def assert_migration_equivalence(
    config_v1: Dict[str, Any], config_v2: Dict[str, Any]
) -> None:
    """Assert that v1 and v2 configurations are equivalent after migration.

    Args:
        config_v1: Original v1 configuration
        config_v2: Migrated v2 configuration

    Raises:
        AssertionError: If configurations are not equivalent
    """
    # Check ID mapping
    assert config_v2.get("test_id") == config_v1.get(
        "prompt_id"
    ), f"test_id mismatch: {config_v2.get('test_id')} != {config_v1.get('prompt_id')}"

    # Check template mapping
    if "prompt_template" in config_v1:
        assert "pipeline" in config_v2, "Pipeline missing in v2 config"
        template_steps = [
            step for step in config_v2["pipeline"] if step.get("type") == "template"
        ]
        assert len(template_steps) > 0, "No template step found in pipeline"
        assert (
            template_steps[0]["template"] == config_v1["prompt_template"]
        ), f"Template mismatch: {template_steps[0]['template']} != {config_v1['prompt_template']}"

    # Check other fields are preserved
    for key in [
        "primary_model",
        "axes",
        "oracles",
        "n_variants",
        "sampling",
        "statistical_config",
        "metadata",
    ]:
        if key in config_v1:
            assert (
                config_v2.get(key) == config_v1[key]
            ), f"Field {key} not preserved during migration"
