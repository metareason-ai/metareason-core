"""Configuration validation utilities for MetaReason."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import ValidationError

from .loader import load_yaml_config
from .models import EvaluationConfig


class ValidationReport:
    """Report containing validation results and suggestions."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.is_valid = True

    def add_error(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add an error to the report."""
        self.errors.append(
            {"field": field, "message": message, "suggestion": suggestion}
        )
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning to the report."""
        self.warnings.append(message)

    def add_suggestion(self, message: str):
        """Add a suggestion to the report."""
        self.suggestions.append(message)

    def to_string(self) -> str:
        """Convert report to a formatted string."""
        lines = []

        if self.file_path:
            lines.append(f"Validation Report for: {self.file_path}")
            lines.append("=" * 50)

        if self.is_valid:
            lines.append("✅ Configuration is valid!")
        else:
            lines.append("❌ Configuration has errors")

        if self.errors:
            lines.append("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. Field: {error['field']}")
                lines.append(f"     Error: {error['message']}")
                if error["suggestion"]:
                    lines.append(f"     💡 Suggestion: {error['suggestion']}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        if self.suggestions:
            lines.append("\nSuggestions for improvement:")
            for suggestion in self.suggestions:
                lines.append(f"  💡 {suggestion}")

        return "\n".join(lines)


def validate_yaml_file(
    file_path: Union[str, Path], strict: bool = False
) -> Tuple[Optional[EvaluationConfig], ValidationReport]:
    """Validate a YAML configuration file with detailed reporting.

    Args:
        file_path: Path to YAML configuration file
        strict: If True, warnings are treated as errors

    Returns:
        Tuple of (config object if valid, validation report)
    """
    path = Path(file_path)
    report = ValidationReport(path)

    # Check file exists
    if not path.exists():
        report.add_error(
            "file",
            f"File not found: {path}",
            "Check the file path and ensure the file exists",
        )
        return None, report

    # Try to load YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        report.add_error(
            "yaml",
            f"Invalid YAML syntax: {e}",
            "Use a YAML validator to check syntax errors",
        )
        return None, report

    # Try to create config object
    try:
        config = load_yaml_config(path)
    except ValidationError as e:
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            report.add_error(
                field_path, error["msg"], error.get("ctx", {}).get("suggestion")
            )
        return None, report
    except Exception as e:
        report.add_error("general", str(e), None)
        return None, report

    # Add warnings and suggestions
    _add_warnings_and_suggestions(config, data, report, strict)

    return config if report.is_valid else None, report


def _add_warnings_and_suggestions(
    config: EvaluationConfig,
    raw_data: Dict[str, Any],
    report: ValidationReport,
    strict: bool,
):
    """Add warnings and suggestions based on best practices."""

    # Check for unused axes
    template_vars = set()
    import re

    for match in re.finditer(r"\{\{(\w+)\}\}", config.prompt_template):
        template_vars.add(match.group(1))

    unused_axes = set(config.axes.keys()) - template_vars
    if unused_axes:
        msg = f"Unused axes defined: {sorted(unused_axes)}"
        if strict:
            report.add_error(
                "axes", msg, "Remove unused axes or use them in prompt_template"
            )
        else:
            report.add_warning(msg)

    # Check categorical axis complexity
    for axis_name, axis_config in config.axes.items():
        if hasattr(axis_config, "values"):  # Categorical
            if len(axis_config.values) > 10:
                msg = f"Axis '{axis_name}' has {len(axis_config.values)} values (>10)"
                report.add_warning(
                    f"{msg}. Consider reducing for better statistical power."
                )
            elif len(axis_config.values) < 3:
                msg = f"Axis '{axis_name}' has only {len(axis_config.values)} values"
                report.add_suggestion(
                    f"{msg}. Consider adding more values for meaningful variation."
                )

    # Check sample size recommendations
    categorical_combinations = 1
    for axis_config in config.axes.values():
        if hasattr(axis_config, "values"):
            categorical_combinations *= len(axis_config.values)

    if categorical_combinations > 1:
        recommended_samples = categorical_combinations * 10
        if config.n_variants < recommended_samples:
            report.add_suggestion(
                f"With {categorical_combinations} categorical combinations, "
                f"consider using at least {recommended_samples} variants "
                f"(currently {config.n_variants})"
            )

    # Check for statistical config
    if not config.statistical_config:
        report.add_suggestion(
            "Consider adding statistical_config for explicit control over "
            "inference parameters"
        )

    # Check oracle configuration
    oracle_count = sum(
        [
            1 if config.oracles.accuracy else 0,
            1 if config.oracles.explainability else 0,
            1 if config.oracles.confidence_calibration else 0,
            len(config.oracles.custom_oracles) if config.oracles.custom_oracles else 0,
        ]
    )

    if oracle_count == 1:
        report.add_suggestion(
            "Consider using multiple oracles to capture different quality dimensions"
        )

    # Check metadata
    if not config.metadata:
        report.add_suggestion(
            "Consider adding metadata for better governance and tracking"
        )
    elif config.metadata:
        if not config.metadata.created_by:
            report.add_suggestion("Add created_by to metadata for accountability")
        if not config.metadata.review_cycle:
            report.add_suggestion(
                "Add review_cycle to metadata for maintenance planning"
            )


def validate_yaml_directory(
    directory: Union[str, Path], strict: bool = False
) -> Dict[str, ValidationReport]:
    """Validate all YAML files in a directory.

    Args:
        directory: Path to directory containing YAML files
        strict: If True, warnings are treated as errors

    Returns:
        Dictionary mapping file names to validation reports
    """
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory: {dir_path}")

    yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(f"No YAML files found in {dir_path}")

    reports = {}

    for file_path in yaml_files:
        _, report = validate_yaml_file(file_path, strict)
        reports[file_path.name] = report

    return reports
