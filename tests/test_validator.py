"""Tests for the validation utilities."""

from pathlib import Path

import pytest

from metareason.config.validator import (
    ValidationReport,
    validate_yaml_directory,
    validate_yaml_file,
)


class TestValidationReport:
    """Test ValidationReport class."""

    def test_empty_report_is_valid(self):
        """Test that empty report is valid."""
        report = ValidationReport()
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0
        assert len(report.suggestions) == 0

    def test_add_error_makes_invalid(self):
        """Test that adding error makes report invalid."""
        report = ValidationReport()
        report.add_error("test_field", "test error", "test suggestion")

        assert not report.is_valid
        assert len(report.errors) == 1
        assert report.errors[0]["field"] == "test_field"
        assert report.errors[0]["message"] == "test error"
        assert report.errors[0]["suggestion"] == "test suggestion"

    def test_add_warning_keeps_valid(self):
        """Test that adding warning keeps report valid."""
        report = ValidationReport()
        report.add_warning("test warning")

        assert report.is_valid
        assert len(report.warnings) == 1
        assert report.warnings[0] == "test warning"

    def test_add_suggestion_keeps_valid(self):
        """Test that adding suggestion keeps report valid."""
        report = ValidationReport()
        report.add_suggestion("test suggestion")

        assert report.is_valid
        assert len(report.suggestions) == 1
        assert report.suggestions[0] == "test suggestion"

    def test_to_string_valid_report(self):
        """Test string representation of valid report."""
        report = ValidationReport(Path("test.yaml"))
        report.add_warning("test warning")
        report.add_suggestion("test suggestion")

        result = report.to_string()

        assert "test.yaml" in result
        assert "✅ Configuration is valid!" in result
        assert "⚠️  test warning" in result
        assert "💡 test suggestion" in result

    def test_to_string_invalid_report(self):
        """Test string representation of invalid report."""
        report = ValidationReport(Path("test.yaml"))
        report.add_error("field1", "error message", "error suggestion")

        result = report.to_string()

        assert "test.yaml" in result
        assert "❌ Configuration has errors" in result
        assert "1. Field: field1" in result
        assert "Error: error message" in result
        assert "💡 Suggestion: error suggestion" in result

    def test_to_string_no_file_path(self):
        """Test string representation without file path."""
        report = ValidationReport()
        report.add_error("field1", "error message")

        result = report.to_string()

        assert "Validation Report for:" not in result
        assert "❌ Configuration has errors" in result

    def test_to_string_error_without_suggestion(self):
        """Test string representation of error without suggestion."""
        report = ValidationReport()
        report.add_error("field1", "error message", None)

        result = report.to_string()

        assert "Error: error message" in result
        assert "💡 Suggestion:" not in result


class TestValidateYamlFile:
    """Test validate_yaml_file function."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        config, report = validate_yaml_file("nonexistent.yaml")

        assert config is None
        assert not report.is_valid
        assert len(report.errors) == 1
        assert report.errors[0]["field"] == "file"
        assert "File not found" in report.errors[0]["message"]

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test handling of invalid YAML syntax."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: syntax: [")

        config, report = validate_yaml_file(yaml_file)

        assert config is None
        assert not report.is_valid
        assert len(report.errors) == 1
        assert report.errors[0]["field"] == "yaml"
        assert "Invalid YAML syntax" in report.errors[0]["message"]

    def test_validation_error(self, tmp_path):
        """Test handling of validation errors."""
        yaml_content = """
spec_id: ""
pipeline:
  - template: "test"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      test:
        type: categorical
        values: ["a"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "test answer with sufficient detail"
    threshold: 0.8
"""
        yaml_file = tmp_path / "invalid_config.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is None
        assert not report.is_valid
        assert len(report.errors) >= 1

    def test_valid_configuration(self, tmp_path):
        """Test validation of valid configuration."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b", "c"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "valid_config.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert config.spec_id == "test_config"

    def test_unused_axes_warning(self, tmp_path):
        """Test warning for unused axes."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param1}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param1:
        type: categorical
        values: ["a", "b"]
      unused_param:
        type: categorical
        values: ["x", "y"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_with_unused.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.warnings) >= 1
        assert any("Unused axes defined" in w for w in report.warnings)

    def test_unused_axes_error_in_strict_mode(self, tmp_path):
        """Test error for unused axes in strict mode."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param1}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param1:
        type: categorical
        values: ["a", "b"]
      unused_param:
        type: categorical
        values: ["x", "y"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_with_unused.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file, strict=True)

        assert config is None
        assert not report.is_valid
        assert len(report.errors) >= 1
        assert any("Unused axes defined" in e["message"] for e in report.errors)

    def test_categorical_axis_warnings(self, tmp_path):
        """Test warnings for categorical axis complexity."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{many_values}} and {{few_values}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      many_values:
        type: categorical
        values: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
      few_values:
        type: categorical
        values: ["x"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_axis_warnings.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.warnings) >= 1
        assert len(report.suggestions) >= 1

        # Check for warning about too many values
        assert any("many_values" in w and ">10" in w for w in report.warnings)
        # Check for suggestion about too few values
        assert any(
            "few_values" in s and "only 1 values" in s for s in report.suggestions
        )

    def test_sample_size_recommendation(self, tmp_path):
        """Test sample size recommendations."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param1}} and {{param2}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param1:
        type: categorical
        values: ["a", "b", "c"]
      param2:
        type: categorical
        values: ["x", "y"]
n_variants: 100
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_sample_size.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 1

        # Should suggest more variants (3*2=6 combinations, need 60+ variants)
        # But we have 100 which is good, so no suggestion for this case
        # Let's test with insufficient variants

    def test_insufficient_sample_size_suggestion(self, tmp_path):
        """Test suggestion for insufficient sample size."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param1}} and {{param2}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param1:
        type: categorical
        values: ["a", "b", "c", "d"]
      param2:
        type: categorical
        values: ["x", "y", "z"]
n_variants: 100  # 4*3=12 combinations, need 120+
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_insufficient_samples.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 1
        assert any(
            "categorical combinations" in s and "variants" in s
            for s in report.suggestions
        )

    def test_statistical_config_suggestion(self, tmp_path):
        """Test suggestion to add statistical config."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_no_stats.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 1
        assert any("statistical_config" in s for s in report.suggestions)

    def test_oracle_count_suggestion(self, tmp_path):
        """Test suggestion for multiple oracles."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_single_oracle.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 1
        assert any("multiple oracles" in s for s in report.suggestions)

    def test_metadata_suggestions(self, tmp_path):
        """Test suggestions for metadata."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config_no_metadata.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 1
        assert any("metadata" in s for s in report.suggestions)

    def test_incomplete_metadata_suggestions(self, tmp_path):
        """Test suggestions for incomplete metadata."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
metadata:
  version: "1.0.0"
"""
        yaml_file = tmp_path / "config_incomplete_metadata.yaml"
        yaml_file.write_text(yaml_content)

        config, report = validate_yaml_file(yaml_file)

        assert config is not None
        assert report.is_valid
        assert len(report.suggestions) >= 2
        assert any("created_by" in s for s in report.suggestions)
        assert any("review_cycle" in s for s in report.suggestions)

    def test_general_exception_handling(self, tmp_path, monkeypatch):
        """Test handling of general exceptions."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        # Mock load_yaml_config to raise a general exception
        def mock_load_yaml_config(path):
            raise RuntimeError("Unexpected error")

        monkeypatch.setattr(
            "metareason.config.validator.load_yaml_config", mock_load_yaml_config
        )

        config, report = validate_yaml_file(yaml_file)

        assert config is None
        assert not report.is_valid
        assert len(report.errors) == 1
        assert report.errors[0]["field"] == "general"
        assert "Unexpected error" in report.errors[0]["message"]


class TestValidateYamlDirectory:
    """Test validate_yaml_directory function."""

    def test_directory_not_found(self):
        """Test handling of non-existent directory."""
        with pytest.raises(ValueError, match="Invalid directory"):
            validate_yaml_directory("nonexistent_directory")

    def test_path_is_file_not_directory(self, tmp_path):
        """Test handling when path is a file, not directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Invalid directory"):
            validate_yaml_directory(test_file)

    def test_no_yaml_files_in_directory(self, tmp_path):
        """Test handling when no YAML files found."""
        # Create a directory with no YAML files
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="No YAML files found"):
            validate_yaml_directory(tmp_path)

    def test_validate_single_yaml_file(self, tmp_path):
        """Test validating directory with single YAML file."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        reports = validate_yaml_directory(tmp_path)

        assert len(reports) == 1
        assert "config.yaml" in reports
        assert reports["config.yaml"].is_valid

    def test_validate_multiple_yaml_files(self, tmp_path):
        """Test validating directory with multiple YAML files."""
        valid_yaml = """
spec_id: valid_config
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""

        invalid_yaml = """
spec_id: ""
pipeline:
  - template: "Test"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: []
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "short"
    threshold: 0.8
"""

        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text(valid_yaml)

        invalid_file = tmp_path / "invalid.yml"
        invalid_file.write_text(invalid_yaml)

        reports = validate_yaml_directory(tmp_path)

        assert len(reports) == 2
        assert "valid.yaml" in reports
        assert "invalid.yml" in reports
        assert reports["valid.yaml"].is_valid
        assert not reports["invalid.yml"].is_valid

    def test_validate_directory_with_strict_mode(self, tmp_path):
        """Test validating directory in strict mode."""
        yaml_content = """
spec_id: test_config
pipeline:
  - template: "Test {{param1}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param1:
        type: categorical
        values: ["a", "b"]
      unused_param:
        type: categorical
        values: ["x", "y"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        reports = validate_yaml_directory(tmp_path, strict=True)

        assert len(reports) == 1
        assert "config.yaml" in reports
        assert not reports[
            "config.yaml"
        ].is_valid  # Should be invalid due to unused axes in strict mode
