"""Tests for CLI configuration commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from metareason.cli.config import config_group


class TestConfigValidateCommand:
    """Test the config validate command."""

    def test_validate_single_file_valid(self):
        """Test validating a single valid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob", "Charlie"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ["validate", str(temp_path)])

            assert result.exit_code == 0
            assert "All configuration files are valid" in result.output
        finally:
            temp_path.unlink()

    def test_validate_single_file_invalid(self):
        """Test validating a single invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: ""  # Invalid empty prompt_id
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ["validate", str(temp_path)])

            assert result.exit_code == 1
            assert "configuration files have issues" in result.output
        finally:
            temp_path.unlink()

    def test_validate_directory(self):
        """Test validating all files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid config
            valid_config = temp_path / "valid.yaml"
            valid_config.write_text(
                """
prompt_id: valid_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )

            # Create invalid config
            invalid_config = temp_path / "invalid.yaml"
            invalid_config.write_text(
                """
prompt_id: ""  # Invalid
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )

            runner = CliRunner()
            result = runner.invoke(config_group, ["validate", "-d", str(temp_path)])

            assert result.exit_code == 1
            assert "1 of 2 configuration files have issues" in result.output

    def test_validate_json_output(self):
        """Test validation with JSON output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["validate", str(temp_path), "--format", "json"]
            )

            assert result.exit_code == 0

            # Should be valid JSON
            output_data = json.loads(result.output)
            assert str(temp_path) in output_data
            assert output_data[str(temp_path)]["valid"] is True
        finally:
            temp_path.unlink()

    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
  unused_axis:  # This will trigger warning in normal mode, error in strict
    type: categorical
    values: ["X", "Y"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()

            # Normal mode - should pass with warnings
            result = runner.invoke(config_group, ["validate", str(temp_path)])
            assert result.exit_code == 0

            # Strict mode - should fail
            result = runner.invoke(
                config_group, ["validate", str(temp_path), "--strict"]
            )
            assert result.exit_code == 1
        finally:
            temp_path.unlink()

    def test_validate_no_files_specified(self):
        """Test validation with no files specified."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["validate"])

        assert result.exit_code == 1
        assert "No configuration files specified" in result.output

    def test_validate_nonexistent_directory(self):
        """Test validation with non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(
            config_group, ["validate", "-d", "/nonexistent/directory"]
        )

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_validate_fatal_error_handling(self):
        """Test validation with fatal error during file processing."""
        # Mock validate_yaml_file to raise an unexpected exception
        with patch(
            "metareason.cli.config.validate_yaml_file",
            side_effect=Exception("Unexpected error"),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(
                    """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
                )
                temp_path = Path(f.name)

            try:
                runner = CliRunner()
                result = runner.invoke(config_group, ["validate", str(temp_path)])

                assert result.exit_code == 1
                assert "Fatal error validating" in result.output
            finally:
                temp_path.unlink()

    def test_validate_json_output_with_fatal_error(self):
        """Test JSON output format with fatal validation error."""
        # Mock validate_yaml_file to raise an unexpected exception
        with patch(
            "metareason.cli.config.validate_yaml_file",
            side_effect=Exception("Unexpected error"),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(
                    """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
                )
                temp_path = Path(f.name)

            try:
                runner = CliRunner()
                result = runner.invoke(
                    config_group, ["validate", str(temp_path), "--format", "json"]
                )

                assert result.exit_code == 1
                # Output should contain JSON at the end, but may have other text before it
                lines = result.output.strip().split("\n")
                json_start = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        json_start = i
                        break

                json_output = "\n".join(lines[json_start:])
                output_data = json.loads(json_output)
                assert str(temp_path) in output_data
                assert output_data[str(temp_path)]["valid"] is False
                assert output_data[str(temp_path)]["fatal_error"] is True
            finally:
                temp_path.unlink()

    def test_validate_junit_output(self):
        """Test validation with JUnit XML output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: ""  # Invalid empty prompt_id
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["validate", str(temp_path), "--format", "junit"]
            )

            assert result.exit_code == 1
            # Should contain XML elements
            assert "<testsuites>" in result.output
            assert "<testsuite" in result.output
            assert "<testcase" in result.output
            assert "<failure" in result.output
        finally:
            temp_path.unlink()

    def test_validate_junit_output_with_fatal_error(self):
        """Test JUnit XML output with fatal error."""
        # Mock validate_yaml_file to raise an unexpected exception
        with patch(
            "metareason.cli.config.validate_yaml_file",
            side_effect=Exception("Unexpected error"),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(
                    """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
                )
                temp_path = Path(f.name)

            try:
                runner = CliRunner()
                result = runner.invoke(
                    config_group, ["validate", str(temp_path), "--format", "junit"]
                )

                assert result.exit_code == 1
                # Should contain XML with error element
                assert "<testsuites>" in result.output
                assert "<testsuite" in result.output
                assert "<testcase" in result.output
                assert "<error" in result.output
            finally:
                temp_path.unlink()

    def test_validate_output_to_file(self):
        """Test validation with output written to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as output_f:
            output_path = Path(output_f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                [
                    "validate",
                    str(temp_path),
                    "--format",
                    "json",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert "Results written to" in result.output

            # Check that output file was created and contains valid JSON
            assert output_path.exists()
            output_data = json.loads(output_path.read_text())
            assert str(temp_path) in output_data
        finally:
            temp_path.unlink()
            if output_path.exists():
                output_path.unlink()


class TestConfigShowCommand:
    """Test the config show command."""

    def test_show_basic_yaml(self):
        """Test showing configuration in YAML format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ["show", str(temp_path)])

            assert result.exit_code == 0
            assert "test_config" in result.output
            assert temp_path.name in result.output  # Should show filename
        finally:
            temp_path.unlink()

    def test_show_json_format(self):
        """Test showing configuration in JSON format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["show", str(temp_path), "--format", "json"]
            )

            assert result.exit_code == 0

            # Should contain JSON-like structure
            assert '"prompt_id"' in result.output or "prompt_id" in result.output
        finally:
            temp_path.unlink()

    def test_show_with_includes(self):
        """Test showing configuration with includes expanded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create included file
            included_file = temp_path / "shared.yaml"
            included_file.write_text(
                """
shared_value: "from included file"
"""
            )

            # Create main file with include
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                f"""
prompt_id: test_with_includes
prompt_template: "Hello {{{{name}}}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
shared: !include {included_file.name}
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )

            runner = CliRunner()
            result = runner.invoke(
                config_group, ["show", str(main_file), "--expand-includes"]
            )

            assert result.exit_code == 0
            assert "test_with_includes" in result.output

    def test_show_with_env_expansion(self):
        """Test showing configuration with environment variables expanded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: ${TEST_PROMPT_ID:default_id}
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["${USER1:Alice}", "${USER2:Bob}"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            with patch.dict(
                "os.environ", {"TEST_PROMPT_ID": "env_test", "USER1": "Charlie"}
            ):
                runner = CliRunner()
                result = runner.invoke(
                    config_group, ["show", str(temp_path), "--expand-env"]
                )

                assert result.exit_code == 0
                # Should show expanded values
                assert "env_test" in result.output or "default_id" in result.output
        finally:
            temp_path.unlink()

    def test_show_nonexistent_file(self):
        """Test showing non-existent configuration file."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["show", "/nonexistent/file.yaml"])

        assert result.exit_code == 1
        assert "Error loading configuration" in result.output

    def test_show_toml_format_without_toml_package(self):
        """Test showing configuration in TOML format without toml package installed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            # Mock the import to fail specifically for toml
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if name == "toml":
                    raise ImportError("No module named 'toml'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                runner = CliRunner()
                result = runner.invoke(
                    config_group, ["show", str(temp_path), "--format", "toml"]
                )

                assert result.exit_code == 1
                assert "TOML support requires 'toml' package" in result.output
                assert "pip install toml" in result.output
        finally:
            temp_path.unlink()

    def test_show_toml_format_with_toml_package(self):
        """Test showing configuration in TOML format with toml package available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        # Create a mock toml module
        import sys
        import types

        # Create mock toml module
        mock_toml_module = types.ModuleType("toml")
        mock_toml_module.dumps = lambda data: "[prompt]\nid = 'test_config'\n"

        try:
            # Temporarily add the mock module to sys.modules
            sys.modules["toml"] = mock_toml_module

            runner = CliRunner()
            result = runner.invoke(
                config_group, ["show", str(temp_path), "--format", "toml"]
            )

            assert result.exit_code == 0
            # Should display the TOML formatted content
            assert temp_path.name in result.output
        finally:
            # Clean up: remove mock module
            if "toml" in sys.modules:
                del sys.modules["toml"]
            temp_path.unlink()

    def test_show_output_to_file(self):
        """Test showing configuration with output written to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as output_f:
            output_path = Path(output_f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                [
                    "show",
                    str(temp_path),
                    "--format",
                    "yaml",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert "Configuration written to" in result.output

            # Check that output file was created
            assert output_path.exists()
            content = output_path.read_text()
            assert "test_config" in content
        finally:
            temp_path.unlink()
            if output_path.exists():
                output_path.unlink()


class TestConfigDiffCommand:
    """Test the config diff command."""

    def test_diff_identical_files(self):
        """Test diffing identical configuration files."""
        config_content = """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(config_content)
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(config_content)
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["diff", str(temp_path1), str(temp_path2)]
            )

            assert result.exit_code == 0
            assert "No differences found" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_different_files(self):
        """Test diffing different configuration files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
n_variants: 1000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(
                """
prompt_id: config2
prompt_template: "Hi {{name}}"
n_variants: 2000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob", "Charlie"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.9
"""
            )
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["diff", str(temp_path1), str(temp_path2)]
            )

            assert result.exit_code == 1  # Differences found
            assert "Modified:" in result.output or "changes" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_json_format(self):
        """Test diff with JSON output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(
                """
prompt_id: config2
prompt_template: "Hello {{name}}"
n_variants: 2000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
"""
            )
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                ["diff", str(temp_path1), str(temp_path2), "--format", "json"],
            )

            assert result.exit_code == 1

            # Should be valid JSON
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_ignore_fields(self):
        """Test diff with ignored fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: same_config
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
metadata:
  created_date: "2024-01-01"
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(
                """
prompt_id: same_config
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
metadata:
  created_date: "2024-01-02"  # Different date
"""
            )
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                [
                    "diff",
                    str(temp_path1),
                    str(temp_path2),
                    "--ignore-fields",
                    "metadata.created_date",
                ],
            )

            assert result.exit_code == 0  # No differences after ignoring field
            assert "No differences found" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_error_loading_configurations(self):
        """Test diff command with error loading configurations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            # Create malformed YAML that will cause an error
            f2.write("invalid: yaml: content: [unclosed")
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group, ["diff", str(temp_path1), str(temp_path2)]
            )

            assert result.exit_code == 1
            assert "Error comparing configurations" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_error_loading_configurations_json_format(self):
        """Test diff command error handling with JSON output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            # Create malformed YAML that will cause an error
            f2.write("invalid: yaml: content: [unclosed")
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                ["diff", str(temp_path1), str(temp_path2), "--format", "json"],
            )

            assert result.exit_code == 1
            # Should be valid JSON with error information
            output_data = json.loads(result.output)
            assert output_data["error"] == "Error comparing configurations"
            assert output_data["success"] is False
            assert "message" in output_data
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_unified_format(self):
        """Test diff command with unified output format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
n_variants: 1000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(
                """
prompt_id: config2
prompt_template: "Hi {{name}}"
n_variants: 2000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob", "Charlie"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.9
"""
            )
            temp_path2 = Path(f2.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                ["diff", str(temp_path1), str(temp_path2), "--format", "unified"],
            )

            assert result.exit_code == 1  # Differences found
            # Should contain unified diff format markers
            assert "---" in result.output
            assert "+++" in result.output
            assert "+" in result.output or "-" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_diff_output_to_file(self):
        """Test diff command with output written to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            f1.write(
                """
prompt_id: config1
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""
            )
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            f2.write(
                """
prompt_id: config2
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.9
"""
            )
            temp_path2 = Path(f2.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as output_f:
            output_path = Path(output_f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                config_group,
                [
                    "diff",
                    str(temp_path1),
                    str(temp_path2),
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 1  # Differences found
            assert "Diff results written to" in result.output

            # Check that output file was created
            assert output_path.exists()
            content = output_path.read_text()
            assert "Modified:" in content or "changes" in content
        finally:
            temp_path1.unlink()
            temp_path2.unlink()
            if output_path.exists():
                output_path.unlink()


class TestConfigCacheCommand:
    """Test the config cache command."""

    def test_cache_stats(self):
        """Test showing cache statistics."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["cache", "--stats"])

        assert result.exit_code == 0
        assert "caching" in result.output or "Cache" in result.output

    def test_cache_clear(self):
        """Test clearing the cache."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["cache", "--clear"])

        assert result.exit_code == 0
        assert "caching" in result.output or "Cache" in result.output

    def test_cache_disable(self):
        """Test disabling the cache."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["cache", "--disable"])

        assert result.exit_code == 0
        assert "disabled" in result.output

    def test_cache_default_info(self):
        """Test default cache command shows basic info."""
        runner = CliRunner()
        result = runner.invoke(config_group, ["cache"])

        assert result.exit_code == 0
        # Should show some cache information
        assert (
            "caching" in result.output
            or "Cache" in result.output
            or "entries" in result.output
        )

    def test_cache_when_disabled(self):
        """Test cache commands when caching is disabled."""
        with patch("metareason.config.cache.is_caching_enabled", return_value=False):
            runner = CliRunner()

            # Test stats when caching is disabled
            result = runner.invoke(config_group, ["cache", "--stats"])
            assert result.exit_code == 0
            assert "caching is currently disabled" in result.output

            # Test clear when caching is disabled
            result = runner.invoke(config_group, ["cache", "--clear"])
            assert result.exit_code == 0
            assert "caching is currently disabled" in result.output

            # Test default command when caching is disabled
            result = runner.invoke(config_group, ["cache"])
            assert result.exit_code == 0
            assert "caching is currently disabled" in result.output

    def test_cache_clear_and_stats_when_enabled(self):
        """Test cache clear and stats when caching is enabled."""
        mock_cache = patch("metareason.config.cache.get_global_cache")
        mock_enabled = patch(
            "metareason.config.cache.is_caching_enabled", return_value=True
        )

        # Create a mock cache instance with stats
        from unittest.mock import MagicMock

        mock_cache_instance = MagicMock()
        mock_cache_instance.get_stats.return_value = {
            "total_entries": 10,
            "active_entries": 8,
            "expired_entries": 2,
            "ttl_seconds": 300,
            "hot_reload_enabled": True,
        }

        with mock_cache as mock_get_cache, mock_enabled:
            mock_get_cache.return_value = mock_cache_instance

            runner = CliRunner()

            # Test clear
            result = runner.invoke(config_group, ["cache", "--clear"])
            assert result.exit_code == 0
            assert "cache cleared" in result.output
            mock_cache_instance.clear.assert_called_once()

            # Test stats
            result = runner.invoke(config_group, ["cache", "--stats"])
            assert result.exit_code == 0
            # The output uses Rich table formatting, so just check for key content
            assert "Statistics" in result.output
            assert "Total Entries" in result.output
            assert "10" in result.output
            mock_cache_instance.get_stats.assert_called()

            # Test clear and stats together
            result = runner.invoke(config_group, ["cache", "--clear", "--stats"])
            assert result.exit_code == 0
            assert "cache cleared" in result.output
            assert "Statistics" in result.output

    def test_cache_default_info_when_enabled(self):
        """Test default cache info when caching is enabled."""
        mock_cache = patch("metareason.config.cache.get_global_cache")
        mock_enabled = patch(
            "metareason.config.cache.is_caching_enabled", return_value=True
        )

        from unittest.mock import MagicMock

        mock_cache_instance = MagicMock()
        mock_cache_instance.get_stats.return_value = {
            "active_entries": 5,
        }

        with mock_cache as mock_get_cache, mock_enabled:
            mock_get_cache.return_value = mock_cache_instance

            runner = CliRunner()
            result = runner.invoke(config_group, ["cache"])

            assert result.exit_code == 0
            assert "5 active entries" in result.output


class TestConfigCommandIntegration:
    """Test integration between config commands."""

    def test_validate_then_show(self):
        """Test validating then showing a configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: integration_test
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()

            # First validate
            result = runner.invoke(config_group, ["validate", str(temp_path)])
            assert result.exit_code == 0

            # Then show
            result = runner.invoke(config_group, ["show", str(temp_path)])
            assert result.exit_code == 0
            assert "integration_test" in result.output
        finally:
            temp_path.unlink()

    def test_validate_cache_interaction(self):
        """Test that validation interacts properly with cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
prompt_id: cache_test
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()

            # Clear cache first
            runner.invoke(config_group, ["cache", "--clear"])

            # Validate (should cache)
            result = runner.invoke(config_group, ["validate", str(temp_path)])
            assert result.exit_code == 0

            # Check cache stats
            result = runner.invoke(config_group, ["cache", "--stats"])
            assert result.exit_code == 0
        finally:
            temp_path.unlink()


class TestDiffFormattingFunctions:
    """Test the diff formatting functions directly."""

    def test_format_text_diff_with_all_change_types(self):
        """Test _format_text_diff with added, removed, and modified changes."""
        from metareason.cli.config import _format_text_diff

        diff_result = {
            "has_differences": True,
            "file1_name": "config1.yaml",
            "file2_name": "config2.yaml",
            "changes": [
                {"type": "added", "path": "new_field", "new_value": "new_value"},
                {"type": "removed", "path": "old_field", "old_value": "old_value"},
                {
                    "type": "modified",
                    "path": "changed_field",
                    "old_value": "original",
                    "new_value": "updated",
                },
            ],
            "summary": {"total_changes": 3},
        }

        formatted = _format_text_diff(diff_result)

        # Check all sections are present
        assert "➕" in formatted  # Added section
        assert "➖" in formatted  # Removed section
        assert "🔄" in formatted  # Modified section
        assert "new_field: new_value" in formatted
        assert "old_field: old_value" in formatted
        assert "changed_field:" in formatted
        assert "- original" in formatted
        assert "+ updated" in formatted
        assert "3 changes" in formatted

    def test_format_text_diff_with_only_added_changes(self):
        """Test _format_text_diff with only added changes."""
        from metareason.cli.config import _format_text_diff

        diff_result = {
            "has_differences": True,
            "file1_name": "config1.yaml",
            "file2_name": "config2.yaml",
            "changes": [
                {"type": "added", "path": "new_field1", "new_value": "value1"},
                {"type": "added", "path": "new_field2", "new_value": "value2"},
            ],
            "summary": {"total_changes": 2},
        }

        formatted = _format_text_diff(diff_result)

        # Should only have added section
        assert "➕" in formatted
        assert "➖" not in formatted
        assert "🔄" not in formatted
        assert "new_field1: value1" in formatted
        assert "new_field2: value2" in formatted

    def test_format_text_diff_with_only_removed_changes(self):
        """Test _format_text_diff with only removed changes."""
        from metareason.cli.config import _format_text_diff

        diff_result = {
            "has_differences": True,
            "file1_name": "config1.yaml",
            "file2_name": "config2.yaml",
            "changes": [
                {
                    "type": "removed",
                    "path": "deleted_field1",
                    "old_value": "old_value1",
                },
                {
                    "type": "removed",
                    "path": "deleted_field2",
                    "old_value": "old_value2",
                },
            ],
            "summary": {"total_changes": 2},
        }

        formatted = _format_text_diff(diff_result)

        # Should only have removed section
        assert "➕" not in formatted
        assert "➖" in formatted
        assert "🔄" not in formatted
        assert "deleted_field1: old_value1" in formatted
        assert "deleted_field2: old_value2" in formatted

    def test_format_text_diff_with_only_modified_changes(self):
        """Test _format_text_diff with only modified changes."""
        from metareason.cli.config import _format_text_diff

        diff_result = {
            "has_differences": True,
            "file1_name": "config1.yaml",
            "file2_name": "config2.yaml",
            "changes": [
                {
                    "type": "modified",
                    "path": "field1",
                    "old_value": "old1",
                    "new_value": "new1",
                },
                {
                    "type": "modified",
                    "path": "field2",
                    "old_value": "old2",
                    "new_value": "new2",
                },
            ],
            "summary": {"total_changes": 2},
        }

        formatted = _format_text_diff(diff_result)

        # Should only have modified section
        assert "➕" not in formatted
        assert "➖" not in formatted
        assert "🔄" in formatted
        assert "field1:" in formatted
        assert "- old1" in formatted
        assert "+ new1" in formatted
        assert "field2:" in formatted
        assert "- old2" in formatted
        assert "+ new2" in formatted
