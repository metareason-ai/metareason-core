"""Tests for CLI main module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from metareason.cli.main import cli


class TestMainCLI:
    """Test the main CLI command."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "MetaReason CLI" in result.output
        assert "framework for LLM sampling" in result.output

    def test_cli_version(self):
        """Test CLI version flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should contain version information
        assert any(char.isdigit() for char in result.output)

    def test_cli_verbose_flag(self):
        """Test CLI verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info"])

        assert result.exit_code == 0
        assert "MetaReason CLI initialized" in result.output

    def test_cli_config_dir_option(self):
        """Test CLI config-dir option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            result = runner.invoke(cli, ["--config-dir", temp_dir, "--help"])

            assert result.exit_code == 0

    def test_cli_config_dir_nonexistent(self):
        """Test CLI with non-existent config directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config-dir", "/nonexistent/directory", "info"])

        assert result.exit_code == 2  # Click validation error
        assert "does not exist" in result.output


class TestInfoCommand:
    """Test the info command."""

    def test_info_text_format(self):
        """Test info command with text format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "MetaReason CLI" in result.output
        assert "Python" in result.output
        assert "Cache:" in result.output

    def test_info_json_format(self):
        """Test info command with JSON format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--format", "json"])

        assert result.exit_code == 0

        # Should be valid JSON
        try:
            data = json.loads(result.output)
            assert "metareason" in data
            assert "version" in data["metareason"]
            assert "system" in data
            assert "cache" in data
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_info_yaml_format(self):
        """Test info command with YAML format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--format", "yaml"])

        assert result.exit_code == 0
        assert "metareason:" in result.output
        assert "version:" in result.output
        assert "system:" in result.output

    def test_info_with_verbose(self):
        """Test info command with verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info"])

        assert result.exit_code == 0
        assert "MetaReason CLI initialized" in result.output
        assert "MetaReason CLI" in result.output


class TestRunCommand:
    """Test the run command."""

    def test_run_valid_config(self):
        """Test run command with valid configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test_run
pipeline:
  - template: "Hello {{name}}"
    adapter: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
n_variants: 100
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
sampling:
  method: latin_hypercube
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "--spec-file", str(temp_path)])

            assert result.exit_code == 0
            assert "Running evaluation" in result.output
            assert "not yet implemented" in result.output
        finally:
            temp_path.unlink()

    def test_run_dry_run(self):
        """Test run command with dry-run flag."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test_dry_run
pipeline:
  - template: "Hello {{name}}"
    adapter: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
n_variants: 100
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
sampling:
  method: latin_hypercube
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["run", "--spec-file", str(temp_path), "--dry-run"]
            )

            if result.exit_code != 0:
                print(f"Error output: {result.output}")

            assert result.exit_code == 0
            assert "Would run evaluation" in result.output
            assert "test_dry_run" in result.output
            assert "Variants: 100" in result.output
            assert "latin_hypercube" in result.output
        finally:
            temp_path.unlink()

    def test_run_with_output_file(self):
        """Test run command with output file specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test_output
pipeline:
  - template: "Hello {{name}}"
    adapter: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
n_variants: 100
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
sampling:
  method: latin_hypercube
"""
            )
            temp_path = Path(f.name)

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False
            ) as output_file:
                output_path = Path(output_file.name)

            try:
                runner = CliRunner()
                result = runner.invoke(
                    cli,
                    [
                        "run",
                        "--spec-file",
                        str(temp_path),
                        "--output",
                        str(output_path),
                        "--dry-run",
                    ],
                )

                assert result.exit_code == 0
                assert str(output_path) in result.output
            finally:
                output_path.unlink()
        finally:
            temp_path.unlink()

    def test_run_invalid_config(self):
        """Test run command with invalid configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: ""  # Invalid empty ID
pipeline:
  - template: "Hello {{name}}"
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "--spec-file", str(temp_path)])

            assert result.exit_code == 1
            assert "Error loading specification" in result.output
        finally:
            temp_path.unlink()

    def test_run_nonexistent_config(self):
        """Test run command with non-existent configuration file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--spec-file", "/nonexistent/config.yaml"])

        assert result.exit_code == 2  # Click validation error
        assert "does not exist" in result.output

    def test_run_with_verbose(self):
        """Test run command with verbose flag."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test_verbose
pipeline:
  - template: "Hello {{name}}"
    adapter: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
sampling:
  method: latin_hypercube
n_variants: 100
"""
            )
            temp_path = Path(f.name)

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["--verbose", "run", "--spec-file", str(temp_path), "--dry-run"]
            )

            assert result.exit_code == 0
            assert "MetaReason CLI initialized" in result.output
            assert "Would run evaluation" in result.output
        finally:
            temp_path.unlink()


class TestCLIIntegration:
    """Test CLI integration and command discovery."""

    def test_available_commands(self):
        """Test that all expected commands are available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        # Check that config command group is available
        assert "config" in result.output
        assert "info" in result.output
        assert "run" in result.output

    def test_config_command_group_available(self):
        """Test that config command group is properly registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])

        assert result.exit_code == 0
        assert (
            "Configuration management commands" in result.output
            or "config" in result.output.lower()
        )

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code == 2
        assert "No such command" in result.output

    @patch("metareason.cli.main.console")
    def test_console_initialization(self, mock_console):
        """Test that console is properly initialized."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info"])

        assert result.exit_code == 0
        # Console should be used for verbose output
        mock_console.print.assert_called()


class TestCLIContextPassing:
    """Test CLI context passing between commands."""

    def test_context_verbose_flag(self):
        """Test that verbose flag is properly stored in context."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "info", "--format", "text"])

        assert result.exit_code == 0
        assert "MetaReason CLI initialized" in result.output

    def test_context_config_dir(self):
        """Test that config directory is properly stored in context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            result = runner.invoke(cli, ["--config-dir", temp_dir, "info"])

            assert result.exit_code == 0
