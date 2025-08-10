"""Simplified tests for CLI template commands focusing on code coverage."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from metareason.cli.templates import template_group


class TestTemplateCommands:
    """Test the template CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_template_group_help(self, runner):
        """Test template group help."""
        result = runner.invoke(template_group, ["--help"])

        assert result.exit_code == 0
        assert "Template validation and rendering commands" in result.output
        assert "validate" in result.output
        assert "render" in result.output
        assert "test" in result.output
        assert "filters" in result.output

    def test_validate_command_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(template_group, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate template syntax and security" in result.output

    def test_render_command_help(self, runner):
        """Test render command help."""
        result = runner.invoke(template_group, ["render", "--help"])

        assert result.exit_code == 0
        assert "Render sample prompts from configuration" in result.output

    def test_test_command_help(self, runner):
        """Test test command help."""
        result = runner.invoke(template_group, ["test", "--help"])

        assert result.exit_code == 0
        assert "Test a template string directly" in result.output

    def test_filters_command_help(self, runner):
        """Test filters command help."""
        result = runner.invoke(template_group, ["filters", "--help"])

        assert result.exit_code == 0
        assert "List available custom template filters" in result.output

    def test_filters_command(self, runner):
        """Test the filters command."""
        result = runner.invoke(template_group, ["filters"])

        assert result.exit_code == 0
        assert "Available Custom Filters" in result.output
        assert "format_continuous" in result.output
        assert "format_list" in result.output
        assert "conditional_text" in result.output
        assert "Example Template" in result.output

    def test_test_template_basic(self, runner):
        """Test basic template testing."""
        template = "Hello {{name}}!"

        result = runner.invoke(template_group, ["test", template])

        assert result.exit_code == 0
        assert "Template Test Results" in result.output
        assert "Template is valid" in result.output

    def test_test_template_with_context(self, runner):
        """Test template testing with context."""
        template = "Hello {{name}}!"
        context = '{"name": "World"}'

        result = runner.invoke(template_group, ["test", template, "--context", context])

        assert result.exit_code == 0
        assert "Template is valid" in result.output
        assert "Rendered Output" in result.output
        assert "Hello World!" in result.output

    def test_test_template_with_variables(self, runner):
        """Test template testing with expected variables."""
        template = "Hello {{name}}!"

        result = runner.invoke(
            template_group, ["test", template, "--variables", "name"]
        )

        assert result.exit_code == 0
        assert "Template is valid" in result.output

    def test_test_template_invalid_syntax(self, runner):
        """Test template with invalid syntax."""
        template = "Hello {{name} missing brace!"

        result = runner.invoke(template_group, ["test", template])

        assert result.exit_code == 0
        assert "Template validation failed" in result.output

    def test_test_template_invalid_context(self, runner):
        """Test template with invalid JSON context."""
        template = "Hello {{name}}!"

        result = runner.invoke(
            template_group, ["test", template, "--context", "invalid json"]
        )

        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_test_template_complex(self, runner):
        """Test complex template."""
        template = "Hello {{ name | upper }}!"
        context = '{"name": "world"}'

        result = runner.invoke(template_group, ["test", template, "--context", context])

        assert result.exit_code == 0

    def test_nonexistent_file_validation(self, runner):
        """Test validation with non-existent config file."""
        result = runner.invoke(template_group, ["validate", "/nonexistent/file.yaml"])

        assert result.exit_code == 2  # Click's file not found error

    def test_nonexistent_file_render(self, runner):
        """Test rendering with non-existent config file."""
        result = runner.invoke(template_group, ["render", "/nonexistent/file.yaml"])

        assert result.exit_code == 2  # Click's file not found error

    @patch("metareason.cli.templates.load_yaml_config")
    def test_validate_config_load_error(self, mock_load, runner):
        """Test handling of config loading errors in validate."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            mock_load.side_effect = Exception("Config loading failed")

            result = runner.invoke(template_group, ["validate", f.name])

            assert result.exit_code == 1
            assert "Error loading configuration" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    def test_render_config_load_error(self, mock_load, runner):
        """Test handling of config loading errors in render."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            mock_load.side_effect = Exception("Config loading failed")

            result = runner.invoke(template_group, ["render", f.name])

            assert result.exit_code == 1
            assert "Error:" in result.output

    def test_invalid_template_command(self, runner):
        """Test invalid template command."""
        result = runner.invoke(template_group, ["invalid-command"])

        assert result.exit_code == 2
        assert "No such command" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.TemplateValidator")
    def test_validate_successful(self, mock_validator_class, mock_load, runner):
        """Test successful template validation."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.prompt_template = "Hello {{name}}!"
            mock_config.axes = {"name": MagicMock()}
            mock_load.return_value = mock_config

            # Mock validator
            mock_validator = MagicMock()
            mock_result = MagicMock()
            mock_result.is_valid = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.variables = {"name"}
            mock_result.metadata = {"length": 10, "lines": 1}
            mock_validator.validate.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = runner.invoke(template_group, ["validate", f.name])

            assert result.exit_code == 0
            assert "Template is valid" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.TemplateValidator")
    def test_validate_json_output(self, mock_validator_class, mock_load, runner):
        """Test validate with JSON output."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.prompt_template = "Hello {{name}}!"
            mock_config.axes = {"name": MagicMock()}
            mock_load.return_value = mock_config

            # Mock validator
            mock_validator = MagicMock()
            mock_result = MagicMock()
            mock_result.is_valid = True
            mock_result.errors = []
            mock_result.warnings = []
            mock_result.variables = {"name"}
            mock_result.metadata = {"length": 10}
            mock_validator.validate.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = runner.invoke(
                template_group, ["validate", f.name, "--format", "json"]
            )

            assert result.exit_code == 0

            # Should contain JSON output
            json_data = json.loads(result.output.strip())
            assert json_data["is_valid"] is True

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.TemplateValidator")
    def test_validate_with_errors(self, mock_validator_class, mock_load, runner):
        """Test validation with errors."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.prompt_template = "Hello {{name}!"  # Invalid syntax
            mock_config.axes = {"name": MagicMock()}
            mock_load.return_value = mock_config

            # Mock validator
            mock_validator = MagicMock()
            mock_result = MagicMock()
            mock_result.is_valid = False
            mock_result.errors = ["Invalid syntax"]
            mock_result.warnings = ["Warning message"]
            mock_result.variables = set()
            mock_result.metadata = {}
            mock_validator.validate.return_value = mock_result
            mock_validator_class.return_value = mock_validator

            result = runner.invoke(template_group, ["validate", f.name])

            assert result.exit_code == 0  # Command succeeds but shows errors
            assert "Template validation failed" in result.output
            assert "Errors:" in result.output
            assert "Warnings:" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.create_sampler")
    @patch("metareason.cli.templates.generate_prompts_from_config")
    def test_render_successful(self, mock_generate, mock_sampler, mock_load, runner):
        """Test successful prompt rendering."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.n_variants = 5
            mock_config.axes = {}
            mock_config.sampling = None
            mock_load.return_value = mock_config

            # Mock sampler
            mock_sampler.return_value = MagicMock()

            # Mock generation result
            mock_result = MagicMock()
            mock_result.is_successful = True
            mock_result.prompts = ["Prompt 1", "Prompt 2"]
            mock_result.contexts = [{"var": "val1"}, {"var": "val2"}]
            mock_result.render_result.success_count = 2
            mock_result.render_result.success_rate = 100.0
            mock_result.render_result.render_time = 0.1
            mock_result.render_result.errors = []
            mock_generate.return_value = mock_result

            result = runner.invoke(template_group, ["render", f.name, "--samples", "2"])

            assert result.exit_code == 0
            assert "Generated 2 prompts" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.create_sampler")
    @patch("metareason.cli.templates.generate_prompts_from_config")
    def test_render_with_failure(self, mock_generate, mock_sampler, mock_load, runner):
        """Test prompt rendering with failure."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.n_variants = 5
            mock_config.axes = {}
            mock_config.sampling = None
            mock_load.return_value = mock_config

            # Mock sampler
            mock_sampler.return_value = MagicMock()

            # Mock failed generation result
            mock_result = MagicMock()
            mock_result.is_successful = False
            mock_result.validation_result.errors = ["Generation failed"]
            mock_generate.return_value = mock_result

            result = runner.invoke(template_group, ["render", f.name])

            assert result.exit_code == 0  # Command succeeds but shows error
            assert "Prompt generation failed" in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.create_sampler")
    @patch("metareason.cli.templates.generate_prompts_from_config")
    def test_render_json_output(self, mock_generate, mock_sampler, mock_load, runner):
        """Test rendering with JSON output."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.n_variants = 2
            mock_config.axes = {}
            mock_config.sampling = None
            mock_load.return_value = mock_config

            # Mock sampler
            mock_sampler.return_value = MagicMock()

            # Mock generation result
            mock_result = MagicMock()
            mock_result.is_successful = True
            mock_result.prompts = ["Prompt 1", "Prompt 2"]
            mock_result.contexts = [{}, {}]
            mock_result.render_result.success_count = 2
            mock_result.render_result.success_rate = 100.0
            mock_result.render_result.render_time = 0.1
            mock_result.render_result.errors = []
            mock_generate.return_value = mock_result

            result = runner.invoke(
                template_group, ["render", f.name, "--format", "json", "--samples", "2"]
            )

            assert result.exit_code == 0
            # Should contain JSON in output
            assert '"prompts"' in result.output
            assert '"statistics"' in result.output

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.create_sampler")
    @patch("metareason.cli.templates.generate_prompts_from_config")
    def test_render_jsonl_output(self, mock_generate, mock_sampler, mock_load, runner):
        """Test rendering with JSONL output."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            # Mock config
            mock_config = MagicMock()
            mock_config.n_variants = 2
            mock_config.axes = {}
            mock_config.sampling = None
            mock_load.return_value = mock_config

            # Mock sampler
            mock_sampler.return_value = MagicMock()

            # Mock generation result
            mock_result = MagicMock()
            mock_result.is_successful = True
            mock_result.prompts = ["Prompt 1", "Prompt 2"]
            mock_result.contexts = [{}, {}]
            mock_result.render_result.success_count = 2
            mock_result.render_result.success_rate = 100.0
            mock_result.render_result.render_time = 0.1
            mock_result.render_result.errors = []
            mock_generate.return_value = mock_result

            result = runner.invoke(
                template_group,
                ["render", f.name, "--format", "jsonl", "--samples", "2"],
            )

            assert result.exit_code == 0
            # Check for JSONL format lines
            lines = [
                line
                for line in result.output.split("\n")
                if line.strip().startswith("{")
            ]
            assert len(lines) >= 2  # Should have at least 2 JSON lines

    @patch("metareason.cli.templates.load_yaml_config")
    @patch("metareason.cli.templates.create_sampler")
    @patch("metareason.cli.templates.generate_prompts_from_config")
    def test_render_to_file(self, mock_generate, mock_sampler, mock_load, runner):
        """Test rendering to output file."""
        with (
            tempfile.NamedTemporaryFile(suffix=".yaml") as config_file,
            tempfile.NamedTemporaryFile(mode="w", delete=False) as output_file,
        ):

            # Mock config
            mock_config = MagicMock()
            mock_config.n_variants = 2
            mock_config.axes = {}
            mock_config.sampling = None
            mock_load.return_value = mock_config

            # Mock sampler
            mock_sampler.return_value = MagicMock()

            # Mock generation result
            mock_result = MagicMock()
            mock_result.is_successful = True
            mock_result.prompts = ["Prompt 1", "Prompt 2"]
            mock_result.contexts = [{}, {}]
            mock_result.render_result.success_count = 2
            mock_result.render_result.success_rate = 100.0
            mock_result.render_result.render_time = 0.1
            mock_result.render_result.errors = []
            mock_generate.return_value = mock_result

            output_path = Path(output_file.name)

            result = runner.invoke(
                template_group,
                [
                    "render",
                    config_file.name,
                    "--output",
                    str(output_path),
                    "--samples",
                    "2",
                ],
            )

            assert result.exit_code == 0
            assert "Output saved to" in result.output
            assert output_path.exists()

            # Clean up
            output_path.unlink(missing_ok=True)

    def test_test_template_with_multiple_variables(self, runner):
        """Test template with multiple variables."""
        template = "Hello {{name}} and {{friend}}!"

        result = runner.invoke(
            template_group, ["test", template, "--variables", "name,friend"]
        )

        assert result.exit_code == 0
        assert "Template is valid" in result.output
