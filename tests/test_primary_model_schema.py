"""Tests for pipeline step configuration with JSON schema support."""

import pytest
from pydantic import ValidationError

from metareason.config.models import PipelineStep


class TestPipelineStepConfig:
    """Test the PipelineStep with json_schema field."""

    def test_basic_primary_model_config(self):
        """Test basic primary model configuration without schema."""
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )

        assert config.adapter == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.json_schema is None

    def test_primary_model_with_json_schema(self):
        """Test primary model configuration with JSON schema path."""
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4o",
            json_schema="schemas/basic_response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )

        assert config.adapter == "openai"
        assert config.model == "gpt-4o"
        assert config.json_schema == "schemas/basic_response.json"

    def test_json_schema_path_validation(self):
        """Test validation of JSON schema path format."""
        # Valid path
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4",
            json_schema="schemas/response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert config.json_schema == "schemas/response.json"

        # Valid nested path
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4",
            json_schema="path/to/schemas/response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert config.json_schema == "path/to/schemas/response.json"

    def test_json_schema_absolute_path_rejected(self):
        """Test that absolute paths are rejected for security."""
        with pytest.raises(ValidationError, match="must be a relative path"):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema="/absolute/path/schema.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

    def test_json_schema_windows_path_rejected(self):
        """Test that Windows absolute paths are rejected."""
        with pytest.raises(ValidationError, match="must be a relative path"):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema="C:\\absolute\\path\\schema.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

    def test_json_schema_non_json_extension_rejected(self):
        """Test that non-JSON extensions are rejected."""
        with pytest.raises(ValidationError, match="must end with '.json'"):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema="schemas/response.yaml",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

    def test_json_schema_empty_string_rejected(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValidationError, match="cannot be empty string"):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema="",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

    def test_json_schema_whitespace_stripped(self):
        """Test that whitespace is stripped from schema path."""
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4",
            json_schema="  schemas/response.json  ",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert config.json_schema == "schemas/response.json"

    def test_all_adapters_with_schema(self):
        """Test all supported adapters work with JSON schema."""
        adapters = [
            "openai",
            "anthropic",
            "google",
            "ollama",
            "azure_openai",
            "huggingface",
            "custom",
        ]

        for adapter in adapters:
            config = PipelineStep(
                template="Test template {{param}}",
                adapter=adapter,
                model="test-model",
                json_schema="schemas/test.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )
            assert config.adapter == adapter
            assert config.json_schema == "schemas/test.json"

    def test_optional_parameters_with_schema(self):
        """Test that all optional parameters work with schema."""
        config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=-0.1,
            stop=["STOP", "END"],
            json_schema="schemas/detailed.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )

        assert config.adapter == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == -0.1
        assert config.stop == ["STOP", "END"]
        assert config.json_schema == "schemas/detailed.json"

    def test_json_schema_with_different_models(self):
        """Test JSON schema with different model types."""
        # OpenAI model
        openai_config = PipelineStep(
            template="Test template {{param}}",
            adapter="openai",
            model="gpt-4o-mini",
            json_schema="schemas/openai_response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert openai_config.json_schema == "schemas/openai_response.json"

        # Anthropic model
        anthropic_config = PipelineStep(
            template="Test template {{param}}",
            adapter="anthropic",
            model="claude-3-sonnet-20240229",
            json_schema="schemas/anthropic_response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert anthropic_config.json_schema == "schemas/anthropic_response.json"

        # Google model
        google_config = PipelineStep(
            template="Test template {{param}}",
            adapter="google",
            model="gemini-2.0-flash-001",
            json_schema="schemas/google_response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert google_config.json_schema == "schemas/google_response.json"

        # Ollama model
        ollama_config = PipelineStep(
            template="Test template {{param}}",
            adapter="ollama",
            model="llama3",
            json_schema="schemas/ollama_response.json",
            axes={"param": {"type": "categorical", "values": ["test"]}},
        )
        assert ollama_config.json_schema == "schemas/ollama_response.json"

    def test_parameter_constraints_still_work(self):
        """Test that existing parameter constraints still work with schema."""
        # Temperature out of range
        with pytest.raises(ValidationError):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                temperature=3.0,
                json_schema="schemas/test.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

        # Top_p out of range
        with pytest.raises(ValidationError):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                top_p=1.5,
                json_schema="schemas/test.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

        # Invalid adapter
        with pytest.raises(ValidationError):
            PipelineStep(
                template="Test template {{param}}",
                adapter="invalid_adapter",
                model="gpt-4",
                json_schema="schemas/test.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

        # Empty model
        with pytest.raises(ValidationError):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="",
                json_schema="schemas/test.json",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )

    def test_schema_path_formats(self):
        """Test various valid schema path formats."""
        valid_paths = [
            "schema.json",
            "schemas/response.json",
            "deep/nested/path/schema.json",
            "schema_with_underscores.json",
            "schema-with-dashes.json",
            "123_numeric_prefix.json",
        ]

        for path in valid_paths:
            config = PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema=path,
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )
            assert config.json_schema == path

    def test_json_schema_case_sensitive_extension(self):
        """Test that .json extension is case-sensitive."""
        with pytest.raises(ValidationError, match="must end with '.json'"):
            PipelineStep(
                template="Test template {{param}}",
                adapter="openai",
                model="gpt-4",
                json_schema="Schema.JSON",
                axes={"param": {"type": "categorical", "values": ["test"]}},
            )
