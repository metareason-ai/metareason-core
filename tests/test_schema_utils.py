"""Tests for schema utilities module."""

import json
import tempfile
from pathlib import Path

import pytest

from metareason.adapters.schema_utils import (
    SchemaError,
    clear_schema_cache,
    convert_schema_for_google,
    convert_schema_for_openai,
    create_anthropic_schema_prompt,
    get_cache_stats,
    load_schema_file,
    validate_schema_structure,
)


class TestSchemaLoading:
    """Test schema file loading functionality."""

    def test_load_valid_schema(self):
        """Test loading a valid JSON schema file."""
        schema_data = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "test_schema.json"
            with open(schema_file, "w") as f:
                json.dump(schema_data, f)

            loaded_schema = load_schema_file("test_schema.json", temp_dir)
            assert loaded_schema == schema_data

    def test_load_nonexistent_schema(self):
        """Test loading a non-existent schema file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loaded_schema = load_schema_file("nonexistent.json", temp_dir)
            assert loaded_schema is None

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises SchemaError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "invalid.json"
            with open(schema_file, "w") as f:
                f.write("{ invalid json }")

            with pytest.raises(SchemaError, match="Invalid JSON"):
                load_schema_file("invalid.json", temp_dir)

    def test_load_directory_instead_of_file(self):
        """Test loading a directory instead of file raises SchemaError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_dir = Path(temp_dir) / "schema_dir"
            schema_dir.mkdir()

            with pytest.raises(SchemaError, match="not a file"):
                load_schema_file("schema_dir", temp_dir)

    def test_schema_caching(self):
        """Test that schemas are cached after first load."""
        clear_schema_cache()

        schema_data = {"type": "object", "properties": {}}

        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "cached_schema.json"
            with open(schema_file, "w") as f:
                json.dump(schema_data, f)

            # First load
            loaded1 = load_schema_file("cached_schema.json", temp_dir)
            cache_stats1 = get_cache_stats()

            # Second load (from cache)
            loaded2 = load_schema_file("cached_schema.json", temp_dir)
            cache_stats2 = get_cache_stats()

            assert loaded1 == loaded2 == schema_data
            assert cache_stats1["cached_schemas"] == 1
            assert cache_stats2["cached_schemas"] == 1

    def test_clear_cache(self):
        """Test clearing the schema cache."""
        # Clear cache at start to isolate test
        clear_schema_cache()

        schema_data = {"type": "object"}

        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "cache_test.json"
            with open(schema_file, "w") as f:
                json.dump(schema_data, f)

            initial_count = get_cache_stats()["cached_schemas"]
            load_schema_file("cache_test.json", temp_dir)
            after_load_count = get_cache_stats()["cached_schemas"]

            # Should have one more schema cached
            assert after_load_count == initial_count + 1

            clear_schema_cache()
            assert get_cache_stats()["cached_schemas"] == 0

    def test_empty_file_path(self):
        """Test empty file path returns None."""
        result = load_schema_file("", "/tmp")
        assert result is None


class TestSchemaValidation:
    """Test schema validation functionality."""

    def test_validate_valid_object_schema(self):
        """Test validating a valid object schema."""
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        # Should not raise any exception
        validate_schema_structure(schema, "test.json")

    def test_validate_missing_type(self):
        """Test validation fails when type is missing."""
        schema = {"properties": {"score": {"type": "number"}}}
        with pytest.raises(SchemaError, match="must have a 'type' field"):
            validate_schema_structure(schema, "test.json")

    def test_validate_invalid_type(self):
        """Test validation fails with invalid type."""
        schema = {"type": "invalid_type"}
        with pytest.raises(SchemaError, match="Invalid schema type"):
            validate_schema_structure(schema, "test.json")

    def test_validate_non_dict(self):
        """Test validation fails for non-dictionary schema."""
        schema = ["not", "a", "dict"]
        with pytest.raises(SchemaError, match="must be a JSON object"):
            validate_schema_structure(schema, "test.json")


class TestSchemaConversion:
    """Test schema format conversion functionality."""

    def test_convert_for_openai(self):
        """Test converting schema for OpenAI format."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }

        result = convert_schema_for_openai(schema, "test_schema")

        expected = {
            "type": "json_schema",
            "json_schema": {"name": "test_schema", "schema": schema},
        }

        assert result == expected

    def test_convert_for_openai_default_name(self):
        """Test converting schema for OpenAI with default name."""
        schema = {"type": "object", "properties": {}}

        result = convert_schema_for_openai(schema)

        assert result["json_schema"]["name"] == "response_schema"

    def test_convert_for_google(self):
        """Test converting schema for Google format."""
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}

        result = convert_schema_for_google(schema)

        expected = {"response_mime_type": "application/json", "response_schema": schema}

        assert result == expected

    def test_create_anthropic_prompt(self):
        """Test creating Anthropic schema prompt."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }

        prompt = create_anthropic_schema_prompt(schema)

        assert "Please respond with valid JSON" in prompt
        assert "score" in prompt
        assert "reasoning" in prompt
        assert "required" in prompt
        assert "Do not include any text before or after the JSON response" in prompt


class TestCacheStatistics:
    """Test cache statistics functionality."""

    def test_get_cache_stats_empty(self):
        """Test getting cache stats when cache is empty."""
        clear_schema_cache()
        stats = get_cache_stats()

        assert stats["cached_schemas"] == 0
        assert stats["cache_paths"] == []

    def test_get_cache_stats_with_schemas(self):
        """Test getting cache stats with cached schemas."""
        clear_schema_cache()

        schema_data = {"type": "object"}

        with tempfile.TemporaryDirectory() as temp_dir:
            schema_file = Path(temp_dir) / "stats_test.json"
            with open(schema_file, "w") as f:
                json.dump(schema_data, f)

            load_schema_file("stats_test.json", temp_dir)
            stats = get_cache_stats()

            assert stats["cached_schemas"] == 1
            assert len(stats["cache_paths"]) == 1
            assert "stats_test.json" in str(stats["cache_paths"][0])
