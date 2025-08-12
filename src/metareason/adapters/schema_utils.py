"""Utilities for loading and validating JSON schemas for structured output."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cache for loaded schemas to improve performance
_schema_cache: Dict[str, Dict[str, Any]] = {}


class SchemaError(Exception):
    """Exception raised for schema-related errors."""

    pass


def load_schema_file(file_path: str, base_dir: str) -> Optional[Dict[str, Any]]:
    """Load JSON schema from file with caching and validation.

    Args:
        file_path: Relative path to the JSON schema file
        base_dir: Base directory to resolve relative paths from

    Returns:
        Loaded schema dictionary, or None if file doesn't exist

    Raises:
        SchemaError: If schema file exists but is invalid
    """
    if not file_path:
        return None

    # Create absolute path
    absolute_path = Path(base_dir) / file_path
    resolved_path = str(absolute_path.resolve())

    # Check cache first
    if resolved_path in _schema_cache:
        logger.debug(f"Using cached schema from {resolved_path}")
        return _schema_cache[resolved_path]

    # Check if file exists
    if not absolute_path.exists():
        logger.warning(f"Schema file not found: {absolute_path}")
        return None

    if not absolute_path.is_file():
        raise SchemaError(f"Schema path is not a file: {absolute_path}")

    # Load and parse JSON
    try:
        with open(absolute_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

        # Basic validation that it looks like a JSON schema
        validate_schema_structure(schema_data, file_path)

        # Cache the result
        _schema_cache[resolved_path] = schema_data
        logger.debug(f"Loaded and cached schema from {resolved_path}")

        return schema_data

    except json.JSONDecodeError as e:
        raise SchemaError(f"Invalid JSON in schema file {file_path}: {e}")
    except OSError as e:
        raise SchemaError(f"Failed to read schema file {file_path}: {e}")


def validate_schema_structure(schema_data: Dict[str, Any], file_path: str) -> None:
    """Validate that the schema data has a valid structure.

    Args:
        schema_data: The loaded schema dictionary
        file_path: File path for error messages

    Raises:
        SchemaError: If schema structure is invalid
    """
    if not isinstance(schema_data, dict):
        raise SchemaError(
            f"Schema in {file_path} must be a JSON object, got {type(schema_data).__name__}"
        )

    # Check for required fields for a valid JSON schema
    if "type" not in schema_data:
        raise SchemaError(f"Schema in {file_path} must have a 'type' field")

    # For object type, should have properties
    if schema_data.get("type") == "object" and "properties" not in schema_data:
        logger.warning(f"Object schema in {file_path} has no 'properties' field")

    # Basic validation of common schema fields
    valid_types = {"object", "array", "string", "number", "integer", "boolean", "null"}
    schema_type = schema_data["type"]

    if isinstance(schema_type, str) and schema_type not in valid_types:
        raise SchemaError(
            f"Invalid schema type '{schema_type}' in {file_path}. "
            f"Valid types: {sorted(valid_types)}"
        )


def clear_schema_cache() -> None:
    """Clear the schema cache. Useful for testing or development."""
    _schema_cache.clear()
    logger.debug("Schema cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the schema cache.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "cached_schemas": len(_schema_cache),
        "cache_paths": list(_schema_cache.keys()),
    }


def convert_schema_for_openai(
    schema_data: Dict[str, Any], schema_name: str = "response_schema"
) -> Dict[str, Any]:
    """Convert JSON schema to OpenAI's structured output format.

    Args:
        schema_data: The JSON schema dictionary
        schema_name: Name for the schema in OpenAI format

    Returns:
        OpenAI-formatted response_format dictionary
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema_data},
    }


def convert_schema_for_google(schema_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON schema to Google's response_schema format.

    Note: Google supports OpenAPI 3.0 schema format, which is similar to JSON Schema.

    Args:
        schema_data: The JSON schema dictionary

    Returns:
        Google-formatted generation config additions
    """
    return {"response_mime_type": "application/json", "response_schema": schema_data}


def create_anthropic_schema_prompt(schema_data: Dict[str, Any]) -> str:
    """Create prompt instructions for Anthropic based on schema.

    Args:
        schema_data: The JSON schema dictionary

    Returns:
        Prompt text with schema instructions
    """
    schema_str = json.dumps(schema_data, indent=2)

    return f"""Please respond with valid JSON that exactly matches this schema:

{schema_str}

Important:
- Your response must be valid JSON
- Include all required fields from the schema
- Follow the exact data types specified
- Do not include any text before or after the JSON response"""
