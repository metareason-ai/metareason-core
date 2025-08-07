"""Environment variable substitution for MetaReason configurations."""

import os
import re
from typing import Any, Dict, Union


class EnvironmentSubstitutionError(Exception):
    """Exception raised when environment variable substitution fails."""

    pass


def substitute_environment_variables(
    value: Any,
    strict: bool = False,
    default_prefix: str = "${",
    default_suffix: str = "}",
) -> Any:
    """Substitute environment variables in configuration values.

    Supports the following formats:
    - ${VAR} - Required variable, raises error if not found
    - ${VAR:default} - Optional variable with default value
    - ${VAR:-default} - Optional variable with default (bash-style)
    - ${VAR:?error_message} - Required with custom error message

    Args:
        value: Value to process (can be string, dict, list, or primitive)
        strict: If True, all variables must be defined (no defaults allowed)
        default_prefix: Prefix for variable substitution (default: "${")
        default_suffix: Suffix for variable substitution (default: "}")

    Returns:
        Value with environment variables substituted

    Raises:
        EnvironmentSubstitutionError: If required variables are missing
    """
    if isinstance(value, str):
        return _substitute_in_string(value, strict, default_prefix, default_suffix)
    elif isinstance(value, dict):
        return {
            k: substitute_environment_variables(
                v, strict, default_prefix, default_suffix
            )
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [
            substitute_environment_variables(
                item, strict, default_prefix, default_suffix
            )
            for item in value
        ]
    else:
        # Primitive types (int, float, bool, None) - return as-is
        return value


def _substitute_in_string(
    text: str, strict: bool, prefix: str, suffix: str
) -> Union[str, int, float, bool]:
    """Substitute environment variables in a string value."""
    if prefix not in text:
        return text

    # Escape special regex characters
    escaped_prefix = re.escape(prefix)
    escaped_suffix = re.escape(suffix)

    # Pattern to match ${VAR}, ${VAR:default}, ${VAR:-default}, ${VAR:?message}
    pattern = f"{escaped_prefix}([A-Za-z_][A-Za-z0-9_]*)([:?-].*?)?{escaped_suffix}"

    def replace_var(match: Any) -> str:
        var_name = match.group(1)
        modifier = match.group(2)

        # Get environment variable value
        env_value = os.environ.get(var_name)

        if env_value is not None:
            # Variable exists, return its value as string (coercion happens later)
            return env_value

        # Variable doesn't exist, handle based on modifier
        if modifier is None:
            # ${VAR} format - required variable
            if strict:
                raise EnvironmentSubstitutionError(
                    f"Required environment variable '{var_name}' is not set. "
                    f"Suggestion: Set the variable with 'export {var_name}=value'"
                )
            else:
                # In non-strict mode, leave unchanged for debugging
                return str(match.group(0))

        elif modifier.startswith(":?"):
            # ${VAR:?message} format - required with custom error
            error_msg = modifier[2:] or f"Variable {var_name} is required"
            raise EnvironmentSubstitutionError(
                f"Environment variable substitution failed: {error_msg}. "
                f"Suggestion: Set the variable with 'export {var_name}=value'"
            )

        elif modifier.startswith(":-") or modifier.startswith(":"):
            # ${VAR:-default} or ${VAR:default} format - optional with default
            if strict:
                raise EnvironmentSubstitutionError(
                    f"Environment variable '{var_name}' is not set and strict mode "
                    f"is enabled. Suggestion: Set the variable with 'export "
                    f"{var_name}=value'"
                )

            default_value = modifier[2:] if modifier.startswith(":-") else modifier[1:]
            coerced = _coerce_type(default_value)
            return str(coerced) if isinstance(coerced, (int, float, bool)) else coerced

        else:
            # Unknown modifier format
            raise EnvironmentSubstitutionError(
                f"Invalid environment variable syntax: {match.group(0)}. "
                "Supported formats: ${VAR}, ${VAR:default}, ${VAR:-default}, "
                "${VAR:?message}"
            )

    try:
        result = re.sub(pattern, replace_var, text)
        # Apply type coercion if the entire string was a single variable
        if (
            text.count(prefix) == 1
            and text.startswith(prefix)
            and text.endswith(suffix)
            and result != text
        ):
            return _coerce_type(result)
        return result
    except EnvironmentSubstitutionError:
        raise
    except Exception as e:
        raise EnvironmentSubstitutionError(
            f"Error during environment variable substitution: {e}"
        ) from e


def _coerce_type(value: str) -> Union[str, int, float, bool]:
    """Coerce string value to appropriate type."""
    if not isinstance(value, str):
        return value  # type: ignore

    # Handle boolean values
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    elif value.lower() in ("false", "no", "0", "off"):
        return False

    # Handle numeric values
    try:
        # Try integer first
        if "." not in value and "e" not in value.lower():
            return int(value)
        else:
            return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def get_environment_info() -> Dict[str, Any]:
    """Get information about environment variables used in configuration.

    Returns:
        Dictionary with environment variable information
    """
    env_vars = {}

    # Get all environment variables that might be relevant
    for key, value in os.environ.items():
        if key.startswith(("METAREASON_", "MR_")):
            env_vars[key] = {
                "value": value,
                "length": len(value),
                "is_sensitive": _is_sensitive_var(key),
            }

    return {
        "metareason_vars": env_vars,
        "total_env_vars": len(os.environ),
        "common_config_vars": {
            key: os.environ.get(key, "<not set>")
            for key in [
                "HOME",
                "USER",
                "PATH",
                "PWD",
                "METAREASON_CONFIG_DIR",
                "METAREASON_LOG_LEVEL",
                "METAREASON_CACHE_DIR",
            ]
        },
    }


def _is_sensitive_var(var_name: str) -> bool:
    """Check if an environment variable is likely to contain sensitive data."""
    sensitive_keywords = [
        "key",
        "secret",
        "token",
        "password",
        "pwd",
        "pass",
        "credential",
        "auth",
        "api_key",
        "private",
    ]

    var_lower = var_name.lower()
    return any(keyword in var_lower for keyword in sensitive_keywords)


def validate_required_environment_vars(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that all required environment variables are available.

    Args:
        config_data: Configuration data to scan for environment variables

    Returns:
        Dictionary with validation results

    Raises:
        EnvironmentSubstitutionError: If required variables are missing
    """
    missing_vars = []
    found_vars = []

    def scan_for_vars(obj: Any, path: str = "") -> None:
        """Recursively scan for environment variable references."""
        if isinstance(obj, str):
            # Look for ${VAR} patterns
            pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)([:?-].*?)?\}"
            matches = re.findall(pattern, obj)

            for var_name, modifier in matches:
                var_path = f"{path}.{var_name}" if path else var_name

                if os.environ.get(var_name) is not None:
                    found_vars.append(
                        {
                            "name": var_name,
                            "path": var_path,
                            "has_default": modifier
                            and (modifier.startswith(":") or modifier.startswith(":-")),
                        }
                    )
                else:
                    # Check if it has a default value
                    has_default = modifier and (
                        modifier.startswith(":") and not modifier.startswith(":?")
                    )

                    if not has_default:
                        missing_vars.append(
                            {
                                "name": var_name,
                                "path": var_path,
                                "modifier": modifier or "required",
                            }
                        )

        elif isinstance(obj, dict):
            for key, value in obj.items():
                scan_for_vars(value, f"{path}.{key}" if path else key)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                scan_for_vars(item, f"{path}[{i}]" if path else f"[{i}]")

    scan_for_vars(config_data)

    if missing_vars:
        error_lines = ["Missing required environment variables:"]
        for var in missing_vars:
            error_lines.append(f"  - {var['name']} (used in {var['path']})")

        error_lines.append("\nSuggestions:")
        for var in missing_vars:
            error_lines.append(f"  export {var['name']}=your_value")

        raise EnvironmentSubstitutionError("\n".join(error_lines))

    return {
        "missing_vars": missing_vars,
        "found_vars": found_vars,
        "total_vars_found": len(found_vars),
        "all_required_available": len(missing_vars) == 0,
    }
