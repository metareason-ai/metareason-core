"""YAML include/import functionality for MetaReason configurations."""

import os
from pathlib import Path
from typing import Any, Dict, Set, Union

import yaml


class IncludeLoader(yaml.SafeLoader):
    """Custom YAML loader that supports !include tags."""

    def __init__(self, stream, base_dir: Path):
        super().__init__(stream)
        self.base_dir = base_dir
        self.included_files: Set[Path] = set()

    def include(self, node):
        """Handle !include tag."""
        include_path = self.construct_scalar(node)

        # Resolve the include path relative to the current file
        if not os.path.isabs(include_path):
            include_path = self.base_dir / include_path
        else:
            include_path = Path(include_path)

        include_path = include_path.resolve()

        # Check for circular dependencies
        if include_path in self.included_files:
            raise yaml.YAMLError(
                f"Circular dependency detected: {include_path} is already being processed. "
                f"Included files: {sorted(str(f) for f in self.included_files)}"
            )

        # Check if file exists
        if not include_path.exists():
            raise FileNotFoundError(
                f"Include file not found: {include_path}. "
                f"Suggestion: Check the file path and ensure the file exists."
            )

        # Add to included files set to prevent circular dependencies
        self.included_files.add(include_path)

        try:
            # Load the included file
            with open(include_path, "r", encoding="utf-8") as f:
                # Create a nested loader class
                included_files_ref = self.included_files

                class NestedIncludeLoader(IncludeLoader):
                    def __init__(self, stream):
                        super().__init__(stream, include_path.parent)
                        self.included_files = included_files_ref.copy()

                try:
                    data = yaml.load(f, Loader=NestedIncludeLoader)
                    return data
                finally:
                    # Remove from included files set when done
                    self.included_files.discard(include_path)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse included file {include_path}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error processing included file {include_path}: {e}"
            ) from e


# Register the !include constructor
IncludeLoader.add_constructor("!include", IncludeLoader.include)


def load_yaml_with_includes(file_path: Union[str, Path]) -> Any:
    """Load a YAML file with support for !include tags.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML data with includes resolved

    Raises:
        FileNotFoundError: If the file or any included file doesn't exist
        yaml.YAMLError: If there are YAML parsing errors
        RuntimeError: If there are other processing errors
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            f"Suggestion: Check the file path and ensure the file exists."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            # Create a custom loader class for this specific file
            class FileSpecificIncludeLoader(IncludeLoader):
                def __init__(self, stream):
                    super().__init__(stream, path.parent)

            return yaml.load(f, Loader=FileSpecificIncludeLoader)

    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML file {path}: {e}. "
            f"Suggestion: Check YAML syntax and include paths."
        ) from e


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries with deep merging.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to merge on top of base

    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add new key
            result[key] = value

    return result


def resolve_inheritance(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve configuration inheritance using 'inherits' or 'extends' keys.

    Args:
        config_data: Configuration data that may contain inheritance

    Returns:
        Configuration with inheritance resolved

    Example:
        # base.yaml
        prompt_template: "Hello {{name}}"
        axes:
          name:
            type: categorical
            values: ["Alice", "Bob"]

        # child.yaml
        inherits: base.yaml
        axes:
          name:
            values: ["Alice", "Bob", "Charlie"]  # Override values
          age:  # Add new axis
            type: categorical
            values: ["young", "old"]
    """
    # Check for inheritance keywords
    inherit_key = None
    for key in ["inherits", "extends", "inherit_from"]:
        if key in config_data:
            inherit_key = key
            break

    if not inherit_key:
        return config_data

    inherit_path = config_data.pop(inherit_key)

    # Load the parent configuration
    try:
        parent_data = load_yaml_with_includes(inherit_path)
        if not isinstance(parent_data, dict):
            raise ValueError(
                f"Parent configuration must be a dictionary, got {type(parent_data).__name__}"
            )

        # Recursively resolve inheritance in parent
        parent_data = resolve_inheritance(parent_data)

        # Merge parent with current config
        return merge_configs(parent_data, config_data)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Parent configuration file not found: {inherit_path}. "
            f"Suggestion: Check the inheritance path and ensure the file exists."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error resolving inheritance from {inherit_path}: {e}"
        ) from e


def process_includes_and_inheritance(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Process a YAML file with both includes and inheritance support.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Processed configuration data with includes and inheritance resolved
    """
    # First, load with includes
    data = load_yaml_with_includes(file_path)

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML file must contain a dictionary at the root level. "
            f"Got {type(data).__name__} instead."
        )

    # Then, resolve inheritance
    return resolve_inheritance(data)
