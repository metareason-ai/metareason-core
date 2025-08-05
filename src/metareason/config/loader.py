"""YAML configuration loader for MetaReason."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from .axes import CategoricalAxis, ContinuousAxis
from .cache import get_global_cache, is_caching_enabled
from .environment import EnvironmentSubstitutionError, substitute_environment_variables
from .includes import process_includes_and_inheritance
from .models import EvaluationConfig
from .oracles import (
    CustomOracle,
    EmbeddingSimilarityOracle,
    LLMJudgeOracle,
    OracleConfig,
    StatisticalCalibrationOracle,
)


def load_yaml_config(
    file_path: Union[str, Path], 
    use_cache: bool = True,
    enable_includes: bool = True,
    enable_env_substitution: bool = True,
    env_strict: bool = False,
    search_paths: Optional[List[Union[str, Path]]] = None
) -> EvaluationConfig:
    """Load and validate a MetaReason YAML configuration file.
    
    Args:
        file_path: Path to the YAML configuration file
        use_cache: Whether to use configuration caching
        enable_includes: Whether to process !include tags
        enable_env_substitution: Whether to substitute environment variables
        env_strict: Whether environment variable substitution is strict
        search_paths: Additional paths to search for configuration files
        
    Returns:
        Validated EvaluationConfig object
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        ValidationError: If the configuration doesn't match the schema
        EnvironmentSubstitutionError: If environment variable substitution fails
    """
    path = _resolve_config_path(file_path, search_paths)
    
    # Check cache first
    if use_cache and is_caching_enabled():
        cache = get_global_cache()
        cached_config = cache.get(path)
        if cached_config is not None:
            return cached_config
    
    # Validate file extension
    if not path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            f"Invalid file extension: {path.suffix}. "
            f"Suggestion: Use .yaml or .yml extension for configuration files."
        )
    
    # Check file permissions
    try:
        if not os.access(path, os.R_OK):
            raise PermissionError(
                f"Permission denied reading configuration file: {path}. "
                f"Suggestion: Check file permissions with 'ls -la {path}'"
            )
    except PermissionError:
        # Re-raise PermissionError as-is
        raise
    except OSError as e:
        # For other OS errors (like file not found), check if it's a permission issue
        if "permission denied" in str(e).lower() or "access is denied" in str(e).lower():
            raise PermissionError(
                f"Permission denied reading configuration file: {path}. "
                f"Suggestion: Check file permissions with 'ls -la {path}'"
            ) from e
        else:
            raise OSError(
                f"Error accessing configuration file {path}: {e}. "
                f"Suggestion: Ensure the file exists and is readable."
            ) from e
    
    try:
        # Load YAML with includes and inheritance support
        if enable_includes:
            try:
                data = process_includes_and_inheritance(path)
            except Exception as e:
                # Fall back to standard YAML loading if includes fail
                _log_include_fallback(path, e)
                data = _load_standard_yaml(path)
        else:
            data = _load_standard_yaml(path)
        
        # Substitute environment variables
        if enable_env_substitution:
            try:
                data = substitute_environment_variables(data, strict=env_strict)
            except EnvironmentSubstitutionError as e:
                raise EnvironmentSubstitutionError(
                    f"Environment variable substitution failed in {path}: {e}"
                ) from e
        
        # Process the configuration data
        config = _process_config_data(data, path)
        
        # Cache the result
        if use_cache and is_caching_enabled():
            cache = get_global_cache()
            cache.set(path, config)
        
        return config
        
    except (FileNotFoundError, PermissionError, EnvironmentSubstitutionError):
        raise
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML file {path}: {e}. "
            f"Suggestion: Check YAML syntax using a validator. "
            f"Error details: {_extract_yaml_error_details(e)}"
        ) from e
    except ValidationError as e:
        raise ValueError(
            f"Configuration validation failed for {path}:\n{e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading configuration from {path}: {e}"
        ) from e


def load_yaml_configs(directory: Union[str, Path]) -> Dict[str, EvaluationConfig]:
    """Load all YAML configuration files from a directory.
    
    Args:
        directory: Path to directory containing YAML files
        
    Returns:
        Dictionary mapping file names to EvaluationConfig objects
        
    Raises:
        ValueError: If the directory doesn't exist or contains no YAML files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise ValueError(
            f"Directory not found: {dir_path}. "
            f"Suggestion: Create the directory or check the path."
        )
    
    if not dir_path.is_dir():
        raise ValueError(
            f"Path is not a directory: {dir_path}. "
            f"Suggestion: Provide a directory path, not a file path."
        )
    
    yaml_files = list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
    
    if not yaml_files:
        raise ValueError(
            f"No YAML files found in directory: {dir_path}. "
            f"Suggestion: Add .yaml or .yml configuration files to the directory."
        )
    
    configs = {}
    errors = []
    
    for file_path in yaml_files:
        try:
            config = load_yaml_config(file_path)
            configs[file_path.stem] = config
        except Exception as e:
            errors.append(f"{file_path.name}: {str(e)}")
    
    if errors and not configs:
        raise ValueError(
            f"Failed to load any configuration files. Errors:\n" + 
            "\n".join(errors)
        )
    
    return configs


def validate_yaml_string(yaml_content: str) -> EvaluationConfig:
    """Validate a YAML configuration from a string.
    
    Args:
        yaml_content: YAML content as a string
        
    Returns:
        Validated EvaluationConfig object
        
    Raises:
        yaml.YAMLError: If the YAML is invalid
        ValidationError: If the configuration doesn't match the schema
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML: {e}. "
            f"Suggestion: Check YAML syntax using a validator."
        )
    
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML must contain a dictionary at the root level. "
            f"Got {type(data).__name__} instead."
        )
    
    # Handle schema -> axes alias mapping
    if "schema" in data and "axes" not in data:
        data["axes"] = data.pop("schema")
    
    # Convert axis dictionaries to proper axis objects
    if "axes" in data:
        converted_axes = {}
        for axis_name, axis_data in data["axes"].items():
            if axis_data["type"] == "categorical":
                converted_axes[axis_name] = CategoricalAxis(**axis_data)
            else:
                # It's a continuous axis
                converted_axes[axis_name] = ContinuousAxis(**axis_data)
        data["axes"] = converted_axes
    
    # Convert oracle dictionaries to proper oracle objects
    if "oracles" in data:
        oracle_data = data["oracles"]
        oracle_config_data = {}
        
        if "accuracy" in oracle_data:
            oracle_config_data["accuracy"] = EmbeddingSimilarityOracle(**oracle_data["accuracy"])
        
        if "explainability" in oracle_data:
            oracle_config_data["explainability"] = LLMJudgeOracle(**oracle_data["explainability"])
        
        if "confidence_calibration" in oracle_data:
            oracle_config_data["confidence_calibration"] = StatisticalCalibrationOracle(**oracle_data["confidence_calibration"])
        
        if "custom_oracles" in oracle_data:
            # Handle custom oracles if present
            custom_oracles = {}
            for name, custom_data in oracle_data["custom_oracles"].items():
                custom_oracles[name] = CustomOracle(**custom_data)
            oracle_config_data["custom_oracles"] = custom_oracles
        
        # Handle other oracle types that might be custom oracles
        for key, value in oracle_data.items():
            if key not in ["accuracy", "explainability", "confidence_calibration", "custom_oracles"]:
                if isinstance(value, dict) and value.get("type") == "custom":
                    if "custom_oracles" not in oracle_config_data:
                        oracle_config_data["custom_oracles"] = {}
                    oracle_config_data["custom_oracles"][key] = CustomOracle(**value)
        
        data["oracles"] = OracleConfig(**oracle_config_data)
    
    return EvaluationConfig(**data)


def _resolve_config_path(
    file_path: Union[str, Path], 
    search_paths: Optional[List[Union[str, Path]]] = None
) -> Path:
    """Resolve configuration file path with search path support."""
    path = Path(file_path)
    
    # If absolute path or exists as-is, return it
    if path.is_absolute() or path.exists():
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}. "
                f"Suggestion: Check the file path and ensure the file exists."
            )
        return path.resolve()
    
    # Search in provided search paths
    if search_paths:
        for search_path in search_paths:
            candidate = Path(search_path) / file_path
            if candidate.exists():
                return candidate.resolve()
    
    # Search in default locations
    default_search_paths = [
        Path.cwd(),
        Path.cwd() / "config",
        Path.cwd() / "configs",
        Path.home() / ".metareason",
        Path("/etc/metareason") if os.name != "nt" else Path("C:/ProgramData/metareason"),
    ]
    
    for search_path in default_search_paths:
        if search_path.exists():
            candidate = search_path / file_path
            if candidate.exists():
                return candidate.resolve()
    
    # File not found in any location
    searched_locations = (search_paths or []) + [str(p) for p in default_search_paths]
    raise FileNotFoundError(
        f"Configuration file '{file_path}' not found. "
        f"Searched in: {', '.join(str(loc) for loc in searched_locations)}. "
        f"Suggestion: Check the file path or place the file in one of the searched directories."
    )


def _load_standard_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file using standard yaml.safe_load."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML file: {e}. "
            f"Suggestion: Check YAML syntax using a validator."
        ) from e
    
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML file must contain a dictionary at the root level. "
            f"Got {type(data).__name__} instead."
        )
    
    return data


def _process_config_data(data: Dict[str, Any], file_path: Path) -> EvaluationConfig:
    """Process configuration data and convert to EvaluationConfig."""
    # Handle schema -> axes alias mapping
    if "schema" in data and "axes" not in data:
        data["axes"] = data.pop("schema")
    
    # Convert axis dictionaries to proper axis objects
    if "axes" in data:
        converted_axes = {}
        for axis_name, axis_data in data["axes"].items():
            if axis_data["type"] == "categorical":
                converted_axes[axis_name] = CategoricalAxis(**axis_data)
            else:
                # It's a continuous axis
                converted_axes[axis_name] = ContinuousAxis(**axis_data)
        data["axes"] = converted_axes
    
    # Convert oracle dictionaries to proper oracle objects
    if "oracles" in data:
        oracle_data = data["oracles"]
        oracle_config_data = {}
        
        if "accuracy" in oracle_data:
            oracle_config_data["accuracy"] = EmbeddingSimilarityOracle(**oracle_data["accuracy"])
        
        if "explainability" in oracle_data:
            oracle_config_data["explainability"] = LLMJudgeOracle(**oracle_data["explainability"])
        
        if "confidence_calibration" in oracle_data:
            oracle_config_data["confidence_calibration"] = StatisticalCalibrationOracle(**oracle_data["confidence_calibration"])
        
        if "custom_oracles" in oracle_data:
            # Handle custom oracles if present
            custom_oracles = {}
            for name, custom_data in oracle_data["custom_oracles"].items():
                custom_oracles[name] = CustomOracle(**custom_data)
            oracle_config_data["custom_oracles"] = custom_oracles
        
        # Handle other oracle types that might be custom oracles
        for key, value in oracle_data.items():
            if key not in ["accuracy", "explainability", "confidence_calibration", "custom_oracles"]:
                if isinstance(value, dict) and value.get("type") == "custom":
                    if "custom_oracles" not in oracle_config_data:
                        oracle_config_data["custom_oracles"] = {}
                    oracle_config_data["custom_oracles"][key] = CustomOracle(**value)
        
        data["oracles"] = OracleConfig(**oracle_config_data)
    
    try:
        return EvaluationConfig(**data)
    except ValidationError as e:
        # Re-raise with more helpful context
        raise ValueError(
            f"Configuration validation failed for {file_path}:\n{e}"
        ) from e


def _log_include_fallback(file_path: Path, error: Exception) -> None:
    """Log warning about falling back from include processing."""
    # In a production system, this would use proper logging
    # For now, we'll just ignore the error silently
    pass


def _extract_yaml_error_details(error: yaml.YAMLError) -> str:
    """Extract useful details from YAML error for better error messages."""
    error_str = str(error)
    
    # Extract line number if available
    if hasattr(error, 'problem_mark') and error.problem_mark:
        line = error.problem_mark.line + 1
        column = error.problem_mark.column + 1
        return f"Line {line}, Column {column}: {error_str}"
    
    return error_str