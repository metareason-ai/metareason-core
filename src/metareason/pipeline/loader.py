from pathlib import Path
from typing import Any, Union

import yaml

from ..config import CalibrateConfig, CalibrateMultiConfig, SpecConfig


def _resolve_file_reference(value: str, base_dir: Path) -> str:
    """Resolve a 'file:path' reference to its file contents.

    If value starts with 'file:', read the referenced file and return its contents.
    Paths are resolved relative to base_dir. Path traversal outside base_dir
    is rejected for safety.
    Otherwise, return the value as-is.
    """
    if value.startswith("file:"):
        raw_path = value[5:]
        file_path = (base_dir / raw_path).resolve()
        base_resolved = base_dir.resolve()
        if not file_path.is_relative_to(base_resolved):
            raise ValueError(
                f"file: reference '{raw_path}' resolves outside the spec directory"
            )
        with open(file_path, "r") as f:
            return f.read().strip()
    return value


def _resolve_file_references(data: Any, base_dir: Path) -> Any:
    """Recursively resolve all 'file:' prefixed strings in a parsed YAML structure.

    Walks dicts, lists, and strings. Any string starting with 'file:' is replaced
    with the contents of the referenced file (relative to base_dir).
    """
    if isinstance(data, dict):
        return {k: _resolve_file_references(v, base_dir) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_file_references(item, base_dir) for item in data]
    if isinstance(data, str):
        return _resolve_file_reference(data, base_dir)
    return data


def load_spec(file_path: Union[str, Path]) -> SpecConfig:
    """Load and validate YAML specification file.

    Resolves 'file:' prefixes in any string field to load content from
    external files (paths relative to the spec file).
    """
    path = Path(file_path)
    base_dir = path.parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data = _resolve_file_references(data, base_dir)
    return SpecConfig(**data)


def load_calibrate_spec(file_path: Union[str, Path]) -> CalibrateConfig:
    """Load and validate a calibration YAML specification file.

    Resolves 'file:' prefixes in any string field to load content from
    external files (paths relative to the spec file).
    """
    path = Path(file_path)
    base_dir = path.parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data = _resolve_file_references(data, base_dir)
    return CalibrateConfig(**data)


def load_calibrate_multi_spec(
    file_path: Union[str, Path],
) -> CalibrateMultiConfig:
    """Load and validate a multi-judge calibration YAML specification file.

    Resolves 'file:' prefixes in any string field to load content from
    external files (paths relative to the spec file).
    """
    path = Path(file_path)
    base_dir = path.parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data = _resolve_file_references(data, base_dir)
    return CalibrateMultiConfig(**data)
