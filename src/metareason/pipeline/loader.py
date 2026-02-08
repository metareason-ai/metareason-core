from pathlib import Path
from typing import Union

import yaml

from ..config import CalibrateConfig, SpecConfig


def load_spec(file_path: Union[str, Path]) -> SpecConfig:
    """Load and validate YAML specification file."""
    path = Path(file_path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return SpecConfig(**data)


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


def load_calibrate_spec(file_path: Union[str, Path]) -> CalibrateConfig:
    """Load and validate a calibration YAML specification file.

    Resolves 'file:' prefixes in prompt and response fields to load
    content from external files (paths relative to the spec file).
    """
    path = Path(file_path)
    base_dir = path.parent

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "prompt" in data:
        data["prompt"] = _resolve_file_reference(data["prompt"], base_dir)
    if "response" in data:
        data["response"] = _resolve_file_reference(data["response"], base_dir)

    return CalibrateConfig(**data)
