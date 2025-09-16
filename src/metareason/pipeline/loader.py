from pathlib import Path
from typing import Union

import yaml

from ..config import SpecConfig


def load_spec(file_path: Union[str, Path]) -> SpecConfig:
    """Load and validate YAML specification file."""
    path = Path(file_path)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return SpecConfig(**data)
