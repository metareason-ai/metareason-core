"""Verify metareason imports after removing unused dependencies."""

import pathlib


def test_metareason_imports_successfully():
    """Verify that the main metareason package can be imported."""
    import metareason

    assert metareason is not None


def test_core_submodules_import():
    """Verify that core submodules import without error."""
    from metareason.config import models
    from metareason.pipeline import renderer
    from metareason.sampling import lhs_sampler

    assert models is not None
    assert lhs_sampler is not None
    assert renderer is not None


def test_no_direct_imports_of_removed_packages():
    """Verify no source file imports removed unused dependencies."""
    removed_packages = [
        "seaborn",
        "tqdm",
        "aiohttp",
        "structlog",
        "pydantic_settings",
        "pandas",
    ]

    src_dir = pathlib.Path(__file__).parent.parent / "src" / "metareason"
    violations = []

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text()
        for pkg in removed_packages:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if f"import {pkg}" in stripped or f"from {pkg}" in stripped:
                    violations.append(
                        f"{py_file.relative_to(src_dir)}: {stripped} "
                        f"(uses removed dependency '{pkg}')"
                    )

    assert (
        not violations
    ), "Found direct imports of removed dependencies:\n" + "\n".join(violations)
