"""Test version information."""

import metareason


def test_version() -> None:
    """Test that version is accessible."""
    assert hasattr(metareason, "__version__")
    assert isinstance(metareason.__version__, str)
    assert metareason.__version__ == "0.1.0"
