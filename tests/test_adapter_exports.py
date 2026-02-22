"""Test that all adapter classes are exported from the adapters package."""


def test_all_adapters_importable_from_package():
    """All adapter classes should be importable from metareason.adapters."""
    from metareason.adapters import (
        AdapterBase,
        AdapterException,
        AdapterRequest,
        AdapterResponse,
        AnthropicAdapter,
        GoogleAdapter,
        OllamaAdapter,
        OpenAIAdapter,
        get_adapter,
    )

    assert AnthropicAdapter is not None
    assert OpenAIAdapter is not None
    assert GoogleAdapter is not None
    assert OllamaAdapter is not None
    assert AdapterBase is not None
    assert AdapterException is not None
    assert AdapterRequest is not None
    assert AdapterResponse is not None
    assert get_adapter is not None


def test_all_list_contains_all_adapters():
    """The __all__ list should contain all adapter classes."""
    import metareason.adapters as adapters_pkg

    expected = [
        "AdapterBase",
        "AdapterException",
        "AdapterRequest",
        "AdapterResponse",
        "AnthropicAdapter",
        "GoogleAdapter",
        "OllamaAdapter",
        "OpenAIAdapter",
        "get_adapter",
    ]
    for name in expected:
        assert name in adapters_pkg.__all__, f"{name} missing from __all__"
