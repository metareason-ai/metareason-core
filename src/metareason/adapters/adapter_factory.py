"""Adapter factory for creating LLM adapter instances.

This module provides a factory function for creating adapter instances based on
configuration. It handles the instantiation of different adapter types and ensures
proper configuration is passed to each adapter.

Functions:
    get_adapter: Factory function that creates and returns configured adapter instances.
"""

import logging

from .adapter_base import AdapterBase, AdapterException
from .google import GoogleAdapter
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter

logger = logging.getLogger(__name__)

# Registry of available adapters
ADAPTER_REGISTRY = {
    "ollama": OllamaAdapter,
    "google": GoogleAdapter,
    "openai": OpenAIAdapter,
}


def get_adapter(name: str, **kwargs) -> AdapterBase:
    """Create and return an adapter instance based on name.

    This factory function creates the appropriate adapter instance based on the
    provided name. Configuration can be passed via kwargs.

    Args:
        name: Name of the adapter to create ('ollama', 'google', 'openai', etc.).
        **kwargs: Configuration parameters to pass to the adapter.

    Returns:
        Configured adapter instance ready for use.

    Raises:
        AdapterException: If the adapter name is not recognized or initialization fails.

    Example:
        >>> adapter = get_adapter('google', api_key='xxx', project_id='my-project')
        >>> adapter = get_adapter('ollama')
    """
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY.keys())
        raise AdapterException(
            f"Unknown adapter '{name}'. Available adapters: {available}"
        )

    # Log adapter creation (sanitize sensitive data)
    if kwargs:
        safe_config = {
            k: "***" if "key" in k.lower() or "token" in k.lower() else v
            for k, v in kwargs.items()
        }
        logger.debug(f"Creating {name} adapter with config: {safe_config}")
    else:
        logger.debug(f"Creating {name} adapter with no config")

    # Instantiate adapter
    adapter_class = ADAPTER_REGISTRY[name]
    try:
        return adapter_class(**kwargs)
    except Exception as e:
        raise AdapterException(f"Failed to initialize {name} adapter: {str(e)}") from e
