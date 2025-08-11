"""Registry and factory for LLM adapters."""

import importlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import LLMAdapter

from ..config.adapters import (
    AdapterConfigType,
    AdapterType,
    AnthropicConfig,
    AzureOpenAIConfig,
    BaseAdapterConfig,
    CustomAdapterConfig,
    GoogleConfig,
    HuggingFaceConfig,
    OllamaConfig,
    OpenAIConfig,
)

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for adapter implementations."""

    def __init__(self):
        """Initialize adapter registry."""
        self._adapters: Dict[str, Type[LLMAdapter]] = {}
        self._register_builtin_adapters()

    def _register_builtin_adapters(self) -> None:
        """Register built-in adapter implementations."""
        # Register adapters that are available
        try:
            from .openai import OpenAIAdapter

            self.register(AdapterType.OPENAI.value, OpenAIAdapter)
        except Exception as e:
            logger.debug(f"Could not register OpenAI adapter: {e}")

        try:
            from .anthropic import AnthropicAdapter

            self.register(AdapterType.ANTHROPIC.value, AnthropicAdapter)
        except Exception as e:
            logger.debug(f"Could not register Anthropic adapter: {e}")

        try:
            from .ollama import OllamaAdapter

            self.register(AdapterType.OLLAMA.value, OllamaAdapter)
        except Exception as e:
            logger.debug(f"Could not register Ollama adapter: {e}")

        try:
            from .google import GoogleAdapter

            self.register(AdapterType.GOOGLE.value, GoogleAdapter)
        except Exception as e:
            logger.debug(f"Could not register Google adapter: {e}")

        # Note: Azure and HuggingFace adapters not implemented yet
        # try:
        #     from .azure import AzureOpenAIAdapter
        #     self.register(AdapterType.AZURE_OPENAI.value, AzureOpenAIAdapter)
        # except Exception as e:
        #     logger.debug(f"Could not register Azure OpenAI adapter: {e}")

        # try:
        #     from .huggingface import HuggingFaceAdapter
        #     self.register(AdapterType.HUGGINGFACE.value, HuggingFaceAdapter)
        # except Exception as e:
        #     logger.debug(f"Could not register HuggingFace adapter: {e}")

    def _register_lazy(self, name: str, import_path: str) -> None:
        """Register adapter with lazy loading.

        Args:
            name: Adapter name
            import_path: Import path in format 'module:ClassName'
        """
        self._adapters[name] = import_path  # type: ignore

    def register(self, name: str, adapter_class) -> None:
        """Register an adapter implementation.

        Args:
            name: Adapter name
            adapter_class: Adapter class
        """
        # Import here to avoid circular imports
        from .base import LLMAdapter

        if not issubclass(adapter_class, LLMAdapter):
            raise TypeError(f"{adapter_class} must inherit from LLMAdapter")

        self._adapters[name] = adapter_class
        logger.info(f"Registered adapter: {name}")

    def get(self, name: str):
        """Get adapter class by name.

        Args:
            name: Adapter name

        Returns:
            Adapter class or None if not found
        """
        adapter = self._adapters.get(name)

        if adapter is None:
            return None

        # Handle lazy loading
        if isinstance(adapter, str):
            adapter = self._load_adapter_class(adapter)
            if adapter:
                self._adapters[name] = adapter

        return adapter

    def _load_adapter_class(self, import_path: str):
        """Load adapter class from import path.

        Args:
            import_path: Import path in format 'module:ClassName'

        Returns:
            Adapter class or None if loading fails
        """
        try:
            module_path, class_name = import_path.split(":")
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)

            # Import LLMAdapter here to avoid circular imports
            from .base import LLMAdapter

            if not issubclass(adapter_class, LLMAdapter):
                raise TypeError(f"{adapter_class} must inherit from LLMAdapter")

            return adapter_class
        except Exception as e:
            logger.debug(f"Failed to load adapter from {import_path}: {e}")
            return None

    def list_available(self) -> list[str]:
        """List available adapter names.

        Returns:
            List of registered adapter names
        """
        return list(self._adapters.keys())

    def is_available(self, name: str) -> bool:
        """Check if adapter is available.

        Args:
            name: Adapter name

        Returns:
            True if adapter is registered
        """
        return name in self._adapters


# Global registry instance
_registry = AdapterRegistry()


def register_adapter(name: str, adapter_class) -> None:
    """Register an adapter with the global registry.

    Args:
        name: Adapter name
        adapter_class: Adapter class
    """
    _registry.register(name, adapter_class)


def get_adapter_class(name: str):
    """Get adapter class from global registry.

    Args:
        name: Adapter name

    Returns:
        Adapter class or None if not found
    """
    return _registry.get(name)


def list_adapters() -> list[str]:
    """List available adapters in global registry.

    Returns:
        List of adapter names
    """
    return _registry.list_available()


class AdapterFactory:
    """Factory for creating adapter instances."""

    @staticmethod
    def create(config: AdapterConfigType):
        """Create adapter instance from configuration.

        Args:
            config: Adapter configuration

        Returns:
            Adapter instance

        Raises:
            ValueError: If adapter type not found or configuration invalid
        """
        if isinstance(config, CustomAdapterConfig):
            return AdapterFactory._create_custom_adapter(config)

        # Get adapter class from registry
        adapter_class = get_adapter_class(config.type)
        if not adapter_class:
            raise ValueError(
                f"Adapter type '{config.type}' not found. "
                f"Available adapters: {list_adapters()}"
            )

        # Prepare configuration for adapter
        adapter_config = AdapterFactory._prepare_config(config)

        # Create adapter instance
        try:
            return adapter_class(config=adapter_config)
        except Exception as e:
            raise ValueError(f"Failed to create adapter: {e}") from e

    @staticmethod
    def _create_custom_adapter(config: CustomAdapterConfig):
        """Create custom adapter from configuration.

        Args:
            config: Custom adapter configuration

        Returns:
            Custom adapter instance

        Raises:
            ValueError: If custom adapter cannot be created
        """
        try:
            # Parse the adapter class path
            module_path, class_name = config.adapter_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)

            # Import LLMAdapter here to avoid circular imports
            from .base import LLMAdapter

            # Check if it's actually a class and if it inherits from LLMAdapter
            if not isinstance(adapter_class, type):
                raise TypeError(f"{adapter_class} is not a class")

            if not issubclass(adapter_class, LLMAdapter):
                raise TypeError(f"{adapter_class} must inherit from LLMAdapter")

            # Prepare configuration
            adapter_config = {
                **(config.custom_params or {}),
                "api_key": config.api_key,
                "base_url": config.base_url,
                "timeout": config.timeout,
                "retry_config": config.retry.model_dump(),
                "rate_limit_config": config.rate_limit.model_dump(),
            }

            return adapter_class(config=adapter_config)
        except Exception as e:
            raise ValueError(
                f"Failed to create custom adapter '{config.adapter_class}': {e}"
            ) from e

    @staticmethod
    def _prepare_config(config: BaseAdapterConfig) -> Dict[str, Any]:
        """Prepare configuration dictionary for adapter.

        Args:
            config: Adapter configuration

        Returns:
            Configuration dictionary
        """
        import os

        # Get API key from environment if specified
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key environment variable '{config.api_key_env}' not set"
                )

        # Build configuration dictionary
        adapter_config = {
            "api_key": api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "retry_config": config.retry.model_dump(),
            "rate_limit_config": config.rate_limit.model_dump(),
            "default_model": config.default_model,
            "headers": config.headers or {},
        }

        # Add type-specific configuration
        if isinstance(config, OpenAIConfig):
            if config.organization_id:
                adapter_config["organization_id"] = config.organization_id
            if config.api_version:
                adapter_config["api_version"] = config.api_version

        elif isinstance(config, AnthropicConfig):
            adapter_config["api_version"] = config.api_version

        elif isinstance(config, GoogleConfig):
            adapter_config["api_version"] = config.api_version
            adapter_config["use_vertex_ai"] = config.use_vertex_ai
            adapter_config["project_id"] = config.project_id
            adapter_config["location"] = config.location
            adapter_config["batch_size"] = config.batch_size

        elif isinstance(config, AzureOpenAIConfig):
            adapter_config["azure_endpoint"] = config.azure_endpoint
            adapter_config["azure_deployment"] = config.azure_deployment
            adapter_config["api_version"] = config.api_version

        elif isinstance(config, HuggingFaceConfig):
            adapter_config["model_id"] = config.model_id
            adapter_config["inference_endpoint"] = config.inference_endpoint
            adapter_config["use_inference_api"] = config.use_inference_api
            adapter_config["device"] = config.device

        elif isinstance(config, OllamaConfig):
            adapter_config["pull_missing_models"] = config.pull_missing_models
            adapter_config["model_timeout"] = config.model_timeout

        return adapter_config


@lru_cache(maxsize=32)
def create_adapter(config_json: str):
    """Create adapter from JSON configuration (cached).

    Args:
        config_json: JSON string of adapter configuration

    Returns:
        Adapter instance
    """
    import json

    from pydantic import TypeAdapter

    config_dict = json.loads(config_json)
    config = TypeAdapter(AdapterConfigType).validate_python(config_dict)
    return AdapterFactory.create(config)
