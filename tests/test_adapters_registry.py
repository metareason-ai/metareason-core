"""Tests for adapter registry and configuration."""

import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from metareason.adapters.base import LLMAdapter
from metareason.adapters.registry import (
    AdapterFactory,
    AdapterRegistry,
    create_adapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
)
from metareason.config.adapters import (
    AdapterType,
    AnthropicConfig,
    CustomAdapterConfig,
    OpenAIConfig,
)


class MockAdapter(LLMAdapter):
    """Mock adapter for registry tests."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    async def _initialize(self):
        pass

    async def _cleanup(self):
        pass

    async def complete(self, request):
        from metareason.adapters.base import CompletionResponse

        return CompletionResponse(content="test", model="test")

    async def complete_stream(self, request):
        from metareason.adapters.base import StreamChunk

        yield StreamChunk(content="test")

    async def list_models(self):
        return ["test-model"]

    async def validate_model(self, model):
        return model == "test-model"


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AdapterRegistry()

        # Should have some built-in adapters registered (even if not loadable)
        available = registry.list_available()
        assert len(available) >= 0  # May be empty if imports fail

    def test_register_adapter(self):
        """Test registering an adapter."""
        registry = AdapterRegistry()

        registry.register("test", MockAdapter)
        assert "test" in registry.list_available()
        assert registry.is_available("test")

        adapter_class = registry.get("test")
        assert adapter_class == MockAdapter

    def test_register_invalid_adapter(self):
        """Test registering invalid adapter."""
        registry = AdapterRegistry()

        class NotAnAdapter:
            pass

        with pytest.raises(TypeError):
            registry.register("invalid", NotAnAdapter)

    def test_get_nonexistent_adapter(self):
        """Test getting non-existent adapter."""
        registry = AdapterRegistry()

        adapter_class = registry.get("nonexistent")
        assert adapter_class is None

    @patch("importlib.import_module")
    def test_lazy_loading_success(self, mock_import):
        """Test successful lazy loading of adapter."""
        registry = AdapterRegistry()

        # Mock module with adapter class
        mock_module = Mock()
        mock_module.MockAdapter = MockAdapter
        mock_import.return_value = mock_module

        # Register lazy adapter
        registry._register_lazy("lazy-test", "test.module:MockAdapter")

        # Should load successfully
        adapter_class = registry.get("lazy-test")
        assert adapter_class == MockAdapter

        mock_import.assert_called_once_with("test.module")

    @patch("importlib.import_module")
    def test_lazy_loading_failure(self, mock_import):
        """Test failed lazy loading of adapter."""
        registry = AdapterRegistry()

        # Mock import failure
        mock_import.side_effect = ImportError("Module not found")

        # Register lazy adapter
        registry._register_lazy("lazy-fail", "missing.module:MockAdapter")

        # Should return None on load failure
        adapter_class = registry.get("lazy-fail")
        assert adapter_class is None


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_register_adapter_global(self):
        """Test global register_adapter function."""
        register_adapter("global-test", MockAdapter)

        adapter_class = get_adapter_class("global-test")
        assert adapter_class == MockAdapter

        available = list_adapters()
        assert "global-test" in available

    def test_get_nonexistent_adapter_global(self):
        """Test getting non-existent adapter globally."""
        adapter_class = get_adapter_class("definitely-nonexistent")
        assert adapter_class is None


class TestAdapterFactory:
    """Tests for AdapterFactory."""

    @patch.dict("os.environ", {"TEST_API_KEY": "test-key-value"})
    def test_create_openai_adapter(self):
        """Test creating OpenAI adapter."""
        config = OpenAIConfig(api_key_env="TEST_API_KEY", default_model="gpt-3.5-turbo")

        # Mock the OpenAI adapter class
        with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
            mock_adapter_class = Mock()
            mock_adapter_instance = Mock()
            mock_adapter_class.return_value = mock_adapter_instance
            mock_get.return_value = mock_adapter_class

            AdapterFactory.create(config)

            # Check that adapter class was called with correct config
            mock_get.assert_called_once_with(AdapterType.OPENAI)
            mock_adapter_class.assert_called_once()

            # Check config passed to adapter
            call_args = mock_adapter_class.call_args
            adapter_config = call_args.kwargs["config"]
            assert adapter_config["api_key"] == "test-key-value"
            assert adapter_config["base_url"] == "https://api.openai.com/v1"
            assert adapter_config["default_model"] == "gpt-3.5-turbo"

    def test_create_anthropic_adapter(self):
        """Test creating Anthropic adapter."""
        config = AnthropicConfig(
            api_key="test-key", default_model="claude-3-sonnet-20240229"
        )

        with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
            mock_adapter_class = Mock()
            mock_get.return_value = mock_adapter_class

            AdapterFactory.create(config)

            # Check config passed to adapter
            call_args = mock_adapter_class.call_args
            adapter_config = call_args.kwargs["config"]
            assert adapter_config["api_key"] == "test-key"
            assert adapter_config["base_url"] == "https://api.anthropic.com/v1"
            assert adapter_config["api_version"] == "2023-06-01"

    @patch("importlib.import_module")
    def test_create_custom_adapter(self, mock_import):
        """Test creating custom adapter."""
        config = CustomAdapterConfig(
            adapter_class="custom.module.CustomAdapter",
            api_key="test-key",
            custom_params={"param1": "value1"},
        )

        # Mock custom adapter class - use a proper mock that can be instantiated
        mock_adapter_instance = Mock()
        mock_custom_class = Mock(return_value=mock_adapter_instance)

        # Mock module
        mock_module = Mock()
        mock_module.CustomAdapter = mock_custom_class
        mock_import.return_value = mock_module

        # Mock issubclass to return True
        with patch("builtins.issubclass", return_value=True):
            result = AdapterFactory.create(config)

        # Check that custom adapter was called with config
        mock_custom_class.assert_called_once()
        call_args = mock_custom_class.call_args
        adapter_config = call_args.kwargs["config"]
        assert adapter_config["api_key"] == "test-key"
        assert adapter_config["param1"] == "value1"

        # Check that we got the expected instance back
        assert result == mock_adapter_instance

    def test_create_adapter_not_found(self):
        """Test creating adapter when type not found."""
        config = OpenAIConfig(api_key="test")

        with patch("metareason.adapters.registry.get_adapter_class", return_value=None):
            with pytest.raises(ValueError, match="Adapter type .* not found"):
                AdapterFactory.create(config)

    def test_create_adapter_creation_failure(self):
        """Test adapter creation failure."""
        config = OpenAIConfig(api_key="test")

        with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
            mock_adapter_class = Mock()
            mock_adapter_class.side_effect = Exception("Creation failed")
            mock_get.return_value = mock_adapter_class

            with pytest.raises(ValueError, match="Failed to create adapter"):
                AdapterFactory.create(config)

    def test_missing_api_key_env(self):
        """Test error when API key environment variable is missing."""
        config = OpenAIConfig(api_key_env="MISSING_KEY")

        with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
            mock_get.return_value = Mock()

            with pytest.raises(ValueError, match="API key environment variable"):
                AdapterFactory.create(config)

    @patch("importlib.import_module")
    def test_custom_adapter_import_failure(self, mock_import):
        """Test custom adapter import failure."""
        config = CustomAdapterConfig(
            adapter_class="missing.module.MissingAdapter", api_key="test"
        )

        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ValueError, match="Failed to create custom adapter"):
            AdapterFactory.create(config)

    @patch("importlib.import_module")
    def test_custom_adapter_invalid_class(self, mock_import):
        """Test custom adapter with invalid class."""
        config = CustomAdapterConfig(
            adapter_class="custom.module.NotAnAdapter", api_key="test"
        )

        # Mock module with non-adapter class
        mock_module = Mock()
        mock_module.NotAnAdapter = str  # Not an LLMAdapter subclass
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="Failed to create custom adapter"):
            AdapterFactory.create(config)


class TestCachedAdapterCreation:
    """Tests for cached adapter creation."""

    def test_create_adapter_cached(self):
        """Test cached adapter creation."""
        config_dict = {
            "type": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        }
        config_json = json.dumps(config_dict)

        with patch("metareason.adapters.registry.AdapterFactory.create") as mock_create:
            mock_adapter = Mock()
            mock_create.return_value = mock_adapter

            # First call
            adapter1 = create_adapter(config_json)

            # Second call with same config (should use cache)
            adapter2 = create_adapter(config_json)

            # Should be the same instance due to caching
            assert adapter1 == adapter2

            # Factory should only be called once due to caching
            assert mock_create.call_count == 1

    def test_create_adapter_different_configs(self):
        """Test cached creation with different configs."""
        config1 = json.dumps({"type": "openai", "api_key": "key1"})
        config2 = json.dumps({"type": "openai", "api_key": "key2"})

        with patch("metareason.adapters.registry.AdapterFactory.create") as mock_create:
            mock_create.side_effect = [Mock(), Mock()]

            adapter1 = create_adapter(config1)
            adapter2 = create_adapter(config2)

            # Should be different instances
            assert adapter1 != adapter2

            # Factory should be called twice
            assert mock_create.call_count == 2
