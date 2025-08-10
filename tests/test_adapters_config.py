"""Tests for adapter configuration models."""

import pytest
from pydantic import ValidationError

from metareason.config.adapters import (
    AdaptersConfig,
    AdapterType,
    AnthropicConfig,
    AzureOpenAIConfig,
    BaseAdapterConfig,
    CustomAdapterConfig,
    HuggingFaceConfig,
    ModelConfig,
    OpenAIConfig,
    RateLimitConfig,
    RetryConfig,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False

    def test_invalid_max_retries(self):
        """Test invalid max_retries values."""
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValidationError):
            RetryConfig(max_retries=11)

    def test_invalid_delays(self):
        """Test invalid delay values."""
        with pytest.raises(ValidationError):
            RetryConfig(initial_delay=0.0)

        with pytest.raises(ValidationError):
            RetryConfig(max_delay=0.0)

        with pytest.raises(ValidationError):
            RetryConfig(initial_delay=30.0, max_delay=10.0)

    def test_invalid_exponential_base(self):
        """Test invalid exponential base values."""
        with pytest.raises(ValidationError):
            RetryConfig(exponential_base=0.5)

        with pytest.raises(ValidationError):
            RetryConfig(exponential_base=11.0)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_second is None
        assert config.requests_per_minute is None
        assert config.concurrent_requests == 10
        assert config.burst_size == 20

    def test_requests_per_second(self):
        """Test rate limiting by requests per second."""
        config = RateLimitConfig(requests_per_second=5.0)
        assert config.requests_per_second == 5.0
        assert config.requests_per_minute is None

    def test_requests_per_minute(self):
        """Test rate limiting by requests per minute."""
        config = RateLimitConfig(requests_per_minute=300.0)
        assert config.requests_per_minute == 300.0
        assert config.requests_per_second is None

    def test_concurrent_and_burst(self):
        """Test concurrent requests and burst size."""
        config = RateLimitConfig(concurrent_requests=5, burst_size=10)
        assert config.concurrent_requests == 5
        assert config.burst_size == 10

    def test_invalid_rate_limits(self):
        """Test invalid rate limit values."""
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_second=0.0)

        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_minute=0.0)

        with pytest.raises(ValidationError):
            RateLimitConfig(concurrent_requests=0)

        with pytest.raises(ValidationError):
            RateLimitConfig(burst_size=0)

    def test_both_rate_limits_specified(self):
        """Test error when both rate limits are specified."""
        with pytest.raises(
            ValidationError,
            match="Specify either requests_per_second or requests_per_minute",
        ):
            RateLimitConfig(requests_per_second=10.0, requests_per_minute=600.0)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_basic_model_config(self):
        """Test basic model configuration."""
        config = ModelConfig(name="gpt-4")
        assert config.name == "gpt-4"
        assert config.version is None
        assert config.context_window is None
        assert config.supports_streaming is True
        assert config.supports_functions is False
        assert config.supports_vision is False

    def test_full_model_config(self):
        """Test full model configuration."""
        config = ModelConfig(
            name="gpt-4-turbo",
            version="2024-01-01",
            context_window=128000,
            max_output_tokens=4096,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_streaming=True,
            supports_functions=True,
            supports_vision=True,
        )
        assert config.name == "gpt-4-turbo"
        assert config.version == "2024-01-01"
        assert config.context_window == 128000
        assert config.max_output_tokens == 4096
        assert config.cost_per_1k_input == 0.01
        assert config.cost_per_1k_output == 0.03
        assert config.supports_streaming is True
        assert config.supports_functions is True
        assert config.supports_vision is True

    def test_invalid_token_counts(self):
        """Test invalid token count values."""
        with pytest.raises(ValidationError):
            ModelConfig(name="test", context_window=0)

        with pytest.raises(ValidationError):
            ModelConfig(name="test", max_output_tokens=0)

    def test_invalid_costs(self):
        """Test invalid cost values."""
        with pytest.raises(ValidationError):
            ModelConfig(name="test", cost_per_1k_input=-0.01)

        with pytest.raises(ValidationError):
            ModelConfig(name="test", cost_per_1k_output=-0.01)


class TestBaseAdapterConfig:
    """Tests for BaseAdapterConfig."""

    def test_minimal_config(self):
        """Test minimal adapter configuration."""
        config = BaseAdapterConfig(type=AdapterType.CUSTOM, api_key="test-key")
        assert config.type == AdapterType.CUSTOM
        assert config.api_key == "test-key"
        assert config.timeout == 30.0
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)

    def test_config_with_env_key(self):
        """Test configuration with environment API key."""
        config = BaseAdapterConfig(type=AdapterType.CUSTOM, api_key_env="MY_API_KEY")
        assert config.api_key_env == "MY_API_KEY"

    def test_missing_api_key_for_required_provider(self):
        """Test error when API key missing for required provider."""
        with pytest.raises(
            ValidationError, match="Either 'api_key' or 'api_key_env' must be provided"
        ):
            BaseAdapterConfig(type=AdapterType.OPENAI)

    def test_optional_api_key_for_custom_provider(self):
        """Test that API key is optional for custom providers."""
        # Should not raise error
        config = BaseAdapterConfig(type=AdapterType.CUSTOM)
        assert config.api_key is None
        assert config.api_key_env is None

    def test_full_config(self):
        """Test full adapter configuration."""
        retry_config = RetryConfig(max_retries=5)
        rate_limit_config = RateLimitConfig(requests_per_second=10.0)
        model_config = ModelConfig(name="test-model")

        config = BaseAdapterConfig(
            type=AdapterType.OPENAI,
            api_key="test-key",
            base_url="https://api.test.com",
            timeout=60.0,
            retry=retry_config,
            rate_limit=rate_limit_config,
            default_model="gpt-4",
            available_models=[model_config],
            headers={"Custom-Header": "value"},
        )

        assert config.base_url == "https://api.test.com"
        assert config.timeout == 60.0
        assert config.retry.max_retries == 5
        assert config.rate_limit.requests_per_second == 10.0
        assert config.default_model == "gpt-4"
        assert len(config.available_models) == 1
        assert config.headers["Custom-Header"] == "value"


class TestSpecificAdapterConfigs:
    """Tests for specific adapter configurations."""

    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = OpenAIConfig(
            api_key="sk-test", organization_id="org-123", api_version="v1"
        )
        assert config.type == AdapterType.OPENAI
        assert config.base_url == "https://api.openai.com/v1"
        assert config.default_model == "gpt-3.5-turbo"
        assert config.organization_id == "org-123"
        assert config.api_version == "v1"

    def test_anthropic_config(self):
        """Test Anthropic configuration."""
        config = AnthropicConfig(api_key="sk-ant-test")
        assert config.type == AdapterType.ANTHROPIC
        assert config.base_url == "https://api.anthropic.com/v1"
        assert config.api_version == "2023-06-01"
        assert config.default_model == "claude-3-sonnet-20240229"

    def test_azure_openai_config(self):
        """Test Azure OpenAI configuration."""
        config = AzureOpenAIConfig(
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4-deployment",
        )
        assert config.type == AdapterType.AZURE_OPENAI
        assert config.azure_endpoint == "https://test.openai.azure.com"
        assert config.azure_deployment == "gpt-4-deployment"
        assert config.api_version == "2024-02-01"
        assert config.base_url == "https://test.openai.azure.com"  # Set by validator

    def test_huggingface_config(self):
        """Test HuggingFace configuration."""
        config = HuggingFaceConfig(
            model_id="microsoft/DialoGPT-medium",
            inference_endpoint="https://api-inference.huggingface.co",
            use_inference_api=True,
            device="cuda:0",
        )
        assert config.type == AdapterType.HUGGINGFACE
        assert config.model_id == "microsoft/DialoGPT-medium"
        assert config.inference_endpoint == "https://api-inference.huggingface.co"
        assert config.use_inference_api is True
        assert config.device == "cuda:0"

    def test_custom_adapter_config(self):
        """Test custom adapter configuration."""
        config = CustomAdapterConfig(
            adapter_class="my.custom.CustomAdapter",
            api_key="test-key",
            custom_params={"param1": "value1", "param2": 123},
        )
        assert config.type == AdapterType.CUSTOM
        assert config.adapter_class == "my.custom.CustomAdapter"
        assert config.custom_params["param1"] == "value1"
        assert config.custom_params["param2"] == 123


class TestAdaptersConfig:
    """Tests for AdaptersConfig."""

    def test_single_adapter_config(self):
        """Test configuration with single adapter."""
        openai_config = OpenAIConfig(api_key="test-key")

        config = AdaptersConfig(
            default_adapter="openai", adapters={"openai": openai_config}
        )

        assert config.default_adapter == "openai"
        assert len(config.adapters) == 1
        assert config.adapters["openai"] == openai_config

    def test_multiple_adapters_config(self):
        """Test configuration with multiple adapters."""
        openai_config = OpenAIConfig(api_key="openai-key")
        anthropic_config = AnthropicConfig(api_key="anthropic-key")

        config = AdaptersConfig(
            default_adapter="openai",
            adapters={"openai": openai_config, "anthropic": anthropic_config},
        )

        assert config.default_adapter == "openai"
        assert len(config.adapters) == 2
        assert isinstance(config.adapters["openai"], OpenAIConfig)
        assert isinstance(config.adapters["anthropic"], AnthropicConfig)

    def test_invalid_default_adapter(self):
        """Test error when default adapter not in adapters dict."""
        openai_config = OpenAIConfig(api_key="test-key")

        with pytest.raises(ValidationError, match="Default adapter .* not found"):
            AdaptersConfig(
                default_adapter="nonexistent", adapters={"openai": openai_config}
            )

    def test_get_adapter_config(self):
        """Test getting adapter configuration."""
        openai_config = OpenAIConfig(api_key="openai-key")
        anthropic_config = AnthropicConfig(api_key="anthropic-key")

        config = AdaptersConfig(
            default_adapter="openai",
            adapters={"openai": openai_config, "anthropic": anthropic_config},
        )

        # Get default adapter
        default = config.get_adapter_config()
        assert default == openai_config

        # Get specific adapter
        anthropic = config.get_adapter_config("anthropic")
        assert anthropic == anthropic_config

        # Get non-existent adapter
        with pytest.raises(KeyError):
            config.get_adapter_config("nonexistent")


class TestConfigSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = OpenAIConfig(
            api_key="test-key", organization_id="org-123", default_model="gpt-4"
        )

        config_dict = config.model_dump()

        assert config_dict["type"] == "openai"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["organization_id"] == "org-123"
        assert config_dict["default_model"] == "gpt-4"
        assert "retry" in config_dict
        assert "rate_limit" in config_dict

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "type": "anthropic",
            "api_key": "test-key",
            "api_version": "2023-06-01",
            "default_model": "claude-3-haiku-20240307",
        }

        config = AnthropicConfig(**config_dict)

        assert config.type == AdapterType.ANTHROPIC
        assert config.api_key == "test-key"
        assert config.api_version == "2023-06-01"
        assert config.default_model == "claude-3-haiku-20240307"

    def test_adapters_config_serialization(self):
        """Test serializing complete adapters configuration."""
        openai_config = OpenAIConfig(api_key="openai-key")
        anthropic_config = AnthropicConfig(api_key="anthropic-key")

        adapters_config = AdaptersConfig(
            default_adapter="openai",
            adapters={"openai": openai_config, "anthropic": anthropic_config},
        )

        # Serialize to dict
        config_dict = adapters_config.model_dump()

        assert config_dict["default_adapter"] == "openai"
        assert len(config_dict["adapters"]) == 2
        assert config_dict["adapters"]["openai"]["type"] == "openai"
        assert config_dict["adapters"]["anthropic"]["type"] == "anthropic"

        # Deserialize from dict
        restored_config = AdaptersConfig(**config_dict)

        assert restored_config.default_adapter == "openai"
        assert len(restored_config.adapters) == 2
        assert isinstance(restored_config.adapters["openai"], OpenAIConfig)
        assert isinstance(restored_config.adapters["anthropic"], AnthropicConfig)
