"""Configuration models for LLM adapters."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class AdapterType(str, Enum):
    """Supported adapter types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts"
    )
    initial_delay: float = Field(
        default=1.0, gt=0, le=60, description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0,
        gt=0,
        le=300,
        description="Maximum delay between retries in seconds",
    )
    exponential_base: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Base for exponential backoff"
    )
    jitter: bool = Field(default=True, description="Add random jitter to retry delays")

    @model_validator(mode="after")
    def validate_delays(self) -> "RetryConfig":
        """Ensure max_delay >= initial_delay."""
        if self.max_delay < self.initial_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= "
                f"initial_delay ({self.initial_delay})"
            )
        return self


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    requests_per_second: Optional[float] = Field(
        default=None, gt=0, le=1000, description="Maximum requests per second"
    )
    requests_per_minute: Optional[float] = Field(
        default=None, gt=0, le=60000, description="Maximum requests per minute"
    )
    concurrent_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent requests"
    )
    burst_size: int = Field(
        default=20, ge=1, le=1000, description="Token bucket burst size"
    )

    @model_validator(mode="after")
    def validate_rate_limits(self) -> "RateLimitConfig":
        """Ensure only one rate limit type is specified."""
        if self.requests_per_second and self.requests_per_minute:
            raise ValueError(
                "Specify either requests_per_second or requests_per_minute, not both"
            )
        return self


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str = Field(..., description="Model identifier (e.g., 'gpt-4', 'claude-3')")
    version: Optional[str] = Field(None, description="Model version")
    context_window: Optional[int] = Field(
        None, gt=0, description="Maximum context window size"
    )
    max_output_tokens: Optional[int] = Field(
        None, gt=0, description="Maximum output tokens"
    )
    cost_per_1k_input: Optional[float] = Field(
        None, ge=0, description="Cost per 1000 input tokens in USD"
    )
    cost_per_1k_output: Optional[float] = Field(
        None, ge=0, description="Cost per 1000 output tokens in USD"
    )
    supports_streaming: bool = Field(
        default=True, description="Whether model supports streaming responses"
    )
    supports_functions: bool = Field(
        default=False, description="Whether model supports function calling"
    )
    supports_vision: bool = Field(
        default=False, description="Whether model supports image inputs"
    )


class BaseAdapterConfig(BaseModel):
    """Base configuration for all adapters."""

    type: AdapterType = Field(..., description="Adapter type")
    api_key: Optional[str] = Field(
        None, description="API key (can also be set via environment variable)"
    )
    api_key_env: Optional[str] = Field(
        None, description="Environment variable containing API key"
    )
    base_url: Optional[str] = Field(None, description="Base URL for API endpoint")
    timeout: float = Field(
        default=30.0, gt=0, le=300, description="Request timeout in seconds"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    default_model: Optional[str] = Field(
        None, description="Default model to use if not specified in request"
    )
    available_models: Optional[List[ModelConfig]] = Field(
        None, description="List of available models with their configurations"
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="Additional HTTP headers"
    )

    @model_validator(mode="after")
    def validate_api_key(self) -> "BaseAdapterConfig":
        """Ensure API key is provided either directly or via environment."""
        if not self.api_key and not self.api_key_env:
            # Some adapters might not require API keys
            if self.type not in [
                AdapterType.CUSTOM,
                AdapterType.HUGGINGFACE,
                AdapterType.OLLAMA,
            ]:
                raise ValueError(
                    f"Either 'api_key' or 'api_key_env' must be provided for {self.type}"
                )
        return self


class OpenAIConfig(BaseAdapterConfig):
    """Configuration for OpenAI adapter."""

    type: Literal[AdapterType.OPENAI] = AdapterType.OPENAI
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    organization_id: Optional[str] = Field(None, description="OpenAI organization ID")
    api_version: Optional[str] = Field(None, description="API version to use")
    default_model: str = Field(default="gpt-3.5-turbo", description="Default model")
    batch_size: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum requests per batch for batch processing",
    )


class AnthropicConfig(BaseAdapterConfig):
    """Configuration for Anthropic adapter."""

    type: Literal[AdapterType.ANTHROPIC] = AdapterType.ANTHROPIC
    base_url: str = Field(
        default="https://api.anthropic.com/v1", description="Anthropic API base URL"
    )
    api_version: str = Field(default="2023-06-01", description="API version")
    default_model: str = Field(
        default="claude-3-sonnet-20240229", description="Default model"
    )


class AzureOpenAIConfig(BaseAdapterConfig):
    """Configuration for Azure OpenAI adapter."""

    type: Literal[AdapterType.AZURE_OPENAI] = AdapterType.AZURE_OPENAI
    azure_endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    azure_deployment: str = Field(..., description="Azure deployment name")
    api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )

    @model_validator(mode="after")
    def set_base_url(self) -> "AzureOpenAIConfig":
        """Set base_url from azure_endpoint."""
        if not self.base_url:
            self.base_url = self.azure_endpoint
        return self


class HuggingFaceConfig(BaseAdapterConfig):
    """Configuration for HuggingFace adapter."""

    type: Literal[AdapterType.HUGGINGFACE] = AdapterType.HUGGINGFACE
    model_id: str = Field(..., description="HuggingFace model ID")
    inference_endpoint: Optional[str] = Field(
        None, description="Custom inference endpoint URL"
    )
    use_inference_api: bool = Field(
        default=True, description="Use HuggingFace Inference API"
    )
    device: Optional[str] = Field(
        None, description="Device to run model on (for local inference)"
    )


class GoogleConfig(BaseAdapterConfig):
    """Configuration for Google Gemini adapter."""

    type: Literal[AdapterType.GOOGLE] = AdapterType.GOOGLE
    base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1",
        description="Google GenAI API base URL",
    )
    api_version: str = Field(default="v1", description="API version")
    default_model: str = Field(
        default="gemini-2.0-flash-001", description="Default model"
    )
    use_vertex_ai: bool = Field(
        default=False,
        description="Use Vertex AI endpoint instead of Gemini Developer API",
    )
    project_id: Optional[str] = Field(
        None, description="Google Cloud project ID (required for Vertex AI)"
    )
    location: Optional[str] = Field(
        default="us-central1", description="Google Cloud location (for Vertex AI)"
    )
    batch_size: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum requests per batch for batch processing",
    )

    @model_validator(mode="after")
    def validate_vertex_ai_config(self) -> "GoogleConfig":
        """Validate Vertex AI configuration."""
        if self.use_vertex_ai and not self.project_id:
            raise ValueError("project_id is required when use_vertex_ai=True")
        return self


class OllamaConfig(BaseAdapterConfig):
    """Configuration for Ollama adapter."""

    type: Literal[AdapterType.OLLAMA] = AdapterType.OLLAMA
    base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    default_model: str = Field(default="llama3", description="Default model")
    pull_missing_models: bool = Field(
        default=False, description="Automatically pull models if not available"
    )
    model_timeout: float = Field(
        default=120.0, gt=0, description="Model inference timeout in seconds"
    )


class CustomAdapterConfig(BaseAdapterConfig):
    """Configuration for custom adapters."""

    type: Literal[AdapterType.CUSTOM] = AdapterType.CUSTOM
    adapter_class: str = Field(
        ..., description="Fully qualified class name for custom adapter"
    )
    custom_params: Optional[Dict[str, Any]] = Field(
        None, description="Custom parameters for adapter"
    )


# Union type for all adapter configurations
AdapterConfigType = Union[
    OpenAIConfig,
    AnthropicConfig,
    GoogleConfig,
    AzureOpenAIConfig,
    HuggingFaceConfig,
    OllamaConfig,
    CustomAdapterConfig,
]


class AdaptersConfig(BaseModel):
    """Configuration for multiple adapters."""

    default_adapter: str = Field(..., description="Name of default adapter to use")
    adapters: Dict[str, AdapterConfigType] = Field(
        ..., description="Named adapter configurations"
    )

    @model_validator(mode="after")
    def validate_default_adapter(self) -> "AdaptersConfig":
        """Ensure default adapter exists in adapters dict."""
        if self.default_adapter not in self.adapters:
            available = list(self.adapters.keys())
            raise ValueError(
                f"Default adapter '{self.default_adapter}' not found in adapters. "
                f"Available adapters: {available}"
            )
        return self

    def get_adapter_config(self, name: Optional[str] = None) -> AdapterConfigType:
        """Get adapter configuration by name.

        Args:
            name: Adapter name, uses default if not specified

        Returns:
            Adapter configuration

        Raises:
            KeyError: If adapter not found
        """
        adapter_name = name or self.default_adapter
        if adapter_name not in self.adapters:
            raise KeyError(f"Adapter '{adapter_name}' not found")
        return self.adapters[adapter_name]
