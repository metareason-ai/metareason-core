"""LLM provider adapters for MetaReason."""

from .base import (
    AdapterError,
    AuthenticationError,
    CompletionRequest,
    CompletionResponse,
    LLMAdapter,
    Message,
    MessageRole,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    StreamChunk,
)
from .http_base import BaseHTTPAdapter, RateLimiter, RetryHandler
from .registry import (
    AdapterFactory,
    AdapterRegistry,
    create_adapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
)

__all__ = [
    # Base classes
    "LLMAdapter",
    "BaseHTTPAdapter",
    # Data models
    "Message",
    "MessageRole",
    "CompletionRequest",
    "CompletionResponse",
    "StreamChunk",
    # Errors
    "AdapterError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ProviderError",
    # Utilities
    "RateLimiter",
    "RetryHandler",
    # Registry
    "AdapterRegistry",
    "AdapterFactory",
    "register_adapter",
    "get_adapter_class",
    "list_adapters",
    "create_adapter",
]
