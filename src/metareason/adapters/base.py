"""Base adapter classes for LLM providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Role of a message in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Single message in conversation."""

    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    """Request for LLM completion."""

    messages: List[Message]
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StreamChunk:
    """Single chunk in streaming response."""

    content: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdapterError(Exception):
    """Base exception for adapter errors."""

    pass


class RateLimitError(AdapterError):
    """Rate limit exceeded error."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(AdapterError):
    """Authentication failed error."""

    pass


class ModelNotFoundError(AdapterError):
    """Requested model not found."""

    pass


class ProviderError(AdapterError):
    """Provider-specific error."""

    pass


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter with configuration.

        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize adapter resources.

        Should be called before first use.
        """
        if not self._initialized:
            await self._initialize()
            self._initialized = True

    @abstractmethod
    async def _initialize(self) -> None:
        """Provider-specific initialization."""
        pass

    async def cleanup(self) -> None:
        """Cleanup adapter resources.

        Should be called when adapter is no longer needed.
        """
        if self._initialized:
            await self._cleanup()
            self._initialized = False

    @abstractmethod
    async def _cleanup(self) -> None:
        """Provider-specific cleanup."""
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion for given request.

        Args:
            request: Completion request parameters

        Returns:
            Completion response

        Raises:
            AdapterError: On any adapter-specific error
        """
        pass

    @abstractmethod
    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion for given request.

        Args:
            request: Completion request parameters

        Yields:
            Stream chunks as they arrive

        Raises:
            AdapterError: On any adapter-specific error
        """
        pass

    async def batch_complete(
        self, requests: List[CompletionRequest]
    ) -> List[Union[CompletionResponse, Exception]]:
        """Process multiple completion requests.

        Default implementation processes requests concurrently.
        Providers can override for more efficient batch processing.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        tasks = [self._safe_complete(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Wrapper for safe completion execution.

        Args:
            request: Completion request

        Returns:
            Completion response
        """
        try:
            return await self.complete(request)
        except Exception as e:
            logger.error(f"Error in completion: {e}")
            raise

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    async def validate_model(self, model: str) -> bool:
        """Check if model is available.

        Args:
            model: Model identifier

        Returns:
            True if model is available
        """
        pass

    async def estimate_cost(self, request: CompletionRequest) -> Optional[float]:
        """Estimate cost for completion request.

        Args:
            request: Completion request

        Returns:
            Estimated cost in USD or None if not available
        """
        return None

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter.

        Returns:
            Dictionary with usage statistics
        """
        return {}

    def __str__(self) -> str:
        """String representation of adapter."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Developer representation of adapter."""
        return f"{self.__class__.__name__}(config={self.config})"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
