"""Anthropic adapter implementation."""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    ModelNotFoundError,
    ProviderError,
    StreamChunk,
)
from .http_base import BaseHTTPAdapter, RateLimitConfig, RetryConfig

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseHTTPAdapter):
    """Adapter for Anthropic API."""

    SUPPORTED_MODELS = {
        # Claude 3 models
        "claude-3-opus-20240229": {"context": 200000, "max_output": 4096},
        "claude-3-sonnet-20240229": {"context": 200000, "max_output": 4096},
        "claude-3-haiku-20240307": {"context": 200000, "max_output": 4096},
        # Claude 2 models
        "claude-2.1": {"context": 200000, "max_output": 4096},
        "claude-2.0": {"context": 100000, "max_output": 4096},
        # Claude Instant
        "claude-instant-1.2": {"context": 100000, "max_output": 4096},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        api_version: str = "2023-06-01",
        default_model: str = "claude-3-sonnet-20240229",
        timeout: float = 30.0,
        retry_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for API
            api_version: API version
            default_model: Default model to use
            timeout: Request timeout
            retry_config: Retry configuration
            rate_limit_config: Rate limit configuration
            config: Additional configuration
            **kwargs: Additional arguments
        """
        # Get API key from environment if not provided
        if not api_key:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
                    "variable or pass api_key parameter."
                )

        # Prepare headers
        headers = kwargs.pop("headers", {})
        headers.update(
            {
                "x-api-key": api_key,  # Anthropic uses x-api-key header
                "anthropic-version": api_version,
            }
        )

        # Parse retry and rate limit configs
        retry_cfg = RetryConfig(**retry_config) if retry_config else RetryConfig()
        rate_limit_cfg = (
            RateLimitConfig(**rate_limit_config)
            if rate_limit_config
            else RateLimitConfig(requests_per_minute=1000)  # Anthropic default
        )

        super().__init__(
            base_url=base_url,
            api_key=None,  # API key is in headers for Anthropic
            headers=headers,
            timeout=timeout,
            retry_config=retry_cfg,
            rate_limit_config=rate_limit_cfg,
            config=config,
        )

        self.default_model = default_model
        self.api_version = api_version

    def _format_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict]]:
        """Format messages for Anthropic API.

        Anthropic has a different format where system message is separate.

        Args:
            messages: List of messages

        Returns:
            Tuple of (system_message, messages)
        """
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Combine multiple system messages if present
                if system_message:
                    system_message += "\n\n" + msg.content
                else:
                    system_message = msg.content
            else:
                # Anthropic uses "user" and "assistant" roles
                role = msg.role.value
                formatted_messages.append(
                    {
                        "role": role,
                        "content": msg.content,
                    }
                )

        return system_message, formatted_messages

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Anthropic API.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            AdapterError: On API errors
        """
        model = request.model or self.default_model

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Format messages
        system_message, messages = self._format_messages(request.messages)

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,  # Required by Anthropic
            "temperature": request.temperature,
            "stream": False,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add optional parameters
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        # Make API request
        try:
            response = await self._request("POST", "messages", json_data=payload)
        except Exception as e:
            raise ProviderError(f"Anthropic API request failed: {e}") from e

        # Extract response
        try:
            content = ""
            for content_block in response.get("content", []):
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")

            return CompletionResponse(
                content=content,
                model=response["model"],
                finish_reason=response.get("stop_reason"),
                usage={
                    "prompt_tokens": response.get("usage", {}).get("input_tokens"),
                    "completion_tokens": response.get("usage", {}).get("output_tokens"),
                    "total_tokens": (
                        response.get("usage", {}).get("input_tokens", 0)
                        + response.get("usage", {}).get("output_tokens", 0)
                    ),
                },
                metadata={
                    "id": response.get("id"),
                    "type": response.get("type"),
                    "role": response.get("role"),
                },
            )
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Invalid Anthropic API response: {e}") from e

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion using Anthropic API.

        Args:
            request: Completion request

        Yields:
            Stream chunks

        Raises:
            AdapterError: On API errors
        """
        model = request.model or self.default_model

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Format messages
        system_message, messages = self._format_messages(request.messages)

        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,  # Required by Anthropic
            "temperature": request.temperature,
            "stream": True,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add optional parameters
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        # Make streaming API request
        try:
            async for chunk_data in self._stream_request(
                "POST", "messages", json_data=payload
            ):
                # Parse chunk based on event type
                try:
                    event_type = chunk_data.get("type")

                    if event_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(
                                content=delta.get("text", ""),
                                metadata={
                                    "index": chunk_data.get("index"),
                                },
                            )

                    elif event_type == "message_stop":
                        # Final chunk
                        yield StreamChunk(
                            content="",
                            finish_reason="stop",
                            metadata={"type": "message_stop"},
                        )

                except (KeyError, IndexError) as e:
                    logger.warning(f"Invalid stream chunk: {e}")
                    continue
        except Exception as e:
            raise ProviderError(f"Anthropic streaming request failed: {e}") from e

    async def list_models(self) -> List[str]:
        """List available Anthropic models.

        Note: Anthropic doesn't have a models endpoint, so we return known models.

        Returns:
            List of model IDs
        """
        return list(self.SUPPORTED_MODELS.keys())

    async def validate_model(self, model: str) -> bool:
        """Check if model is available.

        Args:
            model: Model ID

        Returns:
            True if model is available
        """
        return model in self.SUPPORTED_MODELS

    async def estimate_cost(self, request: CompletionRequest) -> Optional[float]:
        """Estimate cost for completion request.

        Args:
            request: Completion request

        Returns:
            Estimated cost in USD
        """
        model = request.model or self.default_model

        # Cost per 1M tokens (approximate as of 2024)
        costs = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-2.0": {"input": 8.0, "output": 24.0},
            "claude-instant-1.2": {"input": 0.8, "output": 2.4},
        }

        if model not in costs:
            return None

        # Estimate token count (rough approximation)
        # 1 token ≈ 4 characters
        input_tokens = sum(len(msg.content) for msg in request.messages) / 4
        output_tokens = request.max_tokens or 1000  # Default estimate

        cost_info = costs[model]
        input_cost = (input_tokens / 1_000_000) * cost_info["input"]
        output_cost = (output_tokens / 1_000_000) * cost_info["output"]

        return input_cost + output_cost
