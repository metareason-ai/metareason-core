"""OpenAI adapter implementation."""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    CompletionRequest,
    CompletionResponse,
    Message,
    ModelNotFoundError,
    ProviderError,
    StreamChunk,
)
from .http_base import BaseHTTPAdapter, RateLimitConfig, RetryConfig

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseHTTPAdapter):
    """Adapter for OpenAI API."""

    SUPPORTED_MODELS = {
        # GPT-4 models
        "gpt-4-turbo-preview": {"context": 128000, "max_output": 4096},
        "gpt-4-turbo": {"context": 128000, "max_output": 4096},
        "gpt-4": {"context": 8192, "max_output": 4096},
        "gpt-4-32k": {"context": 32768, "max_output": 4096},
        "gpt-4-1106-preview": {"context": 128000, "max_output": 4096},
        "gpt-4-0125-preview": {"context": 128000, "max_output": 4096},
        # GPT-3.5 models
        "gpt-3.5-turbo": {"context": 16385, "max_output": 4096},
        "gpt-3.5-turbo-16k": {"context": 16385, "max_output": 4096},
        "gpt-3.5-turbo-1106": {"context": 16385, "max_output": 4096},
        "gpt-3.5-turbo-0125": {"context": 16385, "max_output": 4096},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        api_version: Optional[str] = None,
        default_model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        retry_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key
            organization_id: OpenAI organization ID
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

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Prepare headers
        headers = kwargs.pop("headers", {})
        if organization_id:
            headers["OpenAI-Organization"] = organization_id
        if api_version:
            headers["OpenAI-Version"] = api_version

        # Parse retry and rate limit configs
        retry_cfg = RetryConfig(**retry_config) if retry_config else RetryConfig()
        rate_limit_cfg = (
            RateLimitConfig(**rate_limit_config)
            if rate_limit_config
            else RateLimitConfig(requests_per_minute=3000)  # OpenAI default
        )

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            headers=headers,
            timeout=timeout,
            retry_config=retry_cfg,
            rate_limit_config=rate_limit_cfg,
            config=config,
        )

        self.default_model = default_model
        self.organization_id = organization_id

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for OpenAI API.

        Args:
            messages: List of messages

        Returns:
            Formatted messages
        """
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenAI API.

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

        # Prepare request payload
        payload = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "stream": False,
        }

        # Add optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.stop:
            payload["stop"] = request.stop

        # Make API request
        try:
            response = await self._request(
                "POST", "chat/completions", json_data=payload
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API request failed: {e}") from e

        # Extract response
        try:
            choice = response["choices"][0]
            return CompletionResponse(
                content=choice["message"]["content"],
                model=response["model"],
                finish_reason=choice.get("finish_reason"),
                usage=response.get("usage"),
                metadata={
                    "id": response.get("id"),
                    "created": response.get("created"),
                    "system_fingerprint": response.get("system_fingerprint"),
                },
            )
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Invalid OpenAI API response: {e}") from e

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion using OpenAI API.

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

        # Prepare request payload
        payload = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "stream": True,
        }

        # Add optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.stop:
            payload["stop"] = request.stop

        # Make streaming API request
        try:
            async for chunk_data in self._stream_request(
                "POST", "chat/completions", json_data=payload
            ):
                # Parse chunk
                try:
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        delta = choice.get("delta", {})

                        if "content" in delta:
                            yield StreamChunk(
                                content=delta["content"],
                                finish_reason=choice.get("finish_reason"),
                                metadata={
                                    "id": chunk_data.get("id"),
                                    "model": chunk_data.get("model"),
                                },
                            )
                except (KeyError, IndexError) as e:
                    logger.warning(f"Invalid stream chunk: {e}")
                    continue
        except Exception as e:
            raise ProviderError(f"OpenAI streaming request failed: {e}") from e

    async def list_models(self) -> List[str]:
        """List available OpenAI models.

        Returns:
            List of model IDs
        """
        try:
            response = await self._request("GET", "models")
            models = []

            for model_data in response.get("data", []):
                model_id = model_data.get("id")
                # Filter for chat models
                if model_id and (
                    model_id.startswith("gpt-") or model_id in self.SUPPORTED_MODELS
                ):
                    models.append(model_id)

            return sorted(models)
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            # Return known models as fallback
            return list(self.SUPPORTED_MODELS.keys())

    async def validate_model(self, model: str) -> bool:
        """Check if model is available.

        Args:
            model: Model ID

        Returns:
            True if model is available
        """
        # Quick check against known models
        if model in self.SUPPORTED_MODELS:
            return True

        # Check with API
        try:
            models = await self.list_models()
            return model in models
        except Exception:
            # Fallback to known models
            return model in self.SUPPORTED_MODELS

    async def estimate_cost(self, request: CompletionRequest) -> Optional[float]:
        """Estimate cost for completion request.

        Args:
            request: Completion request

        Returns:
            Estimated cost in USD
        """
        model = request.model or self.default_model

        # Cost per 1K tokens (approximate as of 2024)
        costs = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002},
        }

        if model not in costs:
            return None

        # Estimate token count (rough approximation)
        # 1 token ≈ 4 characters
        input_tokens = sum(len(msg.content) for msg in request.messages) / 4
        output_tokens = request.max_tokens or 1000  # Default estimate

        cost_info = costs[model]
        input_cost = (input_tokens / 1000) * cost_info["input"]
        output_cost = (output_tokens / 1000) * cost_info["output"]

        return input_cost + output_cost
