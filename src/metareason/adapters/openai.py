"""OpenAI adapter implementation."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .base import (
    AdapterError,
    AuthenticationError,
    CompletionRequest,
    CompletionResponse,
    LLMAdapter,
    Message,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    StreamChunk,
)

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
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
        batch_size: int = 20,
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
            batch_size: Maximum requests per batch
            config: Additional configuration
            **kwargs: Additional arguments
        """
        # Get API key from environment if not provided
        if not api_key:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                    "variable or pass api_key parameter."
                )

        super().__init__(config)

        self.api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url
        self.api_version = api_version
        self.default_model = default_model
        self.timeout = timeout
        self.batch_size = batch_size
        self._client: Optional[AsyncOpenAI] = None
        self._request_count = 0
        self._error_count = 0

    async def _initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._client is None:
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "timeout": self.timeout,
            }

            if self.organization_id:
                client_kwargs["organization"] = self.organization_id
            if self.api_version:
                client_kwargs["default_headers"] = {"OpenAI-Version": self.api_version}

            self._client = AsyncOpenAI(**client_kwargs)

    async def _cleanup(self) -> None:
        """Cleanup OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None

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
        if not self._client:
            await self.initialize()

        model = request.model or self.default_model

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Prepare request parameters
        completion_kwargs = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "stream": False,
        }

        # Add optional parameters
        if request.max_tokens:
            completion_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            completion_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            completion_kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            completion_kwargs["stop"] = request.stop

        # Make API request
        try:
            self._request_count += 1
            response: ChatCompletion = await self._client.chat.completions.create(
                **completion_kwargs
            )
        except openai.AuthenticationError as e:
            self._error_count += 1
            raise AuthenticationError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            self._error_count += 1
            retry_after = getattr(e, "retry_after", None)
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {e}", retry_after=retry_after
            ) from e
        except openai.NotFoundError as e:
            self._error_count += 1
            raise ModelNotFoundError(f"OpenAI model not found: {e}") from e
        except openai.APIError as e:
            self._error_count += 1
            raise ProviderError(f"OpenAI API error: {e}") from e
        except Exception as e:
            self._error_count += 1
            raise AdapterError(f"OpenAI request failed: {e}") from e

        # Extract response
        try:
            choice = response.choices[0]
            usage_dict = None
            if response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return CompletionResponse(
                content=choice.message.content or "",
                model=response.model,
                finish_reason=choice.finish_reason,
                usage=usage_dict,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                },
            )
        except (AttributeError, IndexError) as e:
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
        if not self._client:
            await self.initialize()

        model = request.model or self.default_model

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Prepare request parameters
        completion_kwargs = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "stream": True,
        }

        # Add optional parameters
        if request.max_tokens:
            completion_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            completion_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            completion_kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            completion_kwargs["stop"] = request.stop

        # Make streaming API request
        try:
            self._request_count += 1
            stream = await self._client.chat.completions.create(**completion_kwargs)

            async for chunk in stream:
                try:
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        delta = choice.delta

                        if delta.content:
                            yield StreamChunk(
                                content=delta.content,
                                finish_reason=choice.finish_reason,
                                metadata={
                                    "id": chunk.id,
                                    "model": chunk.model,
                                },
                            )
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Invalid stream chunk: {e}")
                    continue

        except openai.AuthenticationError as e:
            self._error_count += 1
            raise AuthenticationError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            self._error_count += 1
            retry_after = getattr(e, "retry_after", None)
            raise RateLimitError(
                f"OpenAI rate limit exceeded: {e}", retry_after=retry_after
            ) from e
        except openai.NotFoundError as e:
            self._error_count += 1
            raise ModelNotFoundError(f"OpenAI model not found: {e}") from e
        except openai.APIError as e:
            self._error_count += 1
            raise ProviderError(f"OpenAI API error: {e}") from e
        except Exception as e:
            self._error_count += 1
            raise AdapterError(f"OpenAI streaming request failed: {e}") from e

    async def list_models(self) -> List[str]:
        """List available OpenAI models.

        Returns:
            List of model IDs
        """
        if not self._client:
            await self.initialize()

        try:
            response = await self._client.models.list()
            models = []

            for model in response.data:
                model_id = model.id
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

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this adapter.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0
            ),
        }

    async def batch_complete(
        self, requests: List[CompletionRequest]
    ) -> List[Union[CompletionResponse, Exception]]:
        """Process multiple completion requests using OpenAI Batch API.

        For smaller batches (< batch_size), uses concurrent individual requests.
        For larger batches, uses OpenAI's Batch API for cost efficiency.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        if len(requests) <= self.batch_size:
            # Use concurrent processing for smaller batches
            return await super().batch_complete(requests)

        # Use OpenAI Batch API for larger batches
        return await self._batch_complete_with_api(requests)

    async def _batch_complete_with_api(
        self, requests: List[CompletionRequest]
    ) -> List[Union[CompletionResponse, Exception]]:
        """Process requests using OpenAI Batch API.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        if not self._client:
            await self.initialize()

        results = []

        # Process in chunks of batch_size
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]

            try:
                # For now, fall back to concurrent processing
                # OpenAI Batch API implementation would require file upload/download
                # and polling, which is complex for this integration
                tasks = [self._safe_complete(req) for req in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(chunk_results)

            except Exception as e:
                # If batch processing fails, add exceptions for all requests in chunk
                chunk_exceptions = [
                    AdapterError(f"Batch processing failed: {e}") for _ in chunk
                ]
                results.extend(chunk_exceptions)

        return results

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
