"""Anthropic adapter implementation."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import anthropic
from anthropic import AsyncAnthropic

from .base import (
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
from .schema_utils import create_anthropic_schema_prompt

logger = logging.getLogger(__name__)


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic API using official Anthropic client."""

    SUPPORTED_MODELS = {
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022": {"context": 200000, "max_output": 8192},
        "claude-3-5-sonnet-20240620": {"context": 200000, "max_output": 8192},
        "claude-3-5-haiku-20241022": {"context": 200000, "max_output": 8192},
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
        max_retries: int = 3,
        batch_size: int = 10,
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
            max_retries: Maximum retry attempts
            batch_size: Maximum requests per batch
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

        super().__init__(config)

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version

        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client: Optional[AsyncAnthropic] = None

        # Usage statistics tracking
        self._request_count = 0
        self._error_count = 0

    async def _initialize(self) -> None:
        """Initialize Anthropic client."""
        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers={"anthropic-version": self.api_version},
            )

    async def _cleanup(self) -> None:
        """Cleanup Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None

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

        if not self._client:
            await self.initialize()

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Format messages
        system_message, formatted_messages = self._format_messages(request.messages)

        # Prepare request parameters
        completion_kwargs = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": request.max_tokens or 1024,  # Required by Anthropic
            "temperature": request.temperature,
        }

        # Add system message if present, with schema enhancement
        final_system_message = system_message
        if request.json_schema_data:
            try:
                schema_prompt = create_anthropic_schema_prompt(request.json_schema_data)
                if final_system_message:
                    final_system_message = f"{final_system_message}\n\n{schema_prompt}"
                else:
                    final_system_message = schema_prompt
                logger.debug(
                    f"Enhanced system message with JSON schema for Anthropic model {model}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create schema prompt for Anthropic, proceeding without schema: {e}"
                )

        if final_system_message:
            completion_kwargs["system"] = final_system_message

        # Add optional parameters
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p
        if request.stop:
            completion_kwargs["stop_sequences"] = request.stop

        # Make API request
        try:
            self._request_count += 1
            response = await self._client.messages.create(**completion_kwargs)
        except anthropic.AuthenticationError as e:
            self._error_count += 1
            raise AuthenticationError(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            self._error_count += 1
            retry_after = getattr(e, "retry_after", None)
            raise RateLimitError(
                f"Anthropic rate limit exceeded: {e}", retry_after=retry_after
            ) from e
        except anthropic.NotFoundError as e:
            self._error_count += 1
            raise ModelNotFoundError(f"Anthropic model not found: {e}") from e
        except anthropic.APIError as e:
            self._error_count += 1
            raise ProviderError(f"Anthropic API error: {e}") from e
        except Exception as e:
            self._error_count += 1
            raise ProviderError(f"Anthropic request failed: {e}") from e

        # Extract response
        try:
            content = ""
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    content += content_block.text

            usage_dict = None
            if response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }

            return CompletionResponse(
                content=content,
                model=response.model,
                finish_reason=response.stop_reason,
                usage=usage_dict,
                metadata={
                    "id": response.id,
                    "type": response.type,
                    "role": response.role,
                },
            )
        except (AttributeError, TypeError) as e:
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
        if not self._client:
            await self.initialize()

        model = request.model or self.default_model

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ModelNotFoundError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Format messages
        system_message, formatted_messages = self._format_messages(request.messages)

        # Prepare request parameters
        completion_kwargs = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": request.max_tokens or 1024,  # Required by Anthropic
            "temperature": request.temperature,
            "stream": True,
        }

        # Add system message if present, with schema enhancement
        final_system_message = system_message
        if request.json_schema_data:
            try:
                schema_prompt = create_anthropic_schema_prompt(request.json_schema_data)
                if final_system_message:
                    final_system_message = f"{final_system_message}\n\n{schema_prompt}"
                else:
                    final_system_message = schema_prompt
                logger.debug(
                    f"Enhanced system message with JSON schema for Anthropic streaming model {model}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create schema prompt for Anthropic streaming, "
                    f"proceeding without schema: {e}"
                )

        if final_system_message:
            completion_kwargs["system"] = final_system_message

        # Add optional parameters
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p
        if request.stop:
            completion_kwargs["stop_sequences"] = request.stop

        # Make streaming API request
        try:
            self._request_count += 1
            async with self._client.messages.stream(**completion_kwargs) as stream:
                async for chunk in stream:
                    try:
                        if hasattr(chunk, "type"):
                            if chunk.type == "content_block_delta" and hasattr(
                                chunk, "delta"
                            ):
                                if hasattr(chunk.delta, "text"):
                                    yield StreamChunk(
                                        content=chunk.delta.text,
                                        metadata={
                                            "index": getattr(chunk, "index", 0),
                                        },
                                    )
                            elif chunk.type == "message_stop":
                                # Final chunk
                                yield StreamChunk(
                                    content="",
                                    finish_reason="stop",
                                    metadata={"type": "message_stop"},
                                )
                    except (AttributeError, TypeError) as e:
                        logger.warning(f"Invalid stream chunk: {e}")
                        continue
        except anthropic.AuthenticationError as e:
            self._error_count += 1
            raise AuthenticationError(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            self._error_count += 1
            retry_after = getattr(e, "retry_after", None)
            raise RateLimitError(
                f"Anthropic rate limit exceeded: {e}", retry_after=retry_after
            ) from e
        except anthropic.NotFoundError as e:
            self._error_count += 1
            raise ModelNotFoundError(f"Anthropic model not found: {e}") from e
        except anthropic.APIError as e:
            self._error_count += 1
            raise ProviderError(f"Anthropic API error: {e}") from e
        except Exception as e:
            self._error_count += 1
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
            # Claude 3.5 models
            "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
            "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
            "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
            # Claude 3 models
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            # Claude 2 models
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-2.0": {"input": 8.0, "output": 24.0},
            # Claude Instant
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
        """Process multiple completion requests with optimized batching.

        For smaller batches (<= batch_size), uses concurrent individual requests.
        For larger batches, processes in chunks to respect rate limits.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        if len(requests) <= self.batch_size:
            # Use concurrent processing for smaller batches
            return await super().batch_complete(requests)

        # Process larger batches in chunks
        return await self._batch_complete_chunked(requests)

    async def _batch_complete_chunked(
        self, requests: List[CompletionRequest]
    ) -> List[Union[CompletionResponse, Exception]]:
        """Process requests in chunks to manage rate limits.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        results = []

        # Process in chunks of batch_size
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]

            try:
                # Process chunk concurrently
                tasks = [self._safe_complete(req) for req in chunk]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(chunk_results)

                # Small delay between chunks to help with rate limiting
                if i + self.batch_size < len(requests):
                    await asyncio.sleep(0.1)

            except Exception as e:
                # If batch processing fails, add exceptions for all requests in chunk
                chunk_exceptions = [
                    ProviderError(f"Batch processing failed: {e}") for _ in chunk
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
