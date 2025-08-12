"""Google Gemini adapter implementation."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from google import genai
from google.genai import errors as genai_errors

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


class GoogleAdapter(LLMAdapter):
    """Adapter for Google Gemini API using Google GenAI SDK."""

    SUPPORTED_MODELS = {
        # Gemini 2.0 models
        "gemini-2.0-flash-exp": {"context": 1000000, "max_output": 8192},
        "gemini-2.0-flash-001": {"context": 1000000, "max_output": 8192},
        # Gemini 1.5 models
        "gemini-1.5-pro": {"context": 2000000, "max_output": 8192},
        "gemini-1.5-pro-exp-0801": {"context": 2000000, "max_output": 8192},
        "gemini-1.5-flash": {"context": 1000000, "max_output": 8192},
        "gemini-1.5-flash-002": {"context": 1000000, "max_output": 8192},
        "gemini-1.5-flash-8b": {"context": 1000000, "max_output": 8192},
        # Gemini 1.0 models (legacy support)
        "gemini-1.0-pro": {"context": 32760, "max_output": 2048},
        "gemini-1.0-pro-vision": {"context": 16384, "max_output": 2048},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "v1",
        default_model: str = "gemini-2.0-flash-001",
        timeout: float = 30.0,
        use_vertex_ai: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        batch_size: int = 20,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize Google Gemini adapter.

        Args:
            api_key: Google API key (for Gemini Developer API)
            base_url: Base URL for API (optional)
            api_version: API version
            default_model: Default model to use
            timeout: Request timeout
            use_vertex_ai: Use Vertex AI endpoint
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location
            batch_size: Maximum requests per batch
            config: Additional configuration
            **kwargs: Additional arguments
        """
        # Get API key from environment if not provided and not using Vertex AI
        if not use_vertex_ai and not api_key:
            import os

            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "Google API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY "
                    "environment variable or pass api_key parameter."
                )

        super().__init__(config)

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.default_model = default_model
        self.timeout = timeout
        self.use_vertex_ai = use_vertex_ai
        self.project_id = project_id
        self.location = location
        self.batch_size = batch_size
        self._client: Optional[genai.Client] = None
        self._request_count = 0
        self._error_count = 0

    async def _initialize(self) -> None:
        """Initialize Google GenAI client."""
        if self._client is None:
            if self.use_vertex_ai:
                if not self.project_id:
                    raise ValueError("project_id is required when using Vertex AI")

                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                )
            else:
                self._client = genai.Client(api_key=self.api_key)

    async def _cleanup(self) -> None:
        """Cleanup Google GenAI client."""
        if self._client:
            # Google GenAI client doesn't require explicit cleanup
            self._client = None

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for Google GenAI API.

        Args:
            messages: List of messages

        Returns:
            Formatted messages for Google API
        """
        formatted_messages = []

        for msg in messages:
            # Map roles from our format to Google's format
            if msg.role.value == "system":
                # Google uses "user" role for system messages in some contexts
                # We'll prepend system context to the first user message
                formatted_messages.append(
                    {"role": "user", "content": f"System: {msg.content}"}
                )
            elif msg.role.value == "user":
                formatted_messages.append({"role": "user", "content": msg.content})
            elif msg.role.value == "assistant":
                formatted_messages.append(
                    {
                        "role": "model",  # Google uses "model" instead of "assistant"
                        "content": msg.content,
                    }
                )

        return formatted_messages

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Google Gemini API.

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

        # Format messages for Google API
        formatted_messages = self._format_messages(request.messages)

        # Convert to content format expected by Google GenAI
        if len(formatted_messages) == 1:
            # Single message case
            contents = formatted_messages[0]["content"]
        else:
            # Multi-turn conversation
            contents = []
            for msg in formatted_messages:
                contents.append(
                    {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                )

        # Prepare generation config
        generation_config = {
            "temperature": request.temperature,
        }

        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.stop:
            generation_config["stop_sequences"] = request.stop

        # Make API request
        try:
            self._request_count += 1

            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=generation_config,
            )

        except genai_errors.ClientError as e:
            self._error_count += 1
            if "API_KEY" in str(e).upper():
                raise AuthenticationError(f"Google authentication failed: {e}") from e
            elif "QUOTA" in str(e).upper() or "RATE_LIMIT" in str(e).upper():
                raise RateLimitError(f"Google rate limit exceeded: {e}") from e
            elif "NOT_FOUND" in str(e).upper():
                raise ModelNotFoundError(f"Google model not found: {e}") from e
            else:
                raise ProviderError(f"Google API error: {e}") from e
        except Exception as e:
            self._error_count += 1
            raise AdapterError(f"Google request failed: {e}") from e

        # Extract response content
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                content = ""

                if candidate.content and candidate.content.parts:
                    content = "".join(
                        [
                            part.text
                            for part in candidate.content.parts
                            if hasattr(part, "text")
                        ]
                    )

                # Extract usage information if available
                usage_dict = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_dict = {
                        "prompt_tokens": getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        ),
                        "completion_tokens": getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        ),
                        "total_tokens": getattr(
                            response.usage_metadata, "total_token_count", 0
                        ),
                    }

                return CompletionResponse(
                    content=content,
                    model=model,
                    finish_reason=getattr(candidate, "finish_reason", None),
                    usage=usage_dict,
                    metadata={
                        "candidate_count": len(response.candidates),
                        "safety_ratings": getattr(candidate, "safety_ratings", []),
                    },
                )
            else:
                raise ProviderError("No candidates returned in Google API response")

        except (AttributeError, IndexError) as e:
            raise ProviderError(f"Invalid Google API response: {e}") from e

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion using Google Gemini API.

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

        # Format messages for Google API
        formatted_messages = self._format_messages(request.messages)

        # Convert to content format expected by Google GenAI
        if len(formatted_messages) == 1:
            contents = formatted_messages[0]["content"]
        else:
            contents = []
            for msg in formatted_messages:
                contents.append(
                    {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                )

        # Prepare generation config
        generation_config = {
            "temperature": request.temperature,
        }

        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.stop:
            generation_config["stop_sequences"] = request.stop

        # Make streaming API request
        try:
            self._request_count += 1

            stream = self._client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generation_config,
            )

            for chunk in stream:
                try:
                    if chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]

                        if candidate.content and candidate.content.parts:
                            content = "".join(
                                [
                                    part.text
                                    for part in candidate.content.parts
                                    if hasattr(part, "text") and part.text
                                ]
                            )

                            if content:
                                yield StreamChunk(
                                    content=content,
                                    finish_reason=getattr(
                                        candidate, "finish_reason", None
                                    ),
                                    metadata={
                                        "model": model,
                                        "safety_ratings": getattr(
                                            candidate, "safety_ratings", []
                                        ),
                                    },
                                )
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Invalid stream chunk: {e}")
                    continue

        except genai_errors.ClientError as e:
            self._error_count += 1
            if "API_KEY" in str(e).upper():
                raise AuthenticationError(f"Google authentication failed: {e}") from e
            elif "QUOTA" in str(e).upper() or "RATE_LIMIT" in str(e).upper():
                raise RateLimitError(f"Google rate limit exceeded: {e}") from e
            elif "NOT_FOUND" in str(e).upper():
                raise ModelNotFoundError(f"Google model not found: {e}") from e
            else:
                raise ProviderError(f"Google API error: {e}") from e
        except Exception as e:
            self._error_count += 1
            raise AdapterError(f"Google streaming request failed: {e}") from e

    async def list_models(self) -> List[str]:
        """List available Google Gemini models.

        Returns:
            List of model IDs
        """
        if not self._client:
            await self.initialize()

        try:
            models = []
            model_list = self._client.models.list()

            for model in model_list:
                if hasattr(model, "name") and model.name:
                    # Extract model name from full path (e.g., "models/gemini-pro" -> "gemini-pro")
                    model_name = (
                        model.name.split("/")[-1] if "/" in model.name else model.name
                    )

                    # Filter for supported models or Gemini models
                    if model_name in self.SUPPORTED_MODELS or model_name.startswith(
                        "gemini-"
                    ):
                        models.append(model_name)

            return sorted(models) if models else list(self.SUPPORTED_MODELS.keys())

        except Exception as e:
            logger.warning(f"Failed to list Google models: {e}")
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

        # Cost per 1M tokens (as of 2025 - approximate pricing)
        costs = {
            "gemini-2.0-flash-001": {"input": 0.075, "output": 0.30},
            "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
            "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
            "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
        }

        if model not in costs:
            return None

        # Estimate token count (rough approximation)
        # Google's tokenization might differ, but this gives a rough estimate
        input_tokens = sum(
            len(msg.content.split()) * 1.3 for msg in request.messages
        )  # ~1.3 tokens per word
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
        """Process multiple completion requests.

        Google GenAI SDK supports batch processing, but for simplicity and compatibility
        we use concurrent processing for now. This can be enhanced later with
        Google's batch API features.

        Args:
            requests: List of completion requests

        Returns:
            List of responses or exceptions for each request
        """
        if len(requests) <= self.batch_size:
            # Use concurrent processing for smaller batches
            return await super().batch_complete(requests)

        # Process in chunks for larger batches
        results = []

        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i : i + self.batch_size]

            try:
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
