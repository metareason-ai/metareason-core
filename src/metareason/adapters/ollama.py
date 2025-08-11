"""Ollama adapter implementation for local model serving."""

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


class OllamaAdapter(BaseHTTPAdapter):
    """Adapter for Ollama local model serving."""

    # Popular models available via Ollama
    POPULAR_MODELS = {
        # Llama models
        "llama2": {"size": "7B", "family": "llama2"},
        "llama2:13b": {"size": "13B", "family": "llama2"},
        "llama2:70b": {"size": "70B", "family": "llama2"},
        "llama3": {"size": "8B", "family": "llama3"},
        "llama3:70b": {"size": "70B", "family": "llama3"},
        "llama3.1": {"size": "8B", "family": "llama3.1"},
        "llama3.1:70b": {"size": "70B", "family": "llama3.1"},
        "llama3.1:405b": {"size": "405B", "family": "llama3.1"},
        # Code models
        "codellama": {"size": "7B", "family": "codellama"},
        "codellama:13b": {"size": "13B", "family": "codellama"},
        "codellama:34b": {"size": "34B", "family": "codellama"},
        # Mistral models
        "mistral": {"size": "7B", "family": "mistral"},
        "mistral:7b": {"size": "7B", "family": "mistral"},
        "mixtral": {"size": "8x7B", "family": "mistral"},
        "mixtral:8x22b": {"size": "8x22B", "family": "mistral"},
        # Other popular models
        "phi": {"size": "2.7B", "family": "phi"},
        "phi3": {"size": "3.8B", "family": "phi"},
        "gemma": {"size": "7B", "family": "gemma"},
        "gemma:2b": {"size": "2B", "family": "gemma"},
        "neural-chat": {"size": "7B", "family": "neural-chat"},
        "starling-lm": {"size": "7B", "family": "starling"},
        "orca-mini": {"size": "3B", "family": "orca"},
        "vicuna": {"size": "7B", "family": "vicuna"},
        "wizardcoder": {"size": "34B", "family": "wizardcoder"},
        "zephyr": {"size": "7B", "family": "zephyr"},
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,  # Ollama doesn't use API keys
        default_model: str = "llama3",
        timeout: float = 120.0,  # Longer timeout for local inference
        pull_missing_models: bool = False,
        model_timeout: Optional[float] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize Ollama adapter.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            api_key: Not used for Ollama (kept for interface compatibility)
            default_model: Default model to use
            timeout: Request timeout (longer for local inference)
            pull_missing_models: Automatically pull models if not available
            model_timeout: Model-specific timeout override
            retry_config: Retry configuration
            rate_limit_config: Rate limit configuration
            config: Additional configuration
            **kwargs: Additional arguments
        """
        # Ollama doesn't require rate limiting by default (local server)
        default_rate_limit = RateLimitConfig(
            requests_per_second=None,
            requests_per_minute=None,
            concurrent_requests=5,  # Conservative for local inference
            burst_size=10,
        )

        # Default retry config for local server
        default_retry = RetryConfig(
            max_retries=2,  # Fewer retries for local
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,  # Less jitter needed for local
        )

        retry_cfg = RetryConfig(**retry_config) if retry_config else default_retry
        rate_limit_cfg = (
            RateLimitConfig(**rate_limit_config)
            if rate_limit_config
            else default_rate_limit
        )

        super().__init__(
            base_url=base_url.rstrip("/"),
            api_key=None,  # Ollama doesn't use API keys
            headers=kwargs.pop("headers", {}),
            timeout=timeout,
            retry_config=retry_cfg,
            rate_limit_config=rate_limit_cfg,
            config=config,
        )

        self.default_model = default_model
        self.pull_missing_models = pull_missing_models
        self.model_timeout = model_timeout or timeout
        self._available_models: Optional[List[str]] = None
        self._model_info_cache: Dict[str, Dict[str, Any]] = {}

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers for Ollama.

        Override to remove Authorization header since Ollama doesn't use API keys.
        """
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers,
        }

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for Ollama chat API.

        Args:
            messages: List of messages

        Returns:
            Formatted prompt string
        """
        # For simple models, we concatenate messages into a single prompt
        # More sophisticated models could use structured chat format
        formatted_parts = []

        for msg in messages:
            if msg.role.value == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role.value == "user":
                formatted_parts.append(f"Human: {msg.content}")
            elif msg.role.value == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")

        # Add final prompt for assistant response
        formatted_parts.append("Assistant:")

        return "\n\n".join(formatted_parts)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Ollama API.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            AdapterError: On API errors
        """
        model = request.model or self.default_model

        # Validate model availability
        if not await self.validate_model(model):
            available_models = await self.list_models()
            raise ModelNotFoundError(
                f"Model '{model}' not found. " f"Available models: {available_models}"
            )

        # Format prompt for Ollama
        prompt = self._format_messages(request.messages)

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or -1,  # -1 means no limit
                "top_p": request.top_p or 0.9,
                "repeat_penalty": 1.1,  # Default repeat penalty
            },
        }

        # Add stop sequences if provided
        if request.stop:
            payload["options"]["stop"] = request.stop

        # Make API request
        try:
            response = await self._request("POST", "api/generate", json_data=payload)
        except Exception as e:
            raise ProviderError(f"Ollama API request failed: {e}") from e

        # Extract response
        try:
            return CompletionResponse(
                content=response["response"],
                model=model,
                finish_reason="stop",  # Ollama doesn't provide detailed finish reasons
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": (
                        response.get("prompt_eval_count", 0)
                        + response.get("eval_count", 0)
                    ),
                },
                metadata={
                    "load_duration": response.get("load_duration"),
                    "prompt_eval_duration": response.get("prompt_eval_duration"),
                    "eval_duration": response.get("eval_duration"),
                    "total_duration": response.get("total_duration"),
                },
            )
        except (KeyError, TypeError) as e:
            raise ProviderError(f"Invalid Ollama API response: {e}") from e

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate streaming completion using Ollama API.

        Args:
            request: Completion request

        Yields:
            Stream chunks

        Raises:
            AdapterError: On API errors
        """
        model = request.model or self.default_model

        # Validate model availability
        if not await self.validate_model(model):
            available_models = await self.list_models()
            raise ModelNotFoundError(
                f"Model '{model}' not found. " f"Available models: {available_models}"
            )

        # Format prompt for Ollama
        prompt = self._format_messages(request.messages)

        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or -1,
                "top_p": request.top_p or 0.9,
                "repeat_penalty": 1.1,
            },
        }

        # Add stop sequences if provided
        if request.stop:
            payload["options"]["stop"] = request.stop

        # Make streaming API request
        try:
            async for chunk_data in self._stream_request(
                "POST", "api/generate", json_data=payload
            ):
                try:
                    if "response" in chunk_data:
                        yield StreamChunk(
                            content=chunk_data["response"],
                            finish_reason=(
                                "stop" if chunk_data.get("done", False) else None
                            ),
                            metadata={
                                "model": model,
                                "eval_count": chunk_data.get("eval_count"),
                                "eval_duration": chunk_data.get("eval_duration"),
                            },
                        )

                        # Check if generation is complete
                        if chunk_data.get("done", False):
                            break

                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid Ollama stream chunk: {e}")
                    continue
        except Exception as e:
            raise ProviderError(f"Ollama streaming request failed: {e}") from e

    async def list_models(self) -> List[str]:
        """List available Ollama models.

        Returns:
            List of model names
        """
        try:
            # Check if models are cached
            if self._available_models is not None:
                return self._available_models

            response = await self._request("GET", "api/tags")
            models = []

            for model_data in response.get("models", []):
                model_name = model_data.get("name", "").split(":")[0]  # Remove tag
                if model_name and model_name not in models:
                    models.append(model_name)

            # Cache the results
            self._available_models = sorted(models)
            return self._available_models

        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            # Return popular models as fallback
            return list(self.POPULAR_MODELS.keys())

    async def validate_model(self, model: str) -> bool:
        """Check if model is available in Ollama.

        Args:
            model: Model name

        Returns:
            True if model is available
        """
        try:
            available_models = await self.list_models()
            model_available = model in available_models

            # If model not available and auto-pull is enabled, try to pull it
            if not model_available and self.pull_missing_models:
                if (
                    model in self.POPULAR_MODELS or ":" in model
                ):  # Basic model name check
                    logger.info(
                        f"Model '{model}' not found locally. Attempting to pull..."
                    )
                    if await self.pull_model(model):
                        logger.info(f"Successfully pulled model '{model}'")
                        return True
                    else:
                        logger.warning(f"Failed to pull model '{model}'")

            return model_available
        except Exception:
            # Fallback to popular models check
            return model in self.POPULAR_MODELS

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a model.

        Args:
            model: Model name

        Returns:
            Model information dictionary
        """
        # Check cache first
        if model in self._model_info_cache:
            return self._model_info_cache[model]

        try:
            response = await self._request(
                "POST", "api/show", json_data={"name": model}
            )

            model_info = {
                "name": model,
                "size": response.get("size"),
                "digest": response.get("digest"),
                "modified": response.get("modified_at"),
                "format": response.get("details", {}).get("format"),
                "family": response.get("details", {}).get("family"),
                "parameter_size": response.get("details", {}).get("parameter_size"),
                "quantization_level": response.get("details", {}).get(
                    "quantization_level"
                ),
            }

            # Cache the result
            self._model_info_cache[model] = model_info
            return model_info

        except Exception as e:
            logger.warning(f"Failed to get model info for {model}: {e}")
            # Return basic info from POPULAR_MODELS if available
            if model in self.POPULAR_MODELS:
                return {"name": model, **self.POPULAR_MODELS[model]}
            return {"name": model}

    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            # This would stream the download progress, but for simplicity
            # we'll just make a simple request
            await self._request("POST", "api/pull", json_data={"name": model})

            # Clear cached models list to force refresh
            self._available_models = None
            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    async def estimate_cost(self, request: CompletionRequest) -> Optional[float]:
        """Estimate cost for completion request.

        Args:
            request: Completion request

        Returns:
            Always returns 0.0 since Ollama is free for local use
        """
        return 0.0  # Local models are free to use

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for Ollama adapter.

        Returns:
            Usage statistics including local-specific metrics
        """
        base_stats = await super().get_usage_stats()

        # Add Ollama-specific stats
        ollama_stats = {
            **base_stats,
            "local_deployment": True,
            "requires_api_key": False,
            "data_privacy": "full",  # No data sent to external servers
            "network_usage": "local_only",  # Only communicates with local server
            "cached_models": len(self._model_info_cache),
            "known_models": len(self._available_models or []),
        }

        return ollama_stats

    async def health_check(self) -> bool:
        """Check if Ollama server is running and accessible.

        Returns:
            True if server is healthy
        """
        try:
            response = await self._request("GET", "api/tags")
            return "models" in response
        except Exception:
            return False

    def get_privacy_info(self) -> Dict[str, Any]:
        """Get privacy information for this adapter.

        Returns:
            Privacy characteristics and guarantees
        """
        return {
            "local_processing": True,
            "data_retention": "local_only",
            "external_api_calls": False,
            "api_key_required": False,
            "telemetry_sent": False,
            "privacy_level": "maximum",
            "compliance": {
                "gdpr_compliant": True,
                "hipaa_friendly": True,
                "data_sovereignty": True,
            },
            "network_requirements": ["local_server_access"],
            "data_flow": "user -> local_ollama_server -> user",
        }

    def is_privacy_preserving(self) -> bool:
        """Check if this adapter preserves privacy.

        Returns:
            Always True for Ollama as it's local-only
        """
        return True

    def _sanitize_request_for_logging(
        self, request: CompletionRequest
    ) -> Dict[str, Any]:
        """Sanitize request data for privacy-safe logging.

        Args:
            request: Original completion request

        Returns:
            Sanitized request data safe for logging
        """
        return {
            "model": request.model or self.default_model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "message_count": len(request.messages),
            "has_system_message": any(
                msg.role.value == "system" for msg in request.messages
            ),
            "stream": request.stream,
            # Note: Never log actual message content for privacy
        }

    def __str__(self) -> str:
        """String representation of Ollama adapter."""
        return (
            f"OllamaAdapter(base_url={self.base_url}, "
            f"default_model={self.default_model})"
        )
