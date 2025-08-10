"""Base HTTP adapter with retry logic and rate limiting."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
from aiohttp import ClientError, ClientSession, ClientTimeout

from .base import (
    AdapterError,
    AuthenticationError,
    CompletionRequest,
    CompletionResponse,
    LLMAdapter,
    ProviderError,
    RateLimitError,
    StreamChunk,
)

# Import config classes to avoid circular imports
try:
    from ..config.adapters import RateLimitConfig, RetryConfig
except ImportError:
    # Fallback to creating simple dataclasses if config module not available
    from dataclasses import dataclass

    @dataclass
    class RetryConfig:
        max_retries: int = 3
        initial_delay: float = 1.0
        max_delay: float = 60.0
        exponential_base: float = 2.0
        jitter: bool = True

    @dataclass
    class RateLimitConfig:
        requests_per_second: Optional[float] = None
        requests_per_minute: Optional[float] = None
        concurrent_requests: int = 10
        burst_size: int = 20


logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrent_requests)

        # Token bucket for rate limiting
        if config.requests_per_second:
            self._rate = config.requests_per_second
        elif config.requests_per_minute:
            self._rate = config.requests_per_minute / 60.0
        else:
            self._rate = None

        self._bucket_size = config.burst_size
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        """Acquire permission to make request."""
        async with self._semaphore:
            if self._rate:
                await self._wait_for_token()
            yield

    async def _wait_for_token(self):
        """Wait for token to become available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update

            # Refill tokens
            if self._rate:
                self._tokens = min(
                    self._bucket_size, self._tokens + elapsed * self._rate
                )
            self._last_update = now

            # Wait if no tokens available
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rate if self._rate else 0
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class RetryHandler:
    """Handle retries with exponential backoff."""

    def __init__(self, config: RetryConfig):
        """Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config

    async def execute(self, func, *args, **kwargs):
        """Execute function with retries.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, asyncio.TimeoutError, ClientError) as e:
                last_exception = e

                if attempt == self.config.max_retries:
                    break

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base**attempt),
                    self.config.max_delay,
                )

                # Add jitter if configured
                if self.config.jitter:
                    import secrets

                    delay *= 0.5 + secrets.SystemRandom().random()

                # Use retry_after if provided in RateLimitError
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
            except (AuthenticationError, ProviderError) as e:
                # Don't retry on authentication or provider errors
                logger.error(f"Non-retryable error: {e}")
                raise

        logger.error(f"All retry attempts exhausted. Last error: {last_exception}")
        raise last_exception


class BaseHTTPAdapter(LLMAdapter):
    """Base adapter for HTTP-based LLM providers."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize HTTP adapter.

        Args:
            base_url: Base URL for API
            api_key: API key for authentication
            headers: Additional headers
            timeout: Request timeout in seconds
            retry_config: Retry configuration
            rate_limit_config: Rate limit configuration
            config: Additional configuration
        """
        super().__init__(config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = ClientTimeout(total=timeout)

        self.retry_handler = RetryHandler(retry_config or RetryConfig())
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())

        self._session: Optional[ClientSession] = None
        self._request_count = 0
        self._error_count = 0

    async def _initialize(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self._session = ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers=self._build_headers(),
            )

    async def _cleanup(self) -> None:
        """Cleanup HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.headers,
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry and rate limiting.

        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: JSON data for request body
            **kwargs: Additional arguments for request

        Returns:
            Response data

        Raises:
            AdapterError: On request failure
        """
        if not self._session:
            await self.initialize()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async def _make_request():
            async with self.rate_limiter.acquire():
                self._request_count += 1

                try:
                    async with self._session.request(
                        method,
                        url,
                        json=json_data,
                        **kwargs,
                    ) as response:
                        # Check for rate limiting
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            retry_after = float(retry_after) if retry_after else None
                            raise RateLimitError(
                                f"Rate limit exceeded: {response.status}",
                                retry_after=retry_after,
                            )

                        # Check for authentication errors
                        if response.status in (401, 403):
                            text = await response.text()
                            raise AuthenticationError(
                                f"Authentication failed: {response.status} - {text}"
                            )

                        # Check for other errors
                        if response.status >= 400:
                            text = await response.text()
                            raise ProviderError(
                                f"Provider error: {response.status} - {text}"
                            )

                        return await response.json()

                except ClientError as e:
                    self._error_count += 1
                    raise AdapterError(f"HTTP request failed: {e}") from e

        return await self.retry_handler.execute(_make_request)

    async def _stream_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Make streaming HTTP request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: JSON data for request body
            **kwargs: Additional arguments for request

        Yields:
            Response chunks

        Raises:
            AdapterError: On request failure
        """
        if not self._session:
            await self.initialize()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with self.rate_limiter.acquire():
            self._request_count += 1

            try:
                async with self._session.request(
                    method,
                    url,
                    json=json_data,
                    **kwargs,
                ) as response:
                    # Check for errors
                    if response.status >= 400:
                        text = await response.text()
                        if response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            retry_after = float(retry_after) if retry_after else None
                            raise RateLimitError(
                                f"Rate limit exceeded: {response.status}",
                                retry_after=retry_after,
                            )
                        elif response.status in (401, 403):
                            raise AuthenticationError(
                                f"Authentication failed: {response.status} - {text}"
                            )
                        else:
                            raise ProviderError(
                                f"Provider error: {response.status} - {text}"
                            )

                    # Stream response
                    async for line in response.content:
                        if line:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    yield json.loads(data)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON: {data}")

            except ClientError as e:
                self._error_count += 1
                raise AdapterError(f"Streaming request failed: {e}") from e

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0
            ),
        }

    # Abstract methods that must be implemented by subclasses
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Must be implemented by subclass."""
        raise NotImplementedError

    async def complete_stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[StreamChunk]:
        """Must be implemented by subclass."""
        raise NotImplementedError
        yield  # Make this an async generator

    async def list_models(self) -> List[str]:
        """Must be implemented by subclass."""
        raise NotImplementedError

    async def validate_model(self, model: str) -> bool:
        """Must be implemented by subclass."""
        raise NotImplementedError
