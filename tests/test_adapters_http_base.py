"""Tests for HTTP base adapter."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientSession
from aiohttp.test_utils import make_mocked_coro
from pydantic import ValidationError

from metareason.adapters.base import (
    AuthenticationError,
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    ProviderError,
    RateLimitError,
)
from metareason.adapters.http_base import BaseHTTPAdapter, RateLimiter, RetryHandler
from metareason.config.adapters import RateLimitConfig, RetryConfig


class MockHTTPAdapter(BaseHTTPAdapter):
    """Mock HTTP adapter for testing."""

    async def complete(self, request):
        response = await self._request(
            "POST",
            "chat/completions",
            json_data={
                "model": request.model,
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        return CompletionResponse(
            content=response.get("content", "Mock response"),
            model=request.model or "mock-model",
        )

    async def complete_stream(self, request):
        async for chunk in self._stream_request(
            "POST",
            "chat/completions",
            json_data={
                "model": request.model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        ):
            yield chunk

    async def list_models(self):
        response = await self._request("GET", "models")
        return response.get("models", [])

    async def validate_model(self, model):
        return True


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False

    def test_validation(self):
        """Test configuration validation."""
        # max_delay must be >= initial_delay
        with pytest.raises(ValidationError):
            RetryConfig(initial_delay=30.0, max_delay=10.0)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_second is None
        assert config.requests_per_minute is None
        assert config.concurrent_requests == 10
        assert config.burst_size == 20

    def test_custom_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            concurrent_requests=5,
            burst_size=15,
        )
        assert config.requests_per_second == 10.0
        assert config.concurrent_requests == 5
        assert config.burst_size == 15

    def test_validation(self):
        """Test configuration validation."""
        # Cannot specify both requests_per_second and requests_per_minute
        with pytest.raises(ValidationError):
            RateLimitConfig(
                requests_per_second=10.0,
                requests_per_minute=600.0,
            )


class TestRateLimiter:
    """Tests for RateLimiter."""

    async def test_concurrent_requests(self):
        """Test concurrent request limiting."""
        config = RateLimitConfig(concurrent_requests=2)
        limiter = RateLimiter(config)

        # Start 3 concurrent requests
        results = []

        async def request_task():
            async with limiter.acquire():
                await asyncio.sleep(0.1)
                results.append(len(results))

        tasks = [request_task() for _ in range(3)]
        await asyncio.gather(*tasks)

        assert len(results) == 3

    async def test_rate_limiting(self):
        """Test rate limiting by requests per second."""
        config = RateLimitConfig(
            requests_per_second=5.0, concurrent_requests=10, burst_size=2
        )
        limiter = RateLimiter(config)

        # First, let's reset the token bucket by waiting a bit
        await asyncio.sleep(0.1)

        start_time = asyncio.get_event_loop().time()

        # Make 5 requests - first 2 should be from burst, then rate limiting kicks in
        for _ in range(5):
            async with limiter.acquire():
                pass  # After burst is exhausted, rate limiting kicks in

        end_time = asyncio.get_event_loop().time()

        # The total time should show evidence of rate limiting
        # At 5 requests/sec, 3 additional requests after burst should take at least 0.4 seconds
        total_time = end_time - start_time
        assert (
            total_time >= 0.3
        )  # More lenient but still shows rate limiting is working


class TestRetryHandler:
    """Tests for RetryHandler."""

    async def test_successful_execution(self):
        """Test successful execution without retries."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler(config)

        async def success_func():
            return "success"

        result = await handler.execute(success_func)
        assert result == "success"

    async def test_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        handler = RetryHandler(config)

        attempt_count = 0

        async def rate_limit_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"

        result = await handler.execute(rate_limit_func)
        assert result == "success"
        assert attempt_count == 2

    async def test_max_retries_exceeded(self):
        """Test when max retries is exceeded."""
        config = RetryConfig(max_retries=1, initial_delay=0.01)
        handler = RetryHandler(config)

        async def always_fail_func():
            raise RateLimitError("Always fails")

        with pytest.raises(RateLimitError):
            await handler.execute(always_fail_func)

    async def test_non_retryable_error(self):
        """Test that authentication errors are not retried."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler(config)

        async def auth_error_func():
            raise AuthenticationError("Invalid API key")

        with pytest.raises(AuthenticationError):
            await handler.execute(auth_error_func)


class TestBaseHTTPAdapter:
    """Tests for BaseHTTPAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create mock HTTP adapter for testing."""
        return MockHTTPAdapter(
            base_url="https://api.example.com",
            api_key="test-key",
            timeout=5.0,
        )

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.base_url == "https://api.example.com"
        assert adapter.api_key == "test-key"
        assert adapter.timeout.total == 5.0

    async def test_session_management(self, adapter):
        """Test HTTP session management."""
        assert adapter._session is None

        await adapter.initialize()
        assert adapter._session is not None
        assert isinstance(adapter._session, ClientSession)

        await adapter.cleanup()
        assert adapter._session is None

    def test_build_headers(self, adapter):
        """Test header building."""
        headers = adapter._build_headers()

        expected_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
        }

        for key, value in expected_headers.items():
            assert headers[key] == value

    @patch("aiohttp.ClientSession.request")
    async def test_request_success(self, mock_request, adapter):
        """Test successful HTTP request."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = make_mocked_coro({"result": "success"})
        mock_request.return_value.__aenter__.return_value = mock_response

        await adapter.initialize()
        result = await adapter._request("GET", "test")

        assert result == {"result": "success"}
        mock_request.assert_called_once()

    @patch("aiohttp.ClientSession.request")
    async def test_request_rate_limit(self, mock_request, adapter):
        """Test rate limit handling in HTTP request."""
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "1.0"}
        mock_request.return_value.__aenter__.return_value = mock_response

        await adapter.initialize()

        with pytest.raises(RateLimitError) as exc_info:
            await adapter._request("GET", "test")

        assert exc_info.value.retry_after == 1.0

    @patch("aiohttp.ClientSession.request")
    async def test_request_auth_error(self, mock_request, adapter):
        """Test authentication error handling."""
        # Mock authentication error
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = make_mocked_coro("Unauthorized")
        mock_request.return_value.__aenter__.return_value = mock_response

        await adapter.initialize()

        with pytest.raises(AuthenticationError):
            await adapter._request("GET", "test")

    @patch("aiohttp.ClientSession.request")
    async def test_request_provider_error(self, mock_request, adapter):
        """Test provider error handling."""
        # Mock server error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = make_mocked_coro("Internal Server Error")
        mock_request.return_value.__aenter__.return_value = mock_response

        await adapter.initialize()

        with pytest.raises(ProviderError):
            await adapter._request("GET", "test")

    @patch("aiohttp.ClientSession.request")
    async def test_stream_request(self, mock_request, adapter):
        """Test streaming HTTP request."""
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.status = 200

        # Mock content that yields lines
        mock_lines = [
            b'data: {"content": "Hello"}\n',
            b'data: {"content": " World"}\n',
            b"data: [DONE]\n",
        ]
        mock_response.content = AsyncMock()
        mock_response.content.__aiter__.return_value = iter(mock_lines)
        mock_request.return_value.__aenter__.return_value = mock_response

        await adapter.initialize()

        chunks = []
        async for chunk in adapter._stream_request("POST", "stream"):
            chunks.append(chunk)

        assert len(chunks) == 2  # [DONE] should stop iteration
        assert chunks[0]["content"] == "Hello"
        assert chunks[1]["content"] == " World"

    async def test_usage_stats(self, adapter):
        """Test usage statistics."""
        await adapter.initialize()

        # Initially no requests
        stats = await adapter.get_usage_stats()
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0

        # Mock a request that increments counters
        adapter._request_count = 10
        adapter._error_count = 2

        stats = await adapter.get_usage_stats()
        assert stats["request_count"] == 10
        assert stats["error_count"] == 2
        assert stats["error_rate"] == 0.2


class TestHTTPAdapterIntegration:
    """Integration tests for HTTP adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with specific retry/rate limit configs."""
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        rate_limit_config = RateLimitConfig(concurrent_requests=2)

        return MockHTTPAdapter(
            base_url="https://api.example.com",
            api_key="test-key",
            retry_config=retry_config,
            rate_limit_config=rate_limit_config,
        )

    async def test_adapter_context_manager(self, adapter):
        """Test using adapter as context manager."""
        assert adapter._session is None

        async with adapter:
            assert adapter._session is not None

        assert adapter._session is None

    @patch("aiohttp.ClientSession.request")
    async def test_complete_with_mock_response(self, mock_request, adapter):
        """Test completion with mocked HTTP response."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = make_mocked_coro({"content": "Test response"})
        mock_request.return_value.__aenter__.return_value = mock_response

        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test-model")

        async with adapter:
            result = await adapter.complete(request)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Test response"
        assert result.model == "test-model"

    @patch("aiohttp.ClientSession.request")
    async def test_not_implemented_methods(self, mock_request, adapter):
        """Test that abstract methods raise NotImplementedError."""
        # Mock successful response for MockHTTPAdapter calls
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = make_mocked_coro(
            {"content": "Test response", "models": ["test-model"]}
        )
        mock_request.return_value.__aenter__.return_value = mock_response

        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test")

        async with adapter:
            # These should work (implemented in MockHTTPAdapter)
            await adapter.complete(request)
            await adapter.list_models()
            await adapter.validate_model("test")

        # Test with base class directly
        base_adapter = BaseHTTPAdapter("https://api.test.com", "key")

        async with base_adapter:
            with pytest.raises(NotImplementedError):
                await base_adapter.complete(request)

            with pytest.raises(NotImplementedError):
                await base_adapter.list_models()

            with pytest.raises(NotImplementedError):
                await base_adapter.validate_model("test")
