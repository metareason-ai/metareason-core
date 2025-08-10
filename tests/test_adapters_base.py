"""Tests for base adapter classes."""

from datetime import datetime

import pytest

from metareason.adapters.base import (
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


class MockAdapter(LLMAdapter):
    """Mock adapter for testing."""

    def __init__(self, config=None):
        super().__init__(config)
        self.initialized = False
        self.cleaned_up = False

    async def _initialize(self):
        self.initialized = True

    async def _cleanup(self):
        self.cleaned_up = True

    async def complete(self, request):
        return CompletionResponse(
            content="Mock response",
            model=request.model or "mock-model",
        )

    async def complete_stream(self, request):
        yield StreamChunk(content="Mock ", finish_reason=None)
        yield StreamChunk(content="stream", finish_reason="stop")

    async def list_models(self):
        return ["mock-model-1", "mock-model-2"]

    async def validate_model(self, model):
        return model.startswith("mock-")


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test creating a message with metadata."""
        metadata = {"source": "test"}
        msg = Message(role=MessageRole.ASSISTANT, content="Hi", metadata=metadata)
        assert msg.metadata == metadata


class TestCompletionRequest:
    """Tests for CompletionRequest class."""

    def test_request_creation(self):
        """Test creating a completion request."""
        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test-model")
        assert len(request.messages) == 1
        assert request.model == "test-model"
        assert request.temperature == 0.7  # default

    def test_request_validation(self):
        """Test request parameter validation."""
        messages = [Message(role=MessageRole.USER, content="Hello")]

        # Valid parameters
        request = CompletionRequest(
            messages=messages,
            model="test",
            temperature=0.5,
            max_tokens=100,
            top_p=0.8,
        )
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.top_p == 0.8

    def test_invalid_temperature(self):
        """Test invalid temperature values."""
        messages = [Message(role=MessageRole.USER, content="Hello")]

        with pytest.raises(ValueError):
            CompletionRequest(messages=messages, model="test", temperature=3.0)

        with pytest.raises(ValueError):
            CompletionRequest(messages=messages, model="test", temperature=-0.1)


class TestCompletionResponse:
    """Tests for CompletionResponse class."""

    def test_response_creation(self):
        """Test creating a completion response."""
        response = CompletionResponse(content="Hello", model="test-model")
        assert response.content == "Hello"
        assert response.model == "test-model"
        assert isinstance(response.created_at, datetime)

    def test_response_with_usage(self):
        """Test response with usage statistics."""
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        response = CompletionResponse(
            content="Hello",
            model="test",
            usage=usage,
            finish_reason="stop",
        )
        assert response.usage == usage
        assert response.finish_reason == "stop"


class TestStreamChunk:
    """Tests for StreamChunk class."""

    def test_chunk_creation(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(content="Hello", finish_reason=None)
        assert chunk.content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.metadata == {}


class TestAdapterErrors:
    """Tests for adapter error classes."""

    def test_base_error(self):
        """Test base adapter error."""
        error = AdapterError("Test error")
        assert str(error) == "Test error"

    def test_rate_limit_error(self):
        """Test rate limit error with retry_after."""
        error = RateLimitError("Rate limited", retry_after=30.0)
        assert error.retry_after == 30.0

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid API key")
        assert isinstance(error, AdapterError)

    def test_model_not_found_error(self):
        """Test model not found error."""
        error = ModelNotFoundError("Model not found")
        assert isinstance(error, AdapterError)

    def test_provider_error(self):
        """Test provider error."""
        error = ProviderError("Provider unavailable")
        assert isinstance(error, AdapterError)


class TestLLMAdapter:
    """Tests for base LLMAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create mock adapter for testing."""
        return MockAdapter({"test": "config"})

    async def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert not adapter._initialized
        await adapter.initialize()
        assert adapter._initialized
        assert adapter.initialized

    async def test_cleanup(self, adapter):
        """Test adapter cleanup."""
        await adapter.initialize()
        await adapter.cleanup()
        assert not adapter._initialized
        assert adapter.cleaned_up

    async def test_context_manager(self, adapter):
        """Test using adapter as context manager."""
        async with adapter:
            assert adapter._initialized
        assert not adapter._initialized
        assert adapter.cleaned_up

    async def test_complete(self, adapter):
        """Test completion."""
        await adapter.initialize()

        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test-model")

        response = await adapter.complete(request)
        assert isinstance(response, CompletionResponse)
        assert response.content == "Mock response"
        assert response.model == "test-model"

    async def test_complete_stream(self, adapter):
        """Test streaming completion."""
        await adapter.initialize()

        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test-model")

        chunks = []
        async for chunk in adapter.complete_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Mock "
        assert chunks[1].content == "stream"
        assert chunks[1].finish_reason == "stop"

    async def test_batch_complete(self, adapter):
        """Test batch completion."""
        await adapter.initialize()

        messages = [Message(role=MessageRole.USER, content="Hello")]
        requests = [
            CompletionRequest(messages=messages, model="test-1"),
            CompletionRequest(messages=messages, model="test-2"),
        ]

        responses = await adapter.batch_complete(requests)

        assert len(responses) == 2
        assert all(isinstance(r, CompletionResponse) for r in responses)

    async def test_list_models(self, adapter):
        """Test listing models."""
        await adapter.initialize()
        models = await adapter.list_models()
        assert "mock-model-1" in models
        assert "mock-model-2" in models

    async def test_validate_model(self, adapter):
        """Test model validation."""
        await adapter.initialize()
        assert await adapter.validate_model("mock-valid")
        assert not await adapter.validate_model("invalid-model")

    async def test_estimate_cost_default(self, adapter):
        """Test default cost estimation."""
        await adapter.initialize()
        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test")

        cost = await adapter.estimate_cost(request)
        assert cost is None  # Default implementation

    async def test_usage_stats_default(self, adapter):
        """Test default usage statistics."""
        await adapter.initialize()
        stats = await adapter.get_usage_stats()
        assert stats == {}  # Default implementation

    def test_str_repr(self, adapter):
        """Test string representations."""
        assert "MockAdapter" in str(adapter)
        assert "MockAdapter" in repr(adapter)
        assert "config" in repr(adapter)


class TestAsyncBehavior:
    """Tests for async behavior and error handling."""

    async def test_batch_complete_with_errors(self):
        """Test batch completion with some failures."""

        class FailingAdapter(MockAdapter):
            async def complete(self, request):
                if "fail" in request.model:
                    raise ProviderError("Model failed")
                return await super().complete(request)

        adapter = FailingAdapter()
        await adapter.initialize()

        messages = [Message(role=MessageRole.USER, content="Hello")]
        requests = [
            CompletionRequest(messages=messages, model="success"),
            CompletionRequest(messages=messages, model="fail-model"),
            CompletionRequest(messages=messages, model="success-2"),
        ]

        results = await adapter.batch_complete(requests)

        assert len(results) == 3
        assert isinstance(results[0], CompletionResponse)
        assert isinstance(results[1], ProviderError)
        assert isinstance(results[2], CompletionResponse)

    async def test_safe_complete_error_handling(self):
        """Test error handling in _safe_complete."""

        class ErrorAdapter(MockAdapter):
            async def complete(self, request):
                raise RuntimeError("Test error")

        adapter = ErrorAdapter()
        await adapter.initialize()

        messages = [Message(role=MessageRole.USER, content="Hello")]
        request = CompletionRequest(messages=messages, model="test")

        with pytest.raises(RuntimeError):
            await adapter._safe_complete(request)

    async def test_double_initialization(self):
        """Test that double initialization is handled correctly."""
        adapter = MockAdapter()

        await adapter.initialize()
        first_init = adapter.initialized

        await adapter.initialize()  # Should not reinitialize
        second_init = adapter.initialized

        assert first_init == second_init
        assert adapter._initialized

    async def test_cleanup_before_initialization(self):
        """Test cleanup before initialization."""
        adapter = MockAdapter()

        # Should not raise error
        await adapter.cleanup()
        assert not adapter.cleaned_up  # Should not call _cleanup
