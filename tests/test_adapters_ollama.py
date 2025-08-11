"""Tests for Ollama adapter."""

from unittest.mock import patch

import pytest

from metareason.adapters.base import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    ModelNotFoundError,
    ProviderError,
)
from metareason.adapters.ollama import OllamaAdapter


@pytest.fixture
def ollama_adapter():
    """Create an Ollama adapter for testing."""
    return OllamaAdapter(
        base_url="http://localhost:11434",
        default_model="test-model",
        pull_missing_models=False,
        timeout=30.0,
    )


@pytest.fixture
def sample_request():
    """Create a sample completion request."""
    return CompletionRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello, world!"),
        ],
        model="test-model",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "response": "Hello! How can I help you today?",
        "model": "test-model",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1234567890,
        "load_duration": 123456,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 654321,
        "eval_count": 20,
        "eval_duration": 987654,
    }


@pytest.fixture
def mock_models_response():
    """Mock Ollama models list response."""
    return {
        "models": [
            {
                "name": "test-model:latest",
                "modified_at": "2024-01-15T10:30:00.123456789Z",
                "size": 3825819519,
                "digest": "sha256:abc123",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0",
                },
            },
            {
                "name": "another-model:latest",
                "modified_at": "2024-01-16T11:00:00.123456789Z",
                "size": 7651639038,
                "digest": "sha256:def456",
                "details": {
                    "format": "gguf",
                    "family": "mistral",
                    "parameter_size": "13B",
                    "quantization_level": "Q5_1",
                },
            },
        ]
    }


class TestOllamaAdapter:
    """Test cases for OllamaAdapter."""

    def test_init(self):
        """Test adapter initialization."""
        adapter = OllamaAdapter(
            base_url="http://test:11434",
            default_model="custom-model",
            pull_missing_models=True,
            model_timeout=90.0,
        )

        assert adapter.base_url == "http://test:11434"
        assert adapter.default_model == "custom-model"
        assert adapter.pull_missing_models is True
        assert adapter.model_timeout == 90.0
        assert not adapter._initialized

    def test_init_defaults(self):
        """Test adapter initialization with defaults."""
        adapter = OllamaAdapter()

        assert adapter.base_url == "http://localhost:11434"
        assert adapter.default_model == "llama3"
        assert adapter.pull_missing_models is False
        assert adapter.model_timeout == 120.0

    def test_build_headers(self, ollama_adapter):
        """Test header building (should not include Authorization)."""
        headers = ollama_adapter._build_headers()

        assert "Content-Type" in headers
        assert "Accept" in headers
        assert "Authorization" not in headers  # Ollama doesn't use API keys

    def test_format_messages(self, ollama_adapter):
        """Test message formatting for Ollama."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]

        formatted = ollama_adapter._format_messages(messages)

        expected = (
            "System: You are helpful.\n\n"
            "Human: Hello!\n\n"
            "Assistant: Hi there!\n\n"
            "Assistant:"
        )
        assert formatted == expected

    def test_format_messages_user_only(self, ollama_adapter):
        """Test formatting with only user message."""
        messages = [Message(role=MessageRole.USER, content="Test question")]

        formatted = ollama_adapter._format_messages(messages)

        expected = "Human: Test question\n\nAssistant:"
        assert formatted == expected

    @pytest.mark.asyncio
    async def test_complete_success(
        self, ollama_adapter, sample_request, mock_ollama_response
    ):
        """Test successful completion."""
        with patch.object(
            ollama_adapter, "_request", return_value=mock_ollama_response
        ) as mock_request:
            with patch.object(ollama_adapter, "validate_model", return_value=True):
                response = await ollama_adapter.complete(sample_request)

                assert isinstance(response, CompletionResponse)
                assert response.content == "Hello! How can I help you today?"
                assert response.model == "test-model"
                assert response.finish_reason == "stop"
                assert response.usage["prompt_tokens"] == 10
                assert response.usage["completion_tokens"] == 20
                assert response.usage["total_tokens"] == 30

                # Verify request was made correctly
                mock_request.assert_called_once()
                args, kwargs = mock_request.call_args
                assert args[0] == "POST"
                assert args[1] == "api/generate"

                payload = kwargs["json_data"]
                assert payload["model"] == "test-model"
                assert payload["stream"] is False
                assert "temperature" in payload["options"]

    @pytest.mark.asyncio
    async def test_complete_model_not_found(self, ollama_adapter, sample_request):
        """Test completion with unavailable model."""
        with patch.object(ollama_adapter, "validate_model", return_value=False):
            with patch.object(
                ollama_adapter, "list_models", return_value=["other-model"]
            ):
                with pytest.raises(ModelNotFoundError) as exc_info:
                    await ollama_adapter.complete(sample_request)

                assert "test-model" in str(exc_info.value)
                assert "other-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_api_error(self, ollama_adapter, sample_request):
        """Test completion with API error."""
        with patch.object(ollama_adapter, "validate_model", return_value=True):
            with patch.object(
                ollama_adapter, "_request", side_effect=Exception("API Error")
            ):
                with pytest.raises(ProviderError) as exc_info:
                    await ollama_adapter.complete(sample_request)

                assert "Ollama API request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_invalid_response(self, ollama_adapter, sample_request):
        """Test completion with invalid API response."""
        invalid_response = {"invalid": "response"}

        with patch.object(ollama_adapter, "validate_model", return_value=True):
            with patch.object(
                ollama_adapter, "_request", return_value=invalid_response
            ):
                with pytest.raises(ProviderError) as exc_info:
                    await ollama_adapter.complete(sample_request)

                assert "Invalid Ollama API response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_complete_stream_success(self, ollama_adapter, sample_request):
        """Test successful streaming completion."""
        stream_chunks = [
            {"response": "Hello", "done": False},
            {"response": " there", "done": False},
            {"response": "!", "done": True, "eval_count": 3},
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in stream_chunks:
                yield chunk

        with patch.object(ollama_adapter, "validate_model", return_value=True):
            with patch.object(
                ollama_adapter, "_stream_request", side_effect=mock_stream
            ):
                chunks = []
                async for chunk in ollama_adapter.complete_stream(sample_request):
                    chunks.append(chunk)

                assert len(chunks) == 3
                assert chunks[0].content == "Hello"
                assert chunks[0].finish_reason is None
                assert chunks[1].content == " there"
                assert chunks[2].content == "!"
                assert chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_stream_model_not_found(
        self, ollama_adapter, sample_request
    ):
        """Test streaming with unavailable model."""
        with patch.object(ollama_adapter, "validate_model", return_value=False):
            with pytest.raises(ModelNotFoundError):
                async for _ in ollama_adapter.complete_stream(sample_request):
                    pass

    @pytest.mark.asyncio
    async def test_complete_stream_api_error(self, ollama_adapter, sample_request):
        """Test streaming with API error."""
        with patch.object(ollama_adapter, "validate_model", return_value=True):
            with patch.object(
                ollama_adapter, "_stream_request", side_effect=Exception("Stream error")
            ):
                with pytest.raises(ProviderError) as exc_info:
                    async for _ in ollama_adapter.complete_stream(sample_request):
                        pass

                assert "Ollama streaming request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_adapter, mock_models_response):
        """Test successful model listing."""
        with patch.object(
            ollama_adapter, "_request", return_value=mock_models_response
        ):
            models = await ollama_adapter.list_models()

            assert "test-model" in models
            assert "another-model" in models
            assert len(models) == 2

            # Should be cached
            assert ollama_adapter._available_models == models

    @pytest.mark.asyncio
    async def test_list_models_cached(self, ollama_adapter):
        """Test cached model listing."""
        ollama_adapter._available_models = ["cached-model"]

        # Should not make API call
        with patch.object(ollama_adapter, "_request") as mock_request:
            models = await ollama_adapter.list_models()

            assert models == ["cached-model"]
            mock_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_models_error_fallback(self, ollama_adapter):
        """Test model listing with API error."""
        with patch.object(
            ollama_adapter, "_request", side_effect=Exception("API Error")
        ):
            models = await ollama_adapter.list_models()

            # Should return popular models as fallback
            assert isinstance(models, list)
            assert len(models) > 0
            assert "llama2" in models

    @pytest.mark.asyncio
    async def test_validate_model_available(self, ollama_adapter):
        """Test model validation for available model."""
        with patch.object(
            ollama_adapter, "list_models", return_value=["test-model", "other-model"]
        ):
            is_valid = await ollama_adapter.validate_model("test-model")
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_model_unavailable(self, ollama_adapter):
        """Test model validation for unavailable model."""
        with patch.object(ollama_adapter, "list_models", return_value=["other-model"]):
            is_valid = await ollama_adapter.validate_model("missing-model")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_model_with_auto_pull(self, ollama_adapter):
        """Test model validation with auto-pull enabled."""
        ollama_adapter.pull_missing_models = True

        with patch.object(ollama_adapter, "list_models", return_value=["other-model"]):
            with patch.object(
                ollama_adapter, "pull_model", return_value=True
            ) as mock_pull:
                is_valid = await ollama_adapter.validate_model(
                    "llama3"
                )  # Popular model

                assert is_valid is True
                mock_pull.assert_called_once_with("llama3")

    @pytest.mark.asyncio
    async def test_validate_model_pull_failure(self, ollama_adapter):
        """Test model validation with pull failure."""
        ollama_adapter.pull_missing_models = True

        with patch.object(ollama_adapter, "list_models", return_value=["other-model"]):
            with patch.object(ollama_adapter, "pull_model", return_value=False):
                is_valid = await ollama_adapter.validate_model("llama3")
                assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_model_error_fallback(self, ollama_adapter):
        """Test model validation with API error."""
        with patch.object(
            ollama_adapter, "list_models", side_effect=Exception("API Error")
        ):
            # Should fallback to popular models check
            is_valid = await ollama_adapter.validate_model("llama2")
            assert is_valid is True

            is_valid = await ollama_adapter.validate_model("unknown-model")
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_model_info_success(self, ollama_adapter):
        """Test getting model information."""
        mock_response = {
            "name": "test-model",
            "size": 1234567890,
            "digest": "sha256:abc123",
            "modified_at": "2024-01-15T10:30:00.123456789Z",
            "details": {
                "format": "gguf",
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "Q4_0",
            },
        }

        with patch.object(ollama_adapter, "_request", return_value=mock_response):
            info = await ollama_adapter.get_model_info("test-model")

            assert info["name"] == "test-model"
            assert info["size"] == 1234567890
            assert info["family"] == "llama"
            assert info["parameter_size"] == "7B"

            # Should be cached
            assert ollama_adapter._model_info_cache["test-model"] == info

    @pytest.mark.asyncio
    async def test_get_model_info_cached(self, ollama_adapter):
        """Test cached model info retrieval."""
        cached_info = {"name": "cached-model", "cached": True}
        ollama_adapter._model_info_cache["cached-model"] = cached_info

        with patch.object(ollama_adapter, "_request") as mock_request:
            info = await ollama_adapter.get_model_info("cached-model")

            assert info == cached_info
            mock_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_model_info_error_fallback(self, ollama_adapter):
        """Test model info with API error."""
        with patch.object(
            ollama_adapter, "_request", side_effect=Exception("API Error")
        ):
            info = await ollama_adapter.get_model_info("llama2")  # Popular model

            assert info["name"] == "llama2"
            assert "family" in info  # Should have fallback info

    @pytest.mark.asyncio
    async def test_pull_model_success(self, ollama_adapter):
        """Test successful model pulling."""
        with patch.object(
            ollama_adapter, "_request", return_value={"status": "success"}
        ):
            result = await ollama_adapter.pull_model("new-model")

            assert result is True
            # Should clear cached models
            assert ollama_adapter._available_models is None

    @pytest.mark.asyncio
    async def test_pull_model_failure(self, ollama_adapter):
        """Test model pulling failure."""
        with patch.object(
            ollama_adapter, "_request", side_effect=Exception("Pull failed")
        ):
            result = await ollama_adapter.pull_model("new-model")
            assert result is False

    @pytest.mark.asyncio
    async def test_estimate_cost(self, ollama_adapter, sample_request):
        """Test cost estimation (should be 0 for local)."""
        cost = await ollama_adapter.estimate_cost(sample_request)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, ollama_adapter):
        """Test usage statistics."""
        ollama_adapter._request_count = 10
        ollama_adapter._error_count = 2
        ollama_adapter._model_info_cache = {"model1": {}, "model2": {}}
        ollama_adapter._available_models = ["model1", "model2", "model3"]

        stats = await ollama_adapter.get_usage_stats()

        assert stats["request_count"] == 10
        assert stats["error_count"] == 2
        assert stats["local_deployment"] is True
        assert stats["requires_api_key"] is False
        assert stats["data_privacy"] == "full"
        assert stats["network_usage"] == "local_only"
        assert stats["cached_models"] == 2
        assert stats["known_models"] == 3

    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_adapter):
        """Test successful health check."""
        mock_response = {"models": []}

        with patch.object(ollama_adapter, "_request", return_value=mock_response):
            healthy = await ollama_adapter.health_check()
            assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_adapter):
        """Test failed health check."""
        with patch.object(
            ollama_adapter, "_request", side_effect=Exception("Connection failed")
        ):
            healthy = await ollama_adapter.health_check()
            assert healthy is False

    def test_get_privacy_info(self, ollama_adapter):
        """Test privacy information retrieval."""
        info = ollama_adapter.get_privacy_info()

        assert info["local_processing"] is True
        assert info["external_api_calls"] is False
        assert info["api_key_required"] is False
        assert info["privacy_level"] == "maximum"
        assert info["compliance"]["gdpr_compliant"] is True
        assert info["compliance"]["hipaa_friendly"] is True

    def test_is_privacy_preserving(self, ollama_adapter):
        """Test privacy preservation check."""
        assert ollama_adapter.is_privacy_preserving() is True

    def test_sanitize_request_for_logging(self, ollama_adapter, sample_request):
        """Test request sanitization for logging."""
        sanitized = ollama_adapter._sanitize_request_for_logging(sample_request)

        assert sanitized["model"] == "test-model"
        assert sanitized["temperature"] == 0.7
        assert sanitized["max_tokens"] == 100
        assert sanitized["message_count"] == 2
        assert sanitized["has_system_message"] is True
        assert sanitized["stream"] is False

        # Ensure no actual content is logged
        assert "content" not in str(sanitized)

    def test_str_representation(self, ollama_adapter):
        """Test string representation."""
        result = str(ollama_adapter)
        expected = (
            "OllamaAdapter(base_url=http://localhost:11434, default_model=test-model)"
        )
        assert result == expected


class TestOllamaAdapterIntegration:
    """Integration tests for OllamaAdapter."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_manager(self, ollama_adapter):
        """Test using adapter as context manager."""
        with patch.object(ollama_adapter, "_initialize") as mock_init:
            with patch.object(ollama_adapter, "_cleanup") as mock_cleanup:
                async with ollama_adapter:
                    assert ollama_adapter._initialized is True

                mock_init.assert_called_once()
                mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_complete(self, ollama_adapter):
        """Test batch completion processing."""
        requests = [
            CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=f"Test {i}")],
                model="test-model",
            )
            for i in range(3)
        ]

        mock_response = CompletionResponse(content="Mock response", model="test-model")

        with patch.object(ollama_adapter, "complete", return_value=mock_response):
            results = await ollama_adapter.batch_complete(requests)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, CompletionResponse)
                assert result.content == "Mock response"
