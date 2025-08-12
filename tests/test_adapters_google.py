"""Tests for Google Gemini adapter."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.metareason.adapters.base import (
    AdapterError,
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    ModelNotFoundError,
    ProviderError,
    StreamChunk,
)
from src.metareason.adapters.google import GoogleAdapter


class MockGoogleClient:
    """Mock Google GenAI client for testing."""

    def __init__(self, **kwargs):
        self.models = MagicMock()
        self.models.list = MagicMock()
        self.models.generate_content = MagicMock()
        self.models.generate_content_stream = MagicMock()


class MockResponse:
    """Mock Google API response."""

    def __init__(self, content="Test response", model="gemini-2.0-flash-001"):
        self.candidates = [MockCandidate(content)]
        self.usage_metadata = MockUsageMetadata()


class MockCandidate:
    """Mock response candidate."""

    def __init__(self, content="Test response"):
        self.content = MockContent(content)
        self.finish_reason = "STOP"
        self.safety_ratings = []


class MockContent:
    """Mock content with parts."""

    def __init__(self, text="Test response"):
        self.parts = [MockPart(text)]


class MockPart:
    """Mock content part."""

    def __init__(self, text="Test response"):
        self.text = text


class MockUsageMetadata:
    """Mock usage metadata."""

    def __init__(self):
        self.prompt_token_count = 10
        self.candidates_token_count = 15
        self.total_token_count = 25


class MockModel:
    """Mock model from list_models."""

    def __init__(self, name="models/gemini-2.0-flash-001"):
        self.name = name


@pytest.fixture
def google_adapter():
    """Create GoogleAdapter instance for testing."""
    return GoogleAdapter(
        api_key="test-api-key",
        default_model="gemini-2.0-flash-001",
    )


@pytest.fixture
def vertex_ai_adapter():
    """Create GoogleAdapter instance for Vertex AI testing."""
    return GoogleAdapter(
        use_vertex_ai=True,
        project_id="test-project",
        location="us-central1",
        default_model="gemini-1.5-pro",
    )


@pytest.fixture
def completion_request():
    """Create a test completion request."""
    return CompletionRequest(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello, how are you?"),
        ],
        model="gemini-2.0-flash-001",
        temperature=0.7,
        max_tokens=1000,
    )


class TestGoogleAdapterInitialization:
    """Test adapter initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        adapter = GoogleAdapter(api_key="test-key")
        assert adapter.api_key == "test-key"
        assert not adapter.use_vertex_ai
        assert adapter.default_model == "gemini-2.0-flash-001"

    def test_init_vertex_ai(self):
        """Test initialization with Vertex AI."""
        adapter = GoogleAdapter(
            use_vertex_ai=True, project_id="test-project", location="us-central1"
        )
        assert adapter.use_vertex_ai
        assert adapter.project_id == "test-project"
        assert adapter.location == "us-central1"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-api-key"})
    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        adapter = GoogleAdapter()
        assert adapter.api_key == "env-api-key"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-env-key"})
    def test_init_with_gemini_env_var(self):
        """Test initialization with GEMINI_API_KEY environment variable."""
        adapter = GoogleAdapter()
        assert adapter.api_key == "gemini-env-key"

    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="Google API key not provided"):
            GoogleAdapter()

    def test_init_vertex_ai_without_project_id_raises_error(self):
        """Test that Vertex AI without project_id raises ValueError during initialization."""
        adapter = GoogleAdapter(use_vertex_ai=True)
        with pytest.raises(
            ValueError, match="project_id is required when using Vertex AI"
        ):
            asyncio.run(adapter.initialize())


class TestGoogleAdapterClientInitialization:
    """Test client initialization."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_initialize_developer_api(self, mock_client_class, google_adapter):
        """Test initializing with Developer API."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        await google_adapter.initialize()

        mock_client_class.assert_called_once_with(api_key="test-api-key")
        assert google_adapter._client == mock_client

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_initialize_vertex_ai(self, mock_client_class, vertex_ai_adapter):
        """Test initializing with Vertex AI."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        await vertex_ai_adapter.initialize()

        mock_client_class.assert_called_once_with(
            vertexai=True, project="test-project", location="us-central1"
        )
        assert vertex_ai_adapter._client == mock_client

    async def test_cleanup(self, google_adapter):
        """Test client cleanup."""
        google_adapter._client = MockGoogleClient()
        google_adapter._initialized = True

        await google_adapter.cleanup()

        assert google_adapter._client is None
        assert not google_adapter._initialized


class TestMessageFormatting:
    """Test message formatting for Google API."""

    def test_format_messages_system_user_assistant(self, google_adapter):
        """Test formatting different message types."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]

        formatted = google_adapter._format_messages(messages)

        assert len(formatted) == 3
        assert formatted[0] == {"role": "user", "content": "System: You are helpful."}
        assert formatted[1] == {"role": "user", "content": "Hello!"}
        assert formatted[2] == {"role": "model", "content": "Hi there!"}


class TestCompletion:
    """Test completion functionality."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_success(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test successful completion."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_response = MockResponse()
        mock_client.models.generate_content.return_value = mock_response

        await google_adapter.initialize()
        response = await google_adapter.complete(completion_request)

        assert isinstance(response, CompletionResponse)
        assert response.content == "Test response"
        assert response.model == "gemini-2.0-flash-001"
        assert response.finish_reason == "STOP"
        assert response.usage["total_tokens"] == 25

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_model_validation(self, mock_client_class, google_adapter):
        """Test model validation in completion."""
        await google_adapter.initialize()

        invalid_request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="invalid-model",
        )

        with pytest.raises(
            ModelNotFoundError, match="Model 'invalid-model' not supported"
        ):
            await google_adapter.complete(invalid_request)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_authentication_error(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test authentication error handling."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Create a mock exception that looks like an authentication error
        auth_error = Exception("API_KEY invalid")
        mock_client.models.generate_content.side_effect = auth_error

        await google_adapter.initialize()

        with pytest.raises(AdapterError, match="Google request failed"):
            await google_adapter.complete(completion_request)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_rate_limit_error(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test rate limit error handling."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Create a mock exception that looks like a quota error
        quota_error = Exception("QUOTA exceeded")
        mock_client.models.generate_content.side_effect = quota_error

        await google_adapter.initialize()

        with pytest.raises(AdapterError, match="Google request failed"):
            await google_adapter.complete(completion_request)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_model_not_found_error(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test model not found error handling."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Create a mock exception that looks like a not found error
        not_found_error = Exception("NOT_FOUND")
        mock_client.models.generate_content.side_effect = not_found_error

        await google_adapter.initialize()

        with pytest.raises(AdapterError, match="Google request failed"):
            await google_adapter.complete(completion_request)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_generic_error(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test generic error handling."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Generic error")

        await google_adapter.initialize()

        with pytest.raises(AdapterError, match="Google request failed"):
            await google_adapter.complete(completion_request)


class TestStreamingCompletion:
    """Test streaming completion functionality."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_stream_success(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test successful streaming completion."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Create mock stream chunks
        chunk1 = MockResponse("Hello")
        chunk2 = MockResponse(" world!")
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]

        await google_adapter.initialize()

        chunks = []
        async for chunk in google_adapter.complete_stream(completion_request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world!"
        assert all(isinstance(chunk, StreamChunk) for chunk in chunks)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_complete_stream_error_handling(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test streaming error handling."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Create a mock exception
        stream_error = Exception("API_KEY invalid")
        mock_client.models.generate_content_stream.side_effect = stream_error

        await google_adapter.initialize()

        with pytest.raises(AdapterError):
            chunks = []
            async for chunk in google_adapter.complete_stream(completion_request):
                chunks.append(chunk)


class TestModelOperations:
    """Test model listing and validation."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_list_models_success(self, mock_client_class, google_adapter):
        """Test successful model listing."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_models = [
            MockModel("models/gemini-2.0-flash-001"),
            MockModel("models/gemini-1.5-pro"),
            MockModel("models/gemini-1.5-flash"),
        ]
        mock_client.models.list.return_value = mock_models

        await google_adapter.initialize()
        models = await google_adapter.list_models()

        assert "gemini-2.0-flash-001" in models
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_list_models_fallback(self, mock_client_class, google_adapter):
        """Test model listing fallback to known models."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_client.models.list.side_effect = Exception("API Error")

        await google_adapter.initialize()
        models = await google_adapter.list_models()

        # Should return known models as fallback
        assert len(models) > 0
        assert "gemini-2.0-flash-001" in models

    async def test_validate_model_known(self, google_adapter):
        """Test validation of known models."""
        assert await google_adapter.validate_model("gemini-2.0-flash-001")
        assert await google_adapter.validate_model("gemini-1.5-pro")

    async def test_validate_model_unknown(self, google_adapter):
        """Test validation of unknown models."""
        assert not await google_adapter.validate_model("unknown-model")


class TestCostEstimation:
    """Test cost estimation functionality."""

    async def test_estimate_cost_known_model(self, google_adapter, completion_request):
        """Test cost estimation for known models."""
        cost = await google_adapter.estimate_cost(completion_request)
        assert cost is not None
        assert cost >= 0

    async def test_estimate_cost_unknown_model(self, google_adapter):
        """Test cost estimation for unknown models."""
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="unknown-model",
        )
        cost = await google_adapter.estimate_cost(request)
        assert cost is None

    async def test_estimate_cost_free_model(self, google_adapter):
        """Test cost estimation for free tier models."""
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="gemini-2.0-flash-exp",
        )
        cost = await google_adapter.estimate_cost(request)
        assert cost == 0.0


class TestBatchProcessing:
    """Test batch processing functionality."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_batch_complete_small(self, mock_client_class, google_adapter):
        """Test batch processing with small batch."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_response = MockResponse()
        mock_client.models.generate_content.return_value = mock_response

        requests = [
            CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=f"Hello {i}")],
                model="gemini-2.0-flash-001",
            )
            for i in range(3)
        ]

        await google_adapter.initialize()
        results = await google_adapter.batch_complete(requests)

        assert len(results) == 3
        assert all(isinstance(r, CompletionResponse) for r in results)

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_batch_complete_large(self, mock_client_class, google_adapter):
        """Test batch processing with large batch."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_response = MockResponse()
        mock_client.models.generate_content.return_value = mock_response

        # Create more requests than batch_size (20)
        requests = [
            CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=f"Hello {i}")],
                model="gemini-2.0-flash-001",
            )
            for i in range(25)
        ]

        await google_adapter.initialize()
        results = await google_adapter.batch_complete(requests)

        assert len(results) == 25


class TestUsageStats:
    """Test usage statistics tracking."""

    async def test_usage_stats_initial(self, google_adapter):
        """Test initial usage statistics."""
        stats = await google_adapter.get_usage_stats()
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_usage_stats_after_requests(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test usage statistics after making requests."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_response = MockResponse()
        mock_client.models.generate_content.return_value = mock_response

        await google_adapter.initialize()
        await google_adapter.complete(completion_request)

        stats = await google_adapter.get_usage_stats()
        assert stats["request_count"] == 1
        assert stats["error_count"] == 0
        assert stats["error_rate"] == 0

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_usage_stats_after_error(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test usage statistics after error."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Test error")

        await google_adapter.initialize()

        with pytest.raises(AdapterError):
            await google_adapter.complete(completion_request)

        stats = await google_adapter.get_usage_stats()
        assert stats["request_count"] == 1
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 1.0


class TestContextManager:
    """Test async context manager functionality."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_context_manager(self, mock_client_class):
        """Test async context manager usage."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        async with GoogleAdapter(api_key="test-key") as adapter:
            assert adapter._initialized
            assert adapter._client is not None

        assert not adapter._initialized
        assert adapter._client is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.metareason.adapters.google.genai.Client")
    async def test_empty_response_candidates(
        self, mock_client_class, google_adapter, completion_request
    ):
        """Test handling of empty response candidates."""
        mock_client = MockGoogleClient()
        mock_client_class.return_value = mock_client

        # Mock response with no candidates
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        await google_adapter.initialize()

        with pytest.raises(ProviderError, match="No candidates returned"):
            await google_adapter.complete(completion_request)

    def test_string_representation(self, google_adapter):
        """Test string representations."""
        assert str(google_adapter) == "GoogleAdapter()"
        assert "GoogleAdapter" in repr(google_adapter)
        assert "config" in repr(google_adapter)

    async def test_default_model_fallback(self, google_adapter):
        """Test adapter handles model selection correctly."""
        # Test that the adapter uses the provided model correctly
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model="gemini-2.0-flash-001",  # Use a valid model
        )

        with patch.object(google_adapter, "_client") as mock_client:
            mock_client.models.generate_content.return_value = MockResponse()
            await google_adapter.initialize()

            response = await google_adapter.complete(request)
            assert isinstance(response, CompletionResponse)
            assert response.model == "gemini-2.0-flash-001"
