"""Tests for adapter JSON schema integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from metareason.adapters.anthropic import AnthropicAdapter
from metareason.adapters.base import CompletionRequest, Message, MessageRole
from metareason.adapters.google import GoogleAdapter
from metareason.adapters.openai import OpenAIAdapter


class TestOpenAISchemaIntegration:
    """Test OpenAI adapter schema integration."""

    def test_supports_structured_output(self):
        """Test checking structured output support."""
        adapter = OpenAIAdapter(api_key="test-key")

        # GPT-4o models support structured output
        assert adapter._supports_structured_output("gpt-4o") is True
        assert adapter._supports_structured_output("gpt-4o-2024-08-06") is True
        assert adapter._supports_structured_output("gpt-4o-mini") is True

        # Older models don't support structured output
        assert adapter._supports_structured_output("gpt-4") is False
        assert adapter._supports_structured_output("gpt-3.5-turbo") is False

    @patch("metareason.adapters.openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_with_schema_supported_model(self, mock_openai_class):
        """Test completion with schema on supported model."""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"score": 0.8, "reasoning": "Good response"}'
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = OpenAIAdapter(api_key="test-key")
        await adapter.initialize()

        # Test schema data
        schema_data = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }

        # Create request with schema
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test message")],
            model="gpt-4o",
            json_schema_data=schema_data,
        )

        # Execute completion
        response = await adapter.complete(request)

        # Verify schema was passed to API
        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" in call_args.kwargs
        response_format = call_args.kwargs["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "response_schema"
        assert response_format["json_schema"]["schema"] == schema_data

        # Verify response
        assert response.content == '{"score": 0.8, "reasoning": "Good response"}'
        assert response.model == "gpt-4o"

    @patch("metareason.adapters.openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_with_schema_unsupported_model(self, mock_openai_class):
        """Test completion with schema on unsupported model logs warning."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Regular response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage = None
        mock_response.id = "test-id"
        mock_response.created = 1234567890

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = OpenAIAdapter(api_key="test-key")
        await adapter.initialize()

        schema_data = {"type": "object", "properties": {}}

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test message")],
            model="gpt-3.5-turbo",
            json_schema_data=schema_data,
        )

        with patch("metareason.adapters.openai.logger") as mock_logger:
            await adapter.complete(request)

            # Verify warning was logged
            mock_logger.info.assert_called_with(
                "Model gpt-3.5-turbo does not support native structured output, schema will be ignored"
            )

            # Verify no response_format was passed
            call_args = mock_client.chat.completions.create.call_args
            assert "response_format" not in call_args.kwargs


class TestGoogleSchemaIntegration:
    """Test Google adapter schema integration."""

    def test_supports_structured_output(self):
        """Test checking structured output support for Google models."""
        adapter = GoogleAdapter(api_key="test-key")

        # Gemini 1.5+ and 2.0 models support structured output
        assert adapter._supports_structured_output("gemini-2.0-flash-001") is True
        assert adapter._supports_structured_output("gemini-1.5-pro") is True
        assert adapter._supports_structured_output("gemini-1.5-flash") is True

        # Gemini 1.0 models don't support structured output
        assert adapter._supports_structured_output("gemini-1.0-pro") is False

    @patch("metareason.adapters.google.genai.Client")
    @pytest.mark.asyncio
    async def test_complete_with_schema_supported_model(self, mock_client_class):
        """Test Google completion with schema on supported model."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = '{"overall_score": 0.9}'
        mock_response.candidates[0].finish_reason = "stop"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 25

        mock_client.models.generate_content = Mock(return_value=mock_response)

        adapter = GoogleAdapter(api_key="test-key")
        await adapter.initialize()

        schema_data = {
            "type": "object",
            "properties": {"overall_score": {"type": "number"}},
            "required": ["overall_score"],
        }

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test message")],
            model="gemini-2.0-flash-001",
            json_schema_data=schema_data,
        )

        response = await adapter.complete(request)

        # Verify schema was passed to API
        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert "response_mime_type" in config
        assert config["response_mime_type"] == "application/json"
        assert "response_schema" in config
        assert config["response_schema"] == schema_data

        # Verify response
        assert response.content == '{"overall_score": 0.9}'


class TestAnthropicSchemaIntegration:
    """Test Anthropic adapter schema integration."""

    @patch("metareason.adapters.anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_complete_with_schema_prompt_enhancement(self, mock_anthropic_class):
        """Test Anthropic completion with schema enhances system prompt."""
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = (
            '{"passed": true, "reasoning": "Meets criteria"}'
        )
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.stop_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        mock_response.id = "test-id"
        mock_response.type = "message"
        mock_response.role = "assistant"

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.initialize()

        schema_data = {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "reasoning": {"type": "string"},
            },
            "required": ["passed", "reasoning"],
        }

        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="Original system message"),
                Message(role=MessageRole.USER, content="Test message"),
            ],
            model="claude-3-sonnet-20240229",
            json_schema_data=schema_data,
        )

        response = await adapter.complete(request)

        # Verify system message was enhanced with schema
        call_args = mock_client.messages.create.call_args
        system_message = call_args.kwargs["system"]

        assert "Original system message" in system_message
        assert "Please respond with valid JSON" in system_message
        assert "passed" in system_message
        assert "reasoning" in system_message
        assert "required" in system_message

        # Verify response
        assert response.content == '{"passed": true, "reasoning": "Meets criteria"}'

    @patch("metareason.adapters.anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_complete_with_schema_no_system_message(self, mock_anthropic_class):
        """Test Anthropic completion adds schema prompt when no system message."""
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"score": 0.7}'
        mock_response.model = "claude-3-sonnet-20240229"
        mock_response.stop_reason = "stop"
        mock_response.usage = None
        mock_response.id = "test-id"
        mock_response.type = "message"
        mock_response.role = "assistant"

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = AnthropicAdapter(api_key="test-key")
        await adapter.initialize()

        schema_data = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test message")],
            model="claude-3-sonnet-20240229",
            json_schema_data=schema_data,
        )

        await adapter.complete(request)

        # Verify system message contains only schema prompt
        call_args = mock_client.messages.create.call_args
        system_message = call_args.kwargs["system"]

        assert system_message.startswith("Please respond with valid JSON")
        assert "score" in system_message
        assert "required" in system_message


class TestCompletionRequestWithSchema:
    """Test CompletionRequest with json_schema_data field."""

    def test_completion_request_with_schema_data(self):
        """Test creating CompletionRequest with schema data."""
        schema_data = {
            "type": "object",
            "properties": {"test": {"type": "string"}},
            "required": ["test"],
        }

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test")],
            model="gpt-4o",
            json_schema_data=schema_data,
        )

        assert request.json_schema_data == schema_data
        assert request.model == "gpt-4o"

    def test_completion_request_without_schema_data(self):
        """Test creating CompletionRequest without schema data."""
        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test")], model="gpt-4"
        )

        assert request.json_schema_data is None

    def test_completion_request_with_all_parameters(self):
        """Test CompletionRequest with schema and all other parameters."""
        schema_data = {"type": "object", "properties": {}}

        request = CompletionRequest(
            messages=[Message(role=MessageRole.USER, content="Test")],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=-0.1,
            stop=["STOP"],
            stream=False,
            json_schema_data=schema_data,
        )

        assert request.json_schema_data == schema_data
        assert request.temperature == 0.5
        assert request.max_tokens == 1000
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.1
        assert request.presence_penalty == -0.1
        assert request.stop == ["STOP"]
        assert request.stream is False
