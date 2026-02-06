import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from metareason.adapters.adapter_base import (
    AdapterException,
    AdapterRequest,
    AdapterResponse,
)
from metareason.adapters.adapter_factory import get_adapter
from metareason.adapters.anthropic import AnthropicAdapter, AnthropicAdapterException
from metareason.adapters.google import GoogleAdapter, GoogleAdapterException
from metareason.adapters.ollama import OllamaAdapter, OllamaException
from metareason.adapters.openai import OpenAIAdapter, OpenAIAdapterException


@pytest.fixture
def adapter_request():
    return AdapterRequest(
        model="test-model",
        system_prompt="You are helpful.",
        user_prompt="Hello",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )


@pytest.fixture
def adapter_request_no_system():
    return AdapterRequest(
        model="test-model",
        user_prompt="Hello",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )


# --- AdapterRequest / AdapterResponse validation ---


class TestAdapterModels:
    def test_adapter_request_valid(self):
        req = AdapterRequest(
            model="m", user_prompt="hi", temperature=1.0, top_p=0.5, max_tokens=10
        )
        assert req.model == "m"
        assert req.system_prompt is None

    def test_adapter_request_temperature_too_high(self):
        with pytest.raises(ValidationError):
            AdapterRequest(
                model="m", user_prompt="hi", temperature=3.0, top_p=0.5, max_tokens=10
            )

    def test_adapter_request_top_p_zero(self):
        with pytest.raises(ValidationError):
            AdapterRequest(
                model="m", user_prompt="hi", temperature=1.0, top_p=0.0, max_tokens=10
            )

    def test_adapter_request_max_tokens_zero(self):
        with pytest.raises(ValidationError):
            AdapterRequest(
                model="m", user_prompt="hi", temperature=1.0, top_p=0.5, max_tokens=0
            )

    def test_adapter_response_valid(self):
        resp = AdapterResponse(response_text="hello")
        assert resp.response_text == "hello"


# --- Adapter Factory ---


class TestAdapterFactory:
    def test_get_adapter_ollama(self):
        mock_cls = MagicMock()
        with patch.dict(
            "metareason.adapters.adapter_factory.ADAPTER_REGISTRY",
            {"ollama": mock_cls},
        ):
            adapter = get_adapter("ollama")
        mock_cls.assert_called_once()
        assert adapter is mock_cls.return_value

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(AdapterException, match="Unknown adapter"):
            get_adapter("nonexistent")

    @patch("metareason.adapters.adapter_factory.OllamaAdapter")
    def test_factory_sanitizes_keys_in_log(self, mock_cls, caplog):
        import logging

        mock_cls.return_value = MagicMock()
        with caplog.at_level(logging.DEBUG):
            get_adapter("ollama", api_key="secret123", location="us")
        assert "secret123" not in caplog.text

    def test_factory_wraps_init_exception(self):
        mock_cls = MagicMock(side_effect=TypeError("bad kwarg"))
        with patch.dict(
            "metareason.adapters.adapter_factory.ADAPTER_REGISTRY",
            {"openai": mock_cls},
        ):
            with pytest.raises(AdapterException, match="Failed to initialize"):
                get_adapter("openai")


# --- Ollama Adapter ---


class TestOllamaAdapter:
    @patch("metareason.adapters.ollama.AsyncClient")
    def test_init_creates_client(self, mock_client_cls):
        adapter = OllamaAdapter()
        assert adapter.chat_client is not None

    @patch("metareason.adapters.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_send_request_success(self, mock_client_cls, adapter_request):
        mock_chat = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = "Test response"
        mock_chat.return_value = mock_response
        mock_client_cls.return_value.chat = mock_chat

        adapter = OllamaAdapter()
        result = await adapter.send_request(adapter_request)

        assert isinstance(result, AdapterResponse)
        assert result.response_text == "Test response"
        mock_chat.assert_called_once()

    @patch("metareason.adapters.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_send_request_without_system_prompt(
        self, mock_client_cls, adapter_request_no_system
    ):
        mock_chat = AsyncMock()
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_chat.return_value = mock_response
        mock_client_cls.return_value.chat = mock_chat

        adapter = OllamaAdapter()
        result = await adapter.send_request(adapter_request_no_system)

        assert result.response_text == "Response"
        call_kwargs = mock_chat.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("metareason.adapters.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_send_request_module_not_found(
        self, mock_client_cls, adapter_request
    ):
        mock_client_cls.return_value.chat = AsyncMock(
            side_effect=ModuleNotFoundError("not found")
        )
        adapter = OllamaAdapter()
        with pytest.raises(OllamaException, match="not found"):
            await adapter.send_request(adapter_request)


# --- OpenAI Adapter ---


class TestOpenAIAdapter:
    @patch("metareason.adapters.openai.AsyncOpenAI")
    def test_init_with_api_key(self, mock_openai_cls):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter()
            assert adapter.client is not None

    def test_init_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(OpenAIAdapterException, match="OPENAI_API_KEY"):
                OpenAIAdapter()

    @patch("metareason.adapters.openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_send_request_success(self, mock_openai_cls, adapter_request):
        mock_response = MagicMock()
        mock_response.output_text = "OpenAI response"
        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.responses.create = mock_create
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter()
        result = await adapter.send_request(adapter_request)

        assert isinstance(result, AdapterResponse)
        assert result.response_text == "OpenAI response"

    @patch("metareason.adapters.openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_send_request_includes_system_prompt(
        self, mock_openai_cls, adapter_request
    ):
        mock_response = MagicMock()
        mock_response.output_text = "response"
        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.responses.create = mock_create
        mock_openai_cls.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIAdapter()
        await adapter.send_request(adapter_request)

        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "developer"
        assert messages[0]["content"] == "You are helpful."


# --- Anthropic Adapter ---


class TestAnthropicAdapter:
    @patch("metareason.adapters.anthropic.AsyncAnthropic")
    def test_init_with_api_key(self, mock_anthropic_cls):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter()
            assert adapter.client is not None

    def test_init_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AnthropicAdapterException, match="ANTHROPIC_API_KEY"):
                AnthropicAdapter()

    @patch("metareason.adapters.anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_send_request_success(self, mock_anthropic_cls, adapter_request):
        mock_text_block = MagicMock()
        mock_text_block.text = "Claude response"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        mock_anthropic_cls.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter()
        result = await adapter.send_request(adapter_request)

        assert isinstance(result, AdapterResponse)
        assert result.response_text == "Claude response"

    @patch("metareason.adapters.anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_send_request_passes_system_prompt(
        self, mock_anthropic_cls, adapter_request
    ):
        mock_text_block = MagicMock()
        mock_text_block.text = "response"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_create = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.messages.create = mock_create
        mock_anthropic_cls.return_value = mock_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter()
        await adapter.send_request(adapter_request)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."


# --- Google Adapter ---


class TestGoogleAdapter:
    def test_init_stores_config(self):
        adapter = GoogleAdapter(vertex_ai=True, project_id="proj")
        assert adapter.vertex_ai is True
        assert adapter.config["project_id"] == "proj"

    def test_init_developer_api_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = GoogleAdapter()
            with pytest.raises(GoogleAdapterException, match="api_key is required"):
                adapter._init_developer_api()

    def test_init_vertex_ai_missing_project_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = GoogleAdapter(vertex_ai=True)
            with pytest.raises(GoogleAdapterException, match="project_id is required"):
                adapter._init_vertex_ai()

    def test_init_vertex_ai_missing_location_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = GoogleAdapter(vertex_ai=True, project_id="proj")
            with pytest.raises(GoogleAdapterException, match="location is required"):
                adapter._init_vertex_ai()

    @patch("metareason.adapters.google.Client")
    @pytest.mark.asyncio
    async def test_send_request_developer_api(self, mock_client_cls, adapter_request):
        mock_response = MagicMock()
        mock_response.text = "Google response"
        mock_aio = AsyncMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_aio.aclose = AsyncMock()
        mock_client_cls.return_value.aio = mock_aio

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            adapter = GoogleAdapter()
            result = await adapter.send_request(adapter_request)

        assert isinstance(result, AdapterResponse)
        assert result.response_text == "Google response"
        mock_aio.aclose.assert_called_once()

    @patch("metareason.adapters.google.Client")
    @pytest.mark.asyncio
    async def test_send_request_vertex_ai(self, mock_client_cls, adapter_request):
        mock_response = MagicMock()
        mock_response.text = "Vertex response"
        mock_aio = AsyncMock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_aio.aclose = AsyncMock()
        mock_client_cls.return_value.aio = mock_aio

        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "1"}, clear=False):
            adapter = GoogleAdapter(vertex_ai=True)
            result = await adapter.send_request(adapter_request)

        assert result.response_text == "Vertex response"
        mock_aio.aclose.assert_called_once()
