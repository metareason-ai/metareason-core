import logging

from ollama import AsyncClient, ChatResponse

from .adapter_base import AdapterBase, AdapterException, AdapterRequest, AdapterResponse

logger = logging.getLogger(__name__)


class OllamaException(AdapterException):
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception


class OllamaAdapter(AdapterBase):
    """Adapter for Ollama."""

    def _init(self, **kwargs):
        self.chat_client = AsyncClient()

    async def send_request(self, request: AdapterRequest) -> AdapterResponse:
        try:

            messages = []

            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})

            messages.append({"role": "user", "content": request.user_prompt})

            options = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            }

            response: ChatResponse = await self.chat_client.chat(
                model=request.model, messages=messages, options=options
            )
            logger.info(f"Response from Ollama: {response}")

            return AdapterResponse(response_text=response.message.content)
        except ModuleNotFoundError as e:
            raise OllamaException(
                f"Model {request.model} not found in local Ollama engine: {e}", e
            ) from e
        except Exception as e:
            raise OllamaException(f"Ollama request failed: {e}", e) from e
