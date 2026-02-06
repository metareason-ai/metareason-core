import logging

import httpx
from ollama import AsyncClient, ChatResponse, RequestError, ResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .adapter_base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    AdapterBase,
    AdapterException,
    AdapterRequest,
    AdapterResponse,
)

logger = logging.getLogger(__name__)


class OllamaException(AdapterException):
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception


class OllamaAdapter(AdapterBase):
    """Adapter for Ollama."""

    def _init(self, **kwargs):
        self.chat_client = AsyncClient(timeout=DEFAULT_TIMEOUT)

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(
            (RequestError, ConnectionError, httpx.ConnectError)
        ),
        reraise=True,
    )
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
        except (RequestError, ResponseError) as e:
            raise OllamaException(f"Ollama request failed: {e}", e) from e
