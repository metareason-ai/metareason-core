import os

import httpx
from openai import APIConnectionError, AsyncOpenAI, OpenAIError, RateLimitError
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


class OpenAIAdapterException(AdapterException):
    """Exception raised when OpenAI adapter operations fail.

    Attributes:
        original_exception: The underlying exception that caused this error, if any
    """

    def __init__(self, msg: str, original_exception: Exception = None):
        super().__init__(msg)
        self.original_exception = original_exception


class OpenAIAdapter(AdapterBase):
    """Adapter for OpenAI's API using the modern Responses API.

    This adapter provides async communication with OpenAI's models through
    their official AsyncOpenAI client. It supports system prompts, temperature
    control, and token limits.

    Environment Variables:
        OPENAI_API_KEY: Required API key for authentication

    Raises:
        OpenAIAdapterException: If API key is missing or API request fails
    """

    def _init(self, **kwargs) -> None:
        """Initialize the OpenAI client with API credentials.

        Args:
            **kwargs: Additional keyword arguments (currently unused)

        Raises:
            OpenAIAdapterException: If OPENAI_API_KEY environment variable is not set
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIAdapterException("OPENAI_API_KEY environment variable not set.")
        self.client = AsyncOpenAI(
            api_key=api_key, timeout=httpx.Timeout(DEFAULT_TIMEOUT)
        )

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True,
    )
    async def send_request(self, request: AdapterRequest) -> AdapterResponse:
        """Send a request to the OpenAI API and return the response.

        Constructs messages from the request's system and user prompts, then
        sends them to OpenAI's Responses API with the specified parameters.

        Args:
            request: AdapterRequest containing model name, prompts, and generation parameters

        Returns:
            AdapterResponse containing the generated text from the model

        Raises:
            OpenAIAdapterException: If the API request fails for any reason
        """
        messages = []

        if request.system_prompt:
            messages.append({"role": "developer", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.user_prompt})

        try:
            response = await self.client.responses.create(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                top_p=request.top_p,
            )

            return AdapterResponse(response_text=response.output_text)
        except (RateLimitError, APIConnectionError):
            raise
        except OpenAIError as e:
            raise OpenAIAdapterException(f"OpenAI API request failed: {e}", e) from e
