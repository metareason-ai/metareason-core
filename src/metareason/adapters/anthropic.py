import logging
import os

from anthropic import AnthropicError, AsyncAnthropic

from .adapter_base import AdapterBase, AdapterException, AdapterRequest, AdapterResponse

logger = logging.getLogger(__name__)


class AnthropicAdapterException(AdapterException):
    """Exception raised when Anthropic adapter operations fail.

    Attributes:
        original_exception: The underlying exception that caused this error, if any
    """

    def __init__(self, msg: str, original_exception: Exception = None):
        super().__init__(msg)
        self.original_exception = original_exception


class AnthropicAdapter(AdapterBase):
    """Adapter for Anthropic's Claude API using the Messages API.

    This adapter provides async communication with Anthropic's Claude models through
    their official AsyncAnthropic client. It supports system prompts, temperature
    control, and token limits.

    Environment Variables:
        ANTHROPIC_API_KEY: Required API key for authentication

    Raises:
        AnthropicAdapterException: If API key is missing or API request fails
    """

    def _init(self, **kwargs) -> None:
        """Initialize the Anthropic client with API credentials.

        Args:
            **kwargs: Additional keyword arguments (currently unused)

        Raises:
            AnthropicAdapterException: If ANTHROPIC_API_KEY environment variable is not set
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AnthropicAdapterException(
                "ANTHROPIC_API_KEY environment variable not set."
            )
        self.client = AsyncAnthropic(api_key=api_key)

    async def send_request(self, request: AdapterRequest) -> AdapterResponse:
        """Send a request to the Anthropic API and return the response.

        Constructs messages from the request's user prompt and sends them to
        Anthropic's Messages API with the specified parameters. System prompts
        are passed via the dedicated system parameter, not in the messages array.

        Args:
            request: AdapterRequest containing model name, prompts, and generation parameters

        Returns:
            AdapterResponse containing the generated text from the model

        Raises:
            AnthropicAdapterException: If the API request fails for any reason
        """
        messages = []
        try:
            messages.append({"role": "user", "content": request.user_prompt})

            response = await self.client.messages.create(
                messages=messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt or "",
            )

            return AdapterResponse(response_text=response.content[0].text)
        except AnthropicError as e:
            raise AnthropicAdapterException(
                f"Anthropic API request failed: {e}", e
            ) from e
