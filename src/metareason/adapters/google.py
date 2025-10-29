import logging
import os

from google.genai import Client, types

from .adapter_base import AdapterBase, AdapterException, AdapterRequest, AdapterResponse

logger = logging.getLogger(__name__)


class GoogleAdapterException(AdapterException):
    """Exception raised for Google adapter-specific errors.

    Attributes:
        message: Human-readable error description.
        original_exception: The underlying exception that triggered this error.
    """

    def __init__(self, message: str, original_exception: Exception = None):
        """Initialize GoogleAdapterException.

        Args:
            message: Error message describing what went wrong.
            original_exception: Original exception that caused this error.
        """
        super().__init__(message)
        self.original_exception = original_exception


class GoogleAdapter(AdapterBase):
    """Adapter for Google's generative AI services (Vertex AI and Developer API).

    This adapter provides a unified interface to Google's AI services, automatically
    selecting between Vertex AI (for Google Cloud users) and the Developer API
    based on available credentials and configuration.

    The adapter supports:
    - Vertex AI with project/location configuration
    - Google Developer API with API key authentication
    - Automatic credential detection from environment variables
    - Async request handling

    Configuration is provided via kwargs in __init__:
    - vertex_ai: Whether to use Vertex AI backend (default: False)
    - project_id: Google Cloud project ID (required for Vertex AI)
    - location: Google Cloud region (required for Vertex AI)
    - api_key: API key for Developer API

    Environment variables:
    - GOOGLE_GENAI_USE_VERTEXAI: Set to use Vertex AI with default credentials
    - GOOGLE_API_KEY: API key for Developer API (if not provided in config)
    """

    def __init__(self, **kwargs):
        """Initialize the Google adapter with configuration.

        Args:
            **kwargs: Configuration parameters passed to _init method.
        """
        super().__init__(**kwargs)

    def _init(self, **kwargs):
        """Initialize adapter configuration.

        Args:
            **kwargs: Configuration parameters including:
                - vertex_ai: Whether to use Vertex AI (default: False)
                - project_id: Google Cloud project ID (Vertex AI)
                - location: Google Cloud region (Vertex AI)
                - api_key: API key for Developer API
        """
        self.vertex_ai = kwargs.get("vertex_ai")
        self.config = kwargs

    async def send_request(self, request: AdapterRequest) -> AdapterResponse:
        """Send a request to the google adapter (vertex ai or developer api)."""
        try:
            if self.vertex_ai:
                client = self._init_vertex_ai()
            else:
                client = self._init_developer_api()

            response = await client.models.generate_content(
                model=request.model,
                contents=request.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature,
                ),
            )
            await client.aclose()
            return AdapterResponse(response_text=response.text)
        except Exception as e:
            raise GoogleAdapterException(
                f"Google adapter request failed: {e}", e
            ) from e

    def _init_vertex_ai(self):
        """Initialize the google vertex ai client."""
        if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            if not self.config.get("project_id"):
                raise GoogleAdapterException(
                    "project_id is required for the google vertex ai adapter"
                )
            if not self.config.get("location"):
                raise GoogleAdapterException(
                    "location is required for the google vertex ai adapter"
                )
            return Client(
                vertexai=True,
                project=self.config["project_id"],
                location=self.config["location"],
            ).aio
        else:
            return Client().aio

    def _init_developer_api(self):
        """Initialize the google developer api client."""
        if not os.getenv("GOOGLE_API_KEY"):
            if not self.config.get("api_key"):
                raise GoogleAdapterException(
                    "api_key is required for the google developer api"
                )
            return Client(api_key=self.config["api_key"]).aio
        else:
            return Client().aio
