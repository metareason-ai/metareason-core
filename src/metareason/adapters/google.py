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
        """Send a generation request to Google's AI service.

        This method automatically selects the appropriate backend (Vertex AI or
        Developer API) based on the adapter configuration, sends the request,
        and returns the generated response. The client connection is properly
        closed after the request completes.

        Args:
            request: AdapterRequest containing the prompt, model, and generation
                parameters (temperature, max_tokens, etc.).

        Returns:
            AdapterResponse containing the generated text from the model.

        Raises:
            GoogleAdapterException: If the request fails due to API errors,
                network issues, invalid configuration, or missing credentials.

        Example:
            >>> adapter = GoogleAdapter(vertex_ai=True, project_id="my-project", location="us-central1")
            >>> request = AdapterRequest(
            ...     model="gemini-2.0-flash-exp",
            ...     user_prompt="Explain quantum entanglement",
            ...     temperature=0.7,
            ...     top_p=0.9,
            ...     max_tokens=1000
            ... )
            >>> response = await adapter.send_request(request)
            >>> print(response.response_text)
        """
        client = None
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
                    top_p=request.top_p,
                ),
            )
            return AdapterResponse(response_text=response.text)
        except Exception as e:
            raise GoogleAdapterException(
                f"Google adapter request failed: {e}", e
            ) from e
        finally:
            if client is not None:
                await client.aclose()

    def _init_vertex_ai(self):
        """Initialize Google Vertex AI client with project configuration.

        Creates an async client for Google Cloud's Vertex AI service. If the
        GOOGLE_GENAI_USE_VERTEXAI environment variable is set, uses default
        credentials from the environment. Otherwise, requires explicit project_id
        and location in the adapter configuration.

        Returns:
            Async client configured for Vertex AI.

        Raises:
            GoogleAdapterException: If project_id or location is missing when
                GOOGLE_GENAI_USE_VERTEXAI is not set.
        """
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
            return Client(vertexai=True).aio

    def _init_developer_api(self):
        """Initialize Google Developer API client with API key authentication.

        Creates an async client for Google's Developer API (AI Studio). Uses the
        GOOGLE_API_KEY environment variable if set, otherwise requires an api_key
        in the adapter configuration.

        Returns:
            Async client configured for Developer API.

        Raises:
            GoogleAdapterException: If api_key is missing and GOOGLE_API_KEY
                environment variable is not set.
        """
        if not os.getenv("GOOGLE_API_KEY"):
            if not self.config.get("api_key"):
                raise GoogleAdapterException(
                    "api_key is required for the google developer api"
                )
            return Client(api_key=self.config["api_key"]).aio
        else:
            return Client().aio
