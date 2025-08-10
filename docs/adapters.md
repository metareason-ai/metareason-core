# LLM Adapter System

The MetaReason LLM adapter system provides a unified interface for interacting with different Large Language Model providers. This system supports multiple providers, retry logic, rate limiting, streaming responses, and cost estimation.

## Quick Start

### Basic Usage

```python
import asyncio
from metareason.adapters import AdapterFactory
from metareason.adapters.base import Message, MessageRole, CompletionRequest
from metareason.config.adapters import OpenAIConfig

async def main():
    # Configure adapter
    config = OpenAIConfig(
        api_key="your-openai-api-key",
        default_model="gpt-3.5-turbo"
    )

    # Create adapter
    adapter = AdapterFactory.create(config)

    # Use adapter
    async with adapter:
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        request = CompletionRequest(messages=messages, model="gpt-3.5-turbo")

        response = await adapter.complete(request)
        print(response.content)

asyncio.run(main())
```

### Configuration via YAML

```yaml
adapters:
  default_adapter: "openai"
  adapters:
    openai:
      type: "openai"
      api_key_env: "OPENAI_API_KEY"
      default_model: "gpt-3.5-turbo"
      retry:
        max_retries: 3
        initial_delay: 1.0
      rate_limit:
        requests_per_minute: 3000
        concurrent_requests: 10

    anthropic:
      type: "anthropic"
      api_key_env: "ANTHROPIC_API_KEY"
      default_model: "claude-3-sonnet-20240229"
```

## Supported Providers

### OpenAI

Supports GPT-3.5, GPT-4, and other OpenAI models.

```python
from metareason.config.adapters import OpenAIConfig

config = OpenAIConfig(
    api_key="sk-your-key",
    organization_id="org-your-org",  # Optional
    default_model="gpt-4",
    base_url="https://api.openai.com/v1"  # Default
)
```

**Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key

**Supported Models:**
- `gpt-4-turbo`, `gpt-4`, `gpt-4-32k`
- `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`

### Anthropic

Supports Claude 3 and Claude 2 models.

```python
from metareason.config.adapters import AnthropicConfig

config = AnthropicConfig(
    api_key="sk-ant-your-key",
    default_model="claude-3-sonnet-20240229",
    api_version="2023-06-01"  # Default
)
```

**Environment Variables:**
- `ANTHROPIC_API_KEY`: Your Anthropic API key

**Supported Models:**
- `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- `claude-2.1`, `claude-2.0`, `claude-instant-1.2`

### Azure OpenAI

Supports OpenAI models via Azure.

```python
from metareason.config.adapters import AzureOpenAIConfig

config = AzureOpenAIConfig(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="your-deployment-name",
    api_version="2024-02-01"
)
```

**Environment Variables:**
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure endpoint URL
- `AZURE_OPENAI_DEPLOYMENT`: Your deployment name

### HuggingFace

Supports HuggingFace models via Inference API or local inference.

```python
from metareason.config.adapters import HuggingFaceConfig

config = HuggingFaceConfig(
    model_id="microsoft/DialoGPT-medium",
    api_key="hf_your_token",  # Optional for public models
    use_inference_api=True,
    inference_endpoint="https://api-inference.huggingface.co"
)
```

### Custom Adapters

Create your own adapter by implementing the `LLMAdapter` interface.

```python
from metareason.config.adapters import CustomAdapterConfig

config = CustomAdapterConfig(
    adapter_class="myproject.adapters.MyCustomAdapter",
    api_key="custom-key",
    custom_params={
        "param1": "value1",
        "param2": 123
    }
)
```

## Core Concepts

### Messages

All communication with LLMs uses the `Message` format:

```python
from metareason.adapters.base import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="Hello!"),
    Message(role=MessageRole.ASSISTANT, content="Hi there!")
]
```

### Completion Requests

Requests are structured with full parameter support:

```python
from metareason.adapters.base import CompletionRequest

request = CompletionRequest(
    messages=messages,
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["END"],
    stream=False
)
```

### Responses

Responses include content, metadata, and usage information:

```python
response = await adapter.complete(request)

print(f"Content: {response.content}")
print(f"Model: {response.model}")
print(f"Finish reason: {response.finish_reason}")
print(f"Usage: {response.usage}")
```

## Advanced Features

### Streaming Responses

Get real-time streaming responses:

```python
request = CompletionRequest(messages=messages, model="gpt-3.5-turbo", stream=True)

async for chunk in adapter.complete_stream(request):
    print(chunk.content, end="", flush=True)
    if chunk.finish_reason:
        print(f"\nFinished: {chunk.finish_reason}")
```

### Batch Processing

Process multiple requests concurrently:

```python
requests = [
    CompletionRequest(messages=[Message(role=MessageRole.USER, content=f"Question {i}")], model="gpt-3.5-turbo")
    for i in range(5)
]

responses = await adapter.batch_complete(requests)
for i, response in enumerate(responses):
    if isinstance(response, Exception):
        print(f"Request {i} failed: {response}")
    else:
        print(f"Request {i}: {response.content}")
```

### Error Handling

The system provides comprehensive error handling:

```python
from metareason.adapters.base import (
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ProviderError
)

try:
    response = await adapter.complete(request)
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except ProviderError as e:
    print(f"Provider error: {e}")
```

### Retry Logic

Configure automatic retries with exponential backoff:

```python
from metareason.config.adapters import RetryConfig

retry_config = RetryConfig(
    max_retries=5,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

config = OpenAIConfig(
    api_key="your-key",
    retry=retry_config
)
```

### Rate Limiting

Control request rates to stay within API limits:

```python
from metareason.config.adapters import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_minute=1000,    # Or requests_per_second
    concurrent_requests=5,       # Max concurrent requests
    burst_size=10               # Token bucket size
)

config = OpenAIConfig(
    api_key="your-key",
    rate_limit=rate_config
)
```

### Cost Estimation

Estimate costs before making requests:

```python
estimated_cost = await adapter.estimate_cost(request)
if estimated_cost and estimated_cost > 0.10:  # $0.10 limit
    print(f"Request would cost ${estimated_cost:.4f}")
    if input("Continue? (y/n): ").lower() != 'y':
        return

response = await adapter.complete(request)
```

### Usage Statistics

Monitor adapter usage:

```python
stats = await adapter.get_usage_stats()
print(f"Requests made: {stats.get('request_count', 0)}")
print(f"Error count: {stats.get('error_count', 0)}")
print(f"Error rate: {stats.get('error_rate', 0):.2%}")
```

## Model Management

### Listing Available Models

```python
models = await adapter.list_models()
print(f"Available models: {models}")
```

### Validating Models

```python
is_valid = await adapter.validate_model("gpt-4")
if not is_valid:
    print("Model not available")
```

## Configuration Management

### Multiple Adapters

Manage multiple adapters in one configuration:

```python
from metareason.config.adapters import AdaptersConfig

config = AdaptersConfig(
    default_adapter="openai",
    adapters={
        "openai": OpenAIConfig(api_key="openai-key"),
        "anthropic": AnthropicConfig(api_key="anthropic-key"),
        "azure": AzureOpenAIConfig(
            api_key="azure-key",
            azure_endpoint="https://your.openai.azure.com",
            azure_deployment="gpt-4"
        )
    }
)

# Get specific adapter config
openai_config = config.get_adapter_config("openai")
adapter = AdapterFactory.create(openai_config)
```

### Environment-Based Configuration

Use environment variables for sensitive data:

```python
config = OpenAIConfig(
    api_key_env="OPENAI_API_KEY",  # Will read from environment
    default_model="gpt-3.5-turbo"
)
```

## Testing

### Unit Testing

Run the included unit tests:

```bash
# Activate virtual environment
source venv/bin/activate

# Run adapter tests
pytest tests/test_adapters_*.py -v
```

### Integration Testing

Test with real APIs (requires API keys):

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Run integration tests
python scripts/test_adapters.py --mode real

# Or run mock tests (no API keys needed)
python scripts/test_adapters.py --mode mock
```

### Custom Testing

Test your configurations manually:

```python
from tests.test_adapters_integration import test_adapter_manually

config = OpenAIConfig(api_key="your-key")
results = await test_adapter_manually(config, "my_test")
```

## Creating Custom Adapters

### Implement the Interface

```python
from metareason.adapters.base import LLMAdapter, CompletionRequest, CompletionResponse
from metareason.adapters.http_base import BaseHTTPAdapter

class MyCustomAdapter(BaseHTTPAdapter):
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Format request for your API
        payload = self._format_request(request)

        # Make API call
        response = await self._request("POST", "completions", json_data=payload)

        # Format response
        return CompletionResponse(
            content=response["text"],
            model=request.model,
            usage=response.get("usage")
        )

    async def complete_stream(self, request):
        # Implement streaming if supported
        async for chunk in self._stream_request("POST", "completions", json_data=payload):
            yield StreamChunk(content=chunk.get("text", ""))

    async def list_models(self):
        response = await self._request("GET", "models")
        return [model["id"] for model in response["models"]]

    async def validate_model(self, model: str) -> bool:
        models = await self.list_models()
        return model in models
```

### Register the Adapter

```python
from metareason.adapters import register_adapter

register_adapter("my_custom", MyCustomAdapter)
```

### Use Custom Configuration

```python
from metareason.config.adapters import CustomAdapterConfig

config = CustomAdapterConfig(
    adapter_class="myproject.adapters.MyCustomAdapter",
    api_key="your-key",
    base_url="https://api.myprovider.com/v1",
    custom_params={
        "special_param": "value"
    }
)

adapter = AdapterFactory.create(config)
```

## Best Practices

### Security

1. **Never hardcode API keys** - use environment variables
2. **Use least-privilege access** - limit API key permissions
3. **Rotate keys regularly** - especially in production
4. **Monitor usage** - watch for unexpected usage patterns

### Performance

1. **Use appropriate rate limits** - don't overwhelm APIs
2. **Implement caching** - cache responses when appropriate
3. **Use streaming** - for real-time applications
4. **Batch requests** - when possible

### Error Handling

1. **Handle all error types** - implement proper error recovery
2. **Log errors appropriately** - but don't log sensitive data
3. **Use circuit breakers** - for failing services
4. **Implement graceful degradation** - fallback strategies

### Cost Management

1. **Estimate costs** - before expensive operations
2. **Set usage limits** - prevent runaway costs
3. **Monitor spending** - track API usage
4. **Choose appropriate models** - balance cost vs. performance

## Troubleshooting

### Common Issues

**Authentication Errors:**
- Check API key validity
- Verify environment variables are set
- Ensure API key has required permissions

**Rate Limiting:**
- Reduce `requests_per_minute` setting
- Increase `initial_delay` in retry config
- Use fewer `concurrent_requests`

**Model Not Found:**
- Check model name spelling
- Verify model is available in your region
- Use `list_models()` to see available options

**Timeout Errors:**
- Increase `timeout` setting
- Check network connectivity
- Try smaller `max_tokens` values

### Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("metareason.adapters")
logger.setLevel(logging.DEBUG)
```

### Getting Help

1. Check the error message and type
2. Review configuration parameters
3. Test with mock responses first
4. Use the integration test suite
5. Check API provider documentation
6. Open an issue with detailed error information

## API Reference

See the inline documentation in the source code for detailed API reference:

- `metareason.adapters.base` - Core interfaces and data types
- `metareason.adapters.http_base` - HTTP adapter base class
- `metareason.adapters.registry` - Adapter factory and registry
- `metareason.config.adapters` - Configuration models
- Individual adapter modules for provider-specific implementations
