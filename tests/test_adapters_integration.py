"""Integration test utilities for LLM adapters."""

import asyncio
import os
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from metareason.adapters import (
    AdapterFactory,
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    StreamChunk,
)
from metareason.config.adapters import AdapterType, AnthropicConfig, OpenAIConfig


class AdapterTestSuite:
    """Test suite for adapter implementations."""

    def __init__(self, adapter_config, adapter_name: str):
        """Initialize test suite for an adapter.

        Args:
            adapter_config: Adapter configuration
            adapter_name: Name for the adapter (for test identification)
        """
        self.adapter_config = adapter_config
        self.adapter_name = adapter_name
        self.adapter = None

    async def setup(self):
        """Set up the adapter for testing."""
        try:
            self.adapter = AdapterFactory.create(self.adapter_config)
            await self.adapter.initialize()
            return True
        except Exception as e:
            print(f"Failed to set up {self.adapter_name}: {e}")
            return False

    async def cleanup(self):
        """Clean up the adapter after testing."""
        if self.adapter:
            await self.adapter.cleanup()

    async def test_basic_completion(self) -> Dict[str, Any]:
        """Test basic completion functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            messages = [Message(role=MessageRole.USER, content="Hello, how are you?")]
            request = CompletionRequest(
                messages=messages,
                model=self.adapter_config.default_model,
                temperature=0.7,
                max_tokens=100,
            )

            response = await self.adapter.complete(request)

            assert isinstance(response, CompletionResponse)
            assert len(response.content) > 0
            assert response.model is not None

            return {
                "status": "pass",
                "response_length": len(response.content),
                "model": response.model,
                "finish_reason": response.finish_reason,
            }
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def test_streaming_completion(self) -> Dict[str, Any]:
        """Test streaming completion functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            messages = [Message(role=MessageRole.USER, content="Count from 1 to 5")]
            request = CompletionRequest(
                messages=messages,
                model=self.adapter_config.default_model,
                temperature=0.7,
                max_tokens=50,
                stream=True,
            )

            chunks = []
            async for chunk in self.adapter.complete_stream(request):
                assert isinstance(chunk, StreamChunk)
                chunks.append(chunk)

                # Limit chunks to avoid infinite loops in tests
                if len(chunks) > 50:
                    break

            assert len(chunks) > 0

            # Combine content
            full_content = "".join(chunk.content for chunk in chunks)

            return {
                "status": "pass",
                "chunk_count": len(chunks),
                "total_length": len(full_content),
                "finish_reason": chunks[-1].finish_reason if chunks else None,
            }
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def test_model_listing(self) -> Dict[str, Any]:
        """Test model listing functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            models = await self.adapter.list_models()

            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(model, str) for model in models)

            return {
                "status": "pass",
                "model_count": len(models),
                "models": models[:5],  # Return first 5 for inspection
            }
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def test_model_validation(self) -> Dict[str, Any]:
        """Test model validation functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            # Test with default model (should be valid)
            valid_result = await self.adapter.validate_model(
                self.adapter_config.default_model
            )

            # Test with obviously invalid model
            invalid_result = await self.adapter.validate_model(
                "definitely-not-a-real-model-12345"
            )

            assert valid_result is True
            assert invalid_result is False

            return {
                "status": "pass",
                "default_model_valid": valid_result,
                "invalid_model_valid": invalid_result,
            }
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid requests."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        results = {}

        # Test with invalid model
        try:
            messages = [Message(role=MessageRole.USER, content="Hello")]
            request = CompletionRequest(
                messages=messages, model="invalid-model-name", temperature=0.7
            )

            await self.adapter.complete(request)
            results["invalid_model"] = "unexpected_success"
        except Exception as e:
            results["invalid_model"] = type(e).__name__

        # Test with empty messages (if the adapter validates this)
        try:
            request = CompletionRequest(
                messages=[], model=self.adapter_config.default_model
            )

            await self.adapter.complete(request)
            results["empty_messages"] = "success_or_not_validated"
        except Exception as e:
            results["empty_messages"] = type(e).__name__

        return {"status": "pass", "error_tests": results}

    async def test_cost_estimation(self) -> Dict[str, Any]:
        """Test cost estimation functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            messages = [Message(role=MessageRole.USER, content="Hello, world!")]
            request = CompletionRequest(
                messages=messages,
                model=self.adapter_config.default_model,
                max_tokens=100,
            )

            cost = await self.adapter.estimate_cost(request)

            # Cost can be None if not implemented
            if cost is not None:
                assert isinstance(cost, (int, float))
                assert cost >= 0

            return {
                "status": "pass",
                "cost_estimated": cost is not None,
                "estimated_cost": cost,
            }
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def test_usage_stats(self) -> Dict[str, Any]:
        """Test usage statistics functionality."""
        if not self.adapter:
            return {"status": "skip", "reason": "Adapter not initialized"}

        try:
            stats = await self.adapter.get_usage_stats()

            assert isinstance(stats, dict)

            return {"status": "pass", "stats": stats}
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        if not await self.setup():
            return {"status": "setup_failed"}

        try:
            results = {
                "adapter_name": self.adapter_name,
                "adapter_type": self.adapter_config.type,
            }

            test_methods = [
                ("basic_completion", self.test_basic_completion),
                ("streaming_completion", self.test_streaming_completion),
                ("model_listing", self.test_model_listing),
                ("model_validation", self.test_model_validation),
                ("error_handling", self.test_error_handling),
                ("cost_estimation", self.test_cost_estimation),
                ("usage_stats", self.test_usage_stats),
            ]

            for test_name, test_method in test_methods:
                print(f"Running {test_name} for {self.adapter_name}...")
                results[test_name] = await test_method()

            return results

        finally:
            await self.cleanup()


class MockAdapterTestRunner:
    """Test runner for mock/local testing without API keys."""

    @staticmethod
    def create_mock_configs() -> List[Dict[str, Any]]:
        """Create mock adapter configurations for testing."""
        return [
            {
                "name": "mock_openai",
                "config": OpenAIConfig(
                    type=AdapterType.OPENAI,
                    api_key="mock-key-openai-test",
                    base_url="https://mock-api.example.com/v1",
                    default_model="gpt-3.5-turbo",
                ),
            },
            {
                "name": "mock_anthropic",
                "config": AnthropicConfig(
                    type=AdapterType.ANTHROPIC,
                    api_key="mock-key-anthropic-test",
                    base_url="https://mock-api.example.com/v1",
                    default_model="claude-3-haiku-20240307",
                ),
            },
        ]

    @staticmethod
    async def run_mock_tests():
        """Run tests with mock adapters."""
        results = []

        for adapter_info in MockAdapterTestRunner.create_mock_configs():
            print(f"\n=== Testing {adapter_info['name']} ===")

            # Patch environment variables to provide API keys
            env_patches = {}
            if adapter_info["name"] == "mock_openai":
                env_patches["OPENAI_API_KEY"] = adapter_info["config"].api_key
            elif adapter_info["name"] == "mock_anthropic":
                env_patches["ANTHROPIC_API_KEY"] = adapter_info["config"].api_key

            with (
                patch.dict("os.environ", env_patches),
                patch("aiohttp.ClientSession.request") as mock_request,
            ):
                # Mock successful responses
                MockAdapterTestRunner._setup_mock_responses(mock_request)

                test_suite = AdapterTestSuite(
                    adapter_info["config"], adapter_info["name"]
                )

                test_results = await test_suite.run_all_tests()
                results.append(test_results)

        return results

    @staticmethod
    def _setup_mock_responses(mock_request):
        """Set up mock HTTP responses."""
        from unittest.mock import AsyncMock

        def get_json_response(url, **kwargs):
            url_str = str(url)
            # OpenAI chat completions
            if "chat/completions" in url_str:
                return {
                    "choices": [
                        {
                            "message": {"content": "Mock response from OpenAI API"},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "gpt-3.5-turbo",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            # Anthropic messages
            elif "messages" in url_str:
                return {
                    "content": [
                        {"text": "Mock response from Anthropic API", "type": "text"}
                    ],
                    "model": "claude-3-haiku-20240307",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            # Models listing
            elif "models" in url_str:
                if "anthropic" in url_str.lower() or "claude" in url_str.lower():
                    return {
                        "data": [
                            {"id": "claude-3-haiku-20240307"},
                            {"id": "claude-3-sonnet-20240229"},
                            {"id": "claude-3-opus-20240229"},
                        ]
                    }
                else:
                    return {
                        "data": [
                            {"id": "gpt-3.5-turbo"},
                            {"id": "gpt-4"},
                            {"id": "gpt-4-turbo"},
                        ]
                    }
            # Cost/pricing endpoint
            elif "pricing" in url_str or "cost" in url_str:
                return {
                    "input_cost_per_token": 0.0000015,
                    "output_cost_per_token": 0.000002,
                }
            else:
                return {"result": "success", "status": "ok"}

        # Create a mock that captures the URL from the request() call
        def mock_request_func(method, url, **kwargs):
            # Create a proper async context manager mock
            mock_response = AsyncMock()
            mock_response.status = 200

            async def mock_json():
                return get_json_response(url, **kwargs)

            mock_response.json = mock_json

            # Mock context manager behavior
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_response
            mock_context_manager.__aexit__.return_value = None

            return mock_context_manager

        mock_request.side_effect = mock_request_func


@pytest.mark.integration
class TestAdapterIntegration:
    """Integration tests for real adapters (requires API keys)."""

    def _get_test_configs(self) -> List[Dict[str, Any]]:
        """Get test configurations from environment variables."""
        configs = []

        # OpenAI configuration
        if os.environ.get("OPENAI_API_KEY"):
            configs.append(
                {
                    "name": "openai_gpt35",
                    "config": OpenAIConfig(
                        api_key_env="OPENAI_API_KEY", default_model="gpt-3.5-turbo"
                    ),
                }
            )

        # Anthropic configuration
        if os.environ.get("ANTHROPIC_API_KEY"):
            configs.append(
                {
                    "name": "anthropic_claude",
                    "config": AnthropicConfig(
                        api_key_env="ANTHROPIC_API_KEY",
                        default_model="claude-3-haiku-20240307",  # Cheapest model
                    ),
                }
            )

        return configs

    @pytest.mark.asyncio
    async def test_real_adapters(self):
        """Test real adapters if API keys are available."""
        configs = self._get_test_configs()

        if not configs:
            pytest.skip("No API keys available for integration testing")

        results = []

        for adapter_info in configs:
            print(f"\n=== Testing {adapter_info['name']} ===")

            test_suite = AdapterTestSuite(adapter_info["config"], adapter_info["name"])

            test_results = await test_suite.run_all_tests()
            results.append(test_results)

            # Basic assertions
            assert test_results.get("basic_completion", {}).get("status") in [
                "pass",
                "skip",
            ]
            assert test_results.get("model_listing", {}).get("status") in [
                "pass",
                "skip",
            ]

        # Print summary
        print("\n=== Integration Test Summary ===")
        for result in results:
            print(
                f"{result['adapter_name']}: {result.get('basic_completion', {}).get('status', 'unknown')}"
            )

    @pytest.mark.asyncio
    async def test_openai_adapter_direct(self):
        """Test OpenAI adapter implementation directly with mocks."""
        from metareason.adapters.openai import OpenAIAdapter

        with patch("aiohttp.ClientSession.request") as mock_request:
            # Set up mock responses
            self._setup_openai_mock_responses(mock_request)

            adapter = OpenAIAdapter(
                api_key="test-key-openai",
                base_url="https://mock-api.example.com/v1",
                default_model="gpt-3.5-turbo",
                timeout=5.0,
            )

            async with adapter:
                # Test basic completion
                messages = [Message(role=MessageRole.USER, content="Hello, world!")]
                request = CompletionRequest(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=100,
                )

                response = await adapter.complete(request)

                assert isinstance(response, CompletionResponse)
                assert response.content == "Mock response from OpenAI API"
                assert response.model == "gpt-3.5-turbo"

                # Test model listing
                models = await adapter.list_models()
                assert isinstance(models, list)
                assert "gpt-3.5-turbo" in models

                # Test model validation
                assert await adapter.validate_model("gpt-3.5-turbo") is True
                assert await adapter.validate_model("invalid-model") is False

                # Test cost estimation
                cost = await adapter.estimate_cost(request)
                assert isinstance(cost, (int, float))
                assert cost > 0

                # Test usage stats
                stats = await adapter.get_usage_stats()
                assert isinstance(stats, dict)
                assert "request_count" in stats

    @pytest.mark.asyncio
    async def test_anthropic_adapter_direct(self):
        """Test Anthropic adapter implementation directly with mocks."""
        from metareason.adapters.anthropic import AnthropicAdapter

        with patch("aiohttp.ClientSession.request") as mock_request:
            # Set up mock responses
            self._setup_anthropic_mock_responses(mock_request)

            adapter = AnthropicAdapter(
                api_key="test-key-anthropic",
                base_url="https://mock-api.example.com/v1",
                default_model="claude-3-haiku-20240307",
                api_version="2023-06-01",
                timeout=5.0,
            )

            async with adapter:
                # Test basic completion
                messages = [Message(role=MessageRole.USER, content="Hello, world!")]
                request = CompletionRequest(
                    messages=messages,
                    model="claude-3-haiku-20240307",
                    temperature=0.7,
                    max_tokens=100,
                )

                response = await adapter.complete(request)

                assert isinstance(response, CompletionResponse)
                assert response.content == "Mock response from Anthropic API"
                assert response.model == "claude-3-haiku-20240307"

                # Test model listing
                models = await adapter.list_models()
                assert isinstance(models, list)
                assert "claude-3-haiku-20240307" in models

                # Test model validation
                assert await adapter.validate_model("claude-3-haiku-20240307") is True
                assert await adapter.validate_model("invalid-model") is False

                # Test cost estimation
                cost = await adapter.estimate_cost(request)
                assert isinstance(cost, (int, float))
                assert cost > 0

                # Test usage stats
                stats = await adapter.get_usage_stats()
                assert isinstance(stats, dict)
                assert "request_count" in stats

    @pytest.mark.asyncio
    async def test_http_base_adapter_functionality(self):
        """Test BaseHTTPAdapter functionality with mocks."""
        from metareason.adapters.base import RateLimitError
        from metareason.adapters.http_base import RateLimiter, RetryHandler
        from metareason.config.adapters import RateLimitConfig, RetryConfig

        # Test retry handler
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
        retry_handler = RetryHandler(retry_config)

        attempt_count = 0

        async def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"

        result = await retry_handler.execute(flaky_function)
        assert result == "success"
        assert attempt_count == 2

        # Test rate limiter
        rate_config = RateLimitConfig(
            requests_per_second=10.0, concurrent_requests=2, burst_size=1
        )
        rate_limiter = RateLimiter(rate_config)

        # Test semaphore limiting
        start_time = asyncio.get_event_loop().time()

        async def rate_limited_task():
            async with rate_limiter.acquire():
                await asyncio.sleep(0.05)

        # Run 3 tasks concurrently (should be limited to 2)
        await asyncio.gather(*[rate_limited_task() for _ in range(3)])

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # Should take longer than 0.05s due to concurrency limiting
        assert duration >= 0.08

    @pytest.mark.asyncio
    async def test_adapter_factory_comprehensive(self):
        """Test AdapterFactory with comprehensive coverage."""
        from metareason.adapters.registry import AdapterFactory
        from metareason.config.adapters import (
            AnthropicConfig,
            CustomAdapterConfig,
            OpenAIConfig,
        )

        # Test OpenAI adapter creation with environment variable
        with patch.dict("os.environ", {"TEST_OPENAI_KEY": "test-openai-key"}):
            openai_config = OpenAIConfig(
                api_key_env="TEST_OPENAI_KEY", default_model="gpt-3.5-turbo"
            )

            with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
                # Create a mock adapter class that accepts the config
                mock_adapter_instance = Mock()
                mock_adapter_instance.api_key = "test-openai-key"
                mock_adapter_class = Mock(return_value=mock_adapter_instance)
                mock_get.return_value = mock_adapter_class

                adapter = AdapterFactory.create(openai_config)
                assert adapter is not None
                assert adapter.api_key == "test-openai-key"

        # Test Anthropic adapter creation
        anthropic_config = AnthropicConfig(
            api_key="test-anthropic-key", default_model="claude-3-haiku-20240307"
        )

        with patch("metareason.adapters.registry.get_adapter_class") as mock_get:
            # Create a mock adapter class that accepts the config
            mock_adapter_instance = Mock()
            mock_adapter_instance.api_key = "test-anthropic-key"
            mock_adapter_class = Mock(return_value=mock_adapter_instance)
            mock_get.return_value = mock_adapter_class

            adapter = AdapterFactory.create(anthropic_config)
            assert adapter is not None
            assert adapter.api_key == "test-anthropic-key"

        # Test custom adapter creation
        with patch("importlib.import_module") as mock_import:
            from metareason.adapters.base import LLMAdapter

            # Create a proper mock adapter class that inherits from LLMAdapter
            class MockCustomAdapter(LLMAdapter):
                def __init__(self, config=None, **kwargs):
                    super().__init__(config)
                    self.test_config = config

                async def _initialize(self):
                    pass

                async def _cleanup(self):
                    pass

                async def complete(self, request):
                    from metareason.adapters.base import CompletionResponse

                    return CompletionResponse(content="test", model="test")

                async def complete_stream(self, request):
                    from metareason.adapters.base import StreamChunk

                    yield StreamChunk(content="test")

                async def list_models(self):
                    return ["test-model"]

                async def validate_model(self, model):
                    return True

            mock_module = Mock()
            mock_module.CustomTestAdapter = MockCustomAdapter
            mock_import.return_value = mock_module

            custom_config = CustomAdapterConfig(
                adapter_class="test.module.CustomTestAdapter",
                api_key="test-custom-key",
                custom_params={"param1": "value1"},
            )

            result = AdapterFactory.create(custom_config)

            assert isinstance(result, MockCustomAdapter)
            assert result.test_config["api_key"] == "test-custom-key"
            assert result.test_config["param1"] == "value1"

    def _setup_openai_mock_responses(self, mock_request):
        """Set up OpenAI-specific mock responses."""
        from unittest.mock import AsyncMock

        def create_openai_response(method, url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200

            if "chat/completions" in str(url):

                async def mock_json():
                    return {
                        "choices": [
                            {
                                "message": {"content": "Mock response from OpenAI API"},
                                "finish_reason": "stop",
                            }
                        ],
                        "model": "gpt-3.5-turbo",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }

                mock_response.json = mock_json
            elif "models" in str(url):

                async def mock_json():
                    return {
                        "data": [
                            {"id": "gpt-3.5-turbo"},
                            {"id": "gpt-4"},
                            {"id": "gpt-4-turbo"},
                        ]
                    }

                mock_response.json = mock_json
            else:

                async def mock_json():
                    return {"result": "success"}

                mock_response.json = mock_json

            # Create proper context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_response
            mock_context_manager.__aexit__.return_value = None

            return mock_context_manager

        mock_request.side_effect = create_openai_response

    def _setup_anthropic_mock_responses(self, mock_request):
        """Set up Anthropic-specific mock responses."""
        from unittest.mock import AsyncMock

        def create_anthropic_response(method, url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200

            if "messages" in str(url):

                async def mock_json():
                    return {
                        "content": [
                            {"text": "Mock response from Anthropic API", "type": "text"}
                        ],
                        "model": "claude-3-haiku-20240307",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }

                mock_response.json = mock_json
            elif "models" in str(url):

                async def mock_json():
                    return {
                        "data": [
                            {"id": "claude-3-haiku-20240307"},
                            {"id": "claude-3-sonnet-20240229"},
                            {"id": "claude-3-opus-20240229"},
                        ]
                    }

                mock_response.json = mock_json
            else:

                async def mock_json():
                    return {"result": "success"}

                mock_response.json = mock_json

            # Create proper context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_response
            mock_context_manager.__aexit__.return_value = None

            return mock_context_manager

        mock_request.side_effect = create_anthropic_response

    @pytest.mark.asyncio
    async def test_mock_adapters(self):
        """Test with mock responses (no API keys required)."""
        results = await MockAdapterTestRunner.run_mock_tests()

        assert len(results) >= 2  # At least OpenAI and Anthropic mocks

        for result in results:
            # All tests should pass with mocks
            basic_completion = result.get("basic_completion", {})
            assert basic_completion.get("status") == "pass"

            # Verify other test results
            model_listing = result.get("model_listing", {})
            assert model_listing.get("status") == "pass"

            cost_estimation = result.get("cost_estimation", {})
            assert cost_estimation.get("status") == "pass"

            usage_stats = result.get("usage_stats", {})
            assert usage_stats.get("status") == "pass"


# Utility functions for manual testing
async def run_adapter_manually(adapter_config, adapter_name: str = "manual_test"):
    """Manually test an adapter configuration.

    Usage:
        config = OpenAIConfig(api_key="your-key")
        await run_adapter_manually(config, "my_openai_test")
    """
    test_suite = AdapterTestSuite(adapter_config, adapter_name)
    results = await test_suite.run_all_tests()

    print(f"\n=== Results for {adapter_name} ===")
    for test_name, result in results.items():
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            print(f"{test_name}: {status}")
            if status == "fail":
                print(f"  Error: {result.get('error', 'Unknown')}")
            elif status == "pass" and test_name == "basic_completion":
                print(f"  Response length: {result.get('response_length', 'Unknown')}")

    return results


if __name__ == "__main__":
    # Run mock tests when script is executed directly
    async def main():
        print("Running mock adapter tests...")
        await MockAdapterTestRunner.run_mock_tests()

    asyncio.run(main())
