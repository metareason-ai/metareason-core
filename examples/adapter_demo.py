#!/usr/bin/env python3
"""Demonstration of the MetaReason LLM Adapter System.

This example shows how to use the adapter system with different providers.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metareason.adapters import (  # noqa: E402
    AdapterFactory,
    CompletionRequest,
    Message,
    MessageRole,
)
from metareason.config.adapters import (  # noqa: E402
    AdaptersConfig,
    AnthropicConfig,
    OpenAIConfig,
)


async def demo_basic_usage():
    """Demonstrate basic adapter usage."""
    print("=== Basic Adapter Usage Demo ===\n")

    # Example with mock API key (won't make real calls)
    config = OpenAIConfig(api_key="mock-key-for-demo", default_model="gpt-3.5-turbo")

    try:
        # Create adapter
        adapter = AdapterFactory.create(config)
        print(f"✅ Created adapter: {adapter}")

        # Create a completion request
        messages = [Message(role=MessageRole.USER, content="Hello, how are you?")]
        request = CompletionRequest(
            messages=messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=100
        )

        print(f"📝 Request: {request.messages[0].content}")
        print(
            f"🎛️  Settings: temp={request.temperature}, max_tokens={request.max_tokens}"
        )

        # Note: This would normally make an API call, but will fail with mock key
        # In real usage, you would set a valid API key

    except Exception as e:
        print(f"⚠️  Expected error with mock key: {e}")

    print()


async def demo_configuration_types():
    """Demonstrate different configuration types."""
    print("=== Configuration Types Demo ===\n")

    configs = {
        "OpenAI": OpenAIConfig(
            api_key="mock-openai-key",
            organization_id="org-example",
            default_model="gpt-4",
        ),
        "Anthropic": AnthropicConfig(
            api_key="mock-anthropic-key", default_model="claude-3-sonnet-20240229"
        ),
    }

    for name, config in configs.items():
        print(f"{name} Configuration:")
        print(f"  Type: {config.type}")
        print(f"  Model: {config.default_model}")
        print(f"  Base URL: {config.base_url}")
        print()


async def demo_multiple_adapters():
    """Demonstrate managing multiple adapters."""
    print("=== Multiple Adapters Demo ===\n")

    adapters_config = AdaptersConfig(
        default_adapter="openai",
        adapters={
            "openai": OpenAIConfig(
                api_key="mock-openai-key", default_model="gpt-3.5-turbo"
            ),
            "anthropic": AnthropicConfig(
                api_key="mock-anthropic-key", default_model="claude-3-haiku-20240307"
            ),
        },
    )

    print(f"Default adapter: {adapters_config.default_adapter}")
    print(f"Available adapters: {list(adapters_config.adapters.keys())}")

    # Get specific adapter configs
    openai_config = adapters_config.get_adapter_config("openai")
    anthropic_config = adapters_config.get_adapter_config("anthropic")

    print(f"OpenAI model: {openai_config.default_model}")
    print(f"Anthropic model: {anthropic_config.default_model}")
    print()


async def demo_error_handling():
    """Demonstrate error handling."""
    print("=== Error Handling Demo ===\n")

    from metareason.adapters.base import (
        AuthenticationError,
        ModelNotFoundError,
        ProviderError,
        RateLimitError,
    )

    print("The adapter system provides comprehensive error types:")
    print(f"- RateLimitError: {RateLimitError.__doc__}")
    print(f"- AuthenticationError: {AuthenticationError.__doc__}")
    print(f"- ModelNotFoundError: {ModelNotFoundError.__doc__}")
    print(f"- ProviderError: {ProviderError.__doc__}")
    print()


async def demo_real_adapter_if_available():
    """Demonstrate real adapter if API keys are available."""
    print("=== Real Adapter Demo ===\n")

    # Check for real API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("⚠️  No API keys found in environment variables.")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test real adapters.")
        return

    if openai_key:
        print("🔑 Found OpenAI API key - testing real adapter...")
        config = OpenAIConfig(api_key=openai_key, default_model="gpt-3.5-turbo")

        try:
            adapter = AdapterFactory.create(config)

            async with adapter:
                # List available models
                models = await adapter.list_models()
                print(f"   Available models: {models[:3]}...")

                # Validate default model
                is_valid = await adapter.validate_model(config.default_model)
                print(f"   Model '{config.default_model}' valid: {is_valid}")

                # Estimate cost for a small request
                messages = [Message(role=MessageRole.USER, content="Hi")]
                request = CompletionRequest(
                    messages=messages, model=config.default_model, max_tokens=10
                )
                cost = await adapter.estimate_cost(request)
                if cost:
                    print(f"   Estimated cost for small request: ${cost:.6f}")

                print("   ✅ OpenAI adapter working correctly!")

        except Exception as e:
            print(f"   ❌ OpenAI adapter error: {e}")

    if anthropic_key:
        print("\n🔑 Found Anthropic API key - testing real adapter...")
        config = AnthropicConfig(
            api_key=anthropic_key,
            default_model="claude-3-haiku-20240307",  # Cheapest model
        )

        try:
            adapter = AdapterFactory.create(config)

            async with adapter:
                # List available models
                models = await adapter.list_models()
                print(f"   Available models: {models}")

                # Validate default model
                is_valid = await adapter.validate_model(config.default_model)
                print(f"   Model '{config.default_model}' valid: {is_valid}")

                print("   ✅ Anthropic adapter working correctly!")

        except Exception as e:
            print(f"   ❌ Anthropic adapter error: {e}")


async def main():
    """Run all demos."""
    print("🚀 MetaReason LLM Adapter System Demo\n")

    await demo_basic_usage()
    await demo_configuration_types()
    await demo_multiple_adapters()
    await demo_error_handling()
    await demo_real_adapter_if_available()

    print("🎉 Demo complete! The LLM Adapter MVP is ready for use.")
    print("\nNext steps:")
    print("- Set API keys as environment variables to test real adapters")
    print("- Run the test suite: pytest tests/test_adapters_*.py")
    print("- See docs/adapters.md for detailed usage documentation")


if __name__ == "__main__":
    asyncio.run(main())
