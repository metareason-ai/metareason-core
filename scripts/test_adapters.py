#!/usr/bin/env python3
"""Script to test LLM adapters with real or mock data."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path to import metareason
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metareason.config.adapters import (  # noqa: E402
    AnthropicConfig,
    AzureOpenAIConfig,
    OpenAIConfig,
)
from tests.test_adapters_integration import (  # noqa: E402
    MockAdapterTestRunner,
    test_adapter_manually,
)


def print_banner(text: str):
    """Print a banner with the given text."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_test_result(test_name: str, result: Dict[str, Any], indent: int = 2):
    """Print formatted test result."""
    spaces = " " * indent
    status = result.get("status", "unknown")

    if status == "pass":
        print(f"{spaces}✅ {test_name}: PASS")
        # Show additional info for some tests
        if test_name == "basic_completion":
            length = result.get("response_length", 0)
            model = result.get("model", "unknown")
            print(f"{spaces}   Response: {length} chars, Model: {model}")
        elif test_name == "model_listing":
            count = result.get("model_count", 0)
            print(f"{spaces}   Found {count} models")
        elif test_name == "cost_estimation":
            cost = result.get("estimated_cost")
            if cost is not None:
                print(f"{spaces}   Estimated cost: ${cost:.4f}")
    elif status == "fail":
        print(f"{spaces}❌ {test_name}: FAIL")
        error = result.get("error", "Unknown error")
        print(f"{spaces}   Error: {error}")
    elif status == "skip":
        print(f"{spaces}⏭️  {test_name}: SKIP")
        reason = result.get("reason", "Unknown reason")
        print(f"{spaces}   Reason: {reason}")
    else:
        print(f"{spaces}❓ {test_name}: {status.upper()}")


async def test_openai_adapter(api_key: Optional[str] = None):
    """Test OpenAI adapter."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return None

    config = OpenAIConfig(
        api_key=api_key, default_model="gpt-3.5-turbo"  # Cheapest model
    )

    print("🚀 Testing OpenAI adapter...")
    return await test_adapter_manually(config, "openai_test")


async def test_anthropic_adapter(api_key: Optional[str] = None):
    """Test Anthropic adapter."""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print(
            "❌ Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
        )
        return None

    config = AnthropicConfig(
        api_key=api_key, default_model="claude-3-haiku-20240307"  # Cheapest model
    )

    print("🚀 Testing Anthropic adapter...")
    return await test_adapter_manually(config, "anthropic_test")


async def test_azure_openai_adapter(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
):
    """Test Azure OpenAI adapter."""
    if not api_key:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not deployment:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    if not all([api_key, endpoint, deployment]):
        print("❌ Azure OpenAI credentials not complete.")
        print(
            "   Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT"
        )
        return None

    config = AzureOpenAIConfig(
        api_key=api_key, azure_endpoint=endpoint, azure_deployment=deployment
    )

    print("🚀 Testing Azure OpenAI adapter...")
    return await test_adapter_manually(config, "azure_openai_test")


async def run_mock_tests():
    """Run tests with mock responses."""
    print_banner("Mock Adapter Tests (No API Keys Required)")

    results = await MockAdapterTestRunner.run_mock_tests()

    print("\n📊 Mock Test Results:")
    for result in results:
        adapter_name = result.get("adapter_name", "unknown")
        print(f"\n🔧 {adapter_name}:")

        for key, value in result.items():
            if key not in ["adapter_name", "adapter_type"] and isinstance(value, dict):
                print_test_result(key, value)

    return results


async def run_real_tests():
    """Run tests with real API calls."""
    print_banner("Real Adapter Tests (API Keys Required)")

    results = []

    # Test OpenAI
    openai_result = await test_openai_adapter()
    if openai_result:
        results.append(openai_result)
        print("\n📊 OpenAI Test Results:")
        for key, value in openai_result.items():
            if key not in ["adapter_name", "adapter_type"] and isinstance(value, dict):
                print_test_result(key, value)

    print()  # Spacing

    # Test Anthropic
    anthropic_result = await test_anthropic_adapter()
    if anthropic_result:
        results.append(anthropic_result)
        print("\n📊 Anthropic Test Results:")
        for key, value in anthropic_result.items():
            if key not in ["adapter_name", "adapter_type"] and isinstance(value, dict):
                print_test_result(key, value)

    print()  # Spacing

    # Test Azure OpenAI
    azure_result = await test_azure_openai_adapter()
    if azure_result:
        results.append(azure_result)
        print("\n📊 Azure OpenAI Test Results:")
        for key, value in azure_result.items():
            if key not in ["adapter_name", "adapter_type"] and isinstance(value, dict):
                print_test_result(key, value)

    if not results:
        print("❌ No adapters were tested. Make sure API keys are set.")

    return results


def print_summary(mock_results, real_results):
    """Print test summary."""
    print_banner("Test Summary")

    print("📋 Mock Tests:")
    if mock_results:
        for result in mock_results:
            adapter_name = result.get("adapter_name", "unknown")
            basic_status = result.get("basic_completion", {}).get("status", "unknown")
            print(f"  {adapter_name}: {basic_status}")
    else:
        print("  No mock tests run")

    print("\n📋 Real Tests:")
    if real_results:
        for result in real_results:
            adapter_name = result.get("adapter_name", "unknown")
            basic_status = result.get("basic_completion", {}).get("status", "unknown")
            print(f"  {adapter_name}: {basic_status}")
    else:
        print("  No real tests run")


async def main():
    """Main test function."""
    print_banner("MetaReason LLM Adapter Test Suite")

    import argparse

    parser = argparse.ArgumentParser(description="Test LLM adapters")
    parser.add_argument(
        "--mode",
        choices=["mock", "real", "both"],
        default="both",
        help="Test mode: mock (no API keys), real (with API keys), or both",
    )
    parser.add_argument("--output", help="Output results to JSON file")

    args = parser.parse_args()

    mock_results = []
    real_results = []

    try:
        if args.mode in ["mock", "both"]:
            mock_results = await run_mock_tests()

        if args.mode in ["real", "both"]:
            real_results = await run_real_tests()

        print_summary(mock_results, real_results)

        # Save results to file if requested
        if args.output:
            all_results = {
                "mock_tests": mock_results,
                "real_tests": real_results,
                "summary": {
                    "mock_count": len(mock_results),
                    "real_count": len(real_results),
                    "total_tests": len(mock_results) + len(real_results),
                },
            }

            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            print(f"\n💾 Results saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
