"""Demo script showing how to load and use MetaReason YAML configurations."""

import os
from pathlib import Path

from metareason.config import load_yaml_config, validate_yaml_file
from metareason.config.cache import get_global_cache
from metareason.config.environment import get_environment_info


def demo_basic_loading():
    """Demonstrate basic configuration loading."""
    print("🔧 Basic Configuration Loading")
    print("=" * 50)

    # Path to the example YAML file
    yaml_file = Path(__file__).parent / "simple_evaluation.yaml"

    print(f"Loading: {yaml_file.name}")

    # Load the configuration
    config = load_yaml_config(yaml_file)

    # Display basic information
    print(f"✅ Prompt ID: {config.prompt_id}")
    print(f"📊 Variants: {config.n_variants}")
    print(f"🎲 Sampling: {config.sampling.method}")
    print(f"📏 Axes: {len(config.axes)}")
    oracle_count = sum(
        [
            1
            for o in [
                config.oracles.accuracy,
                config.oracles.explainability,
                config.oracles.confidence_calibration,
            ]
            if o
        ]
    )
    print(f"🔍 Oracles: {oracle_count}")
    print()


def demo_advanced_features():
    """Demonstrate advanced loading features."""
    print("🚀 Advanced Features Demo")
    print("=" * 50)

    # Set environment variables for demo
    os.environ.update(
        {
            "EVALUATION_NAME": "demo_advanced",
            "SYSTEM_PROMPT": "You are an advanced AI assistant specialized in technical analysis.",
            "N_VARIANTS": "1500",
            "ACCURACY_THRESHOLD": "0.85",
            "USER": "demo_user",
            "COMPANY": "metareason",
        }
    )

    yaml_file = Path(__file__).parent / "advanced_evaluation.yaml"

    print(f"Loading: {yaml_file.name}")
    print("Features: inheritance, includes, environment variables")

    try:
        # Load with all features enabled
        config = load_yaml_config(
            yaml_file, enable_includes=True, enable_env_substitution=True
        )

        print("✅ Successfully loaded with inheritance and includes")
        print(f"📋 Prompt ID: {config.prompt_id}")
        print(f"📊 Variants: {config.n_variants}")
        print("🌍 Environment substitution: ✅")
        print("📄 Inheritance: ✅ (from base_config.yaml)")
        print("📎 Includes: ✅ (shared_oracles.yaml)")

        # Show some inherited/substituted values
        print(f"🏢 Company: {config.metadata.created_by}")
        print(f"📅 Created: {config.metadata.created_date}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: This is expected if included files are not found")

    print()


def demo_environment_variables():
    """Demonstrate environment variable substitution."""
    print("🌍 Environment Variable Demo")
    print("=" * 50)

    # Set some demo environment variables
    os.environ.update(
        {
            "PROMPT_ID": "env_demo_test",
            "TASK1": "text_analysis",
            "TASK2": "data_processing",
            "ACCURACY_THRESHOLD": "0.9",
            "JUDGE_MODEL": "gpt-4-turbo",
        }
    )

    yaml_file = Path(__file__).parent / "environment_demo.yaml"

    if yaml_file.exists():
        print(f"Loading: {yaml_file.name}")

        try:
            config = load_yaml_config(
                yaml_file, enable_env_substitution=True, env_strict=False
            )

            print("✅ Environment variables substituted successfully")
            print(f"📋 Prompt ID: {config.prompt_id}")
            print(f"🎯 Accuracy threshold: {config.oracles.accuracy.threshold}")
            print(f"⚖️  Judge model: {config.oracles.explainability.judge_model}")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️  Environment demo file not found, skipping...")

    print()


def demo_caching():
    """Demonstrate configuration caching."""
    print("💾 Configuration Caching Demo")
    print("=" * 50)

    yaml_file = Path(__file__).parent / "simple_evaluation.yaml"
    cache = get_global_cache()

    print(f"Initial cache size: {cache.size()}")

    # First load - should cache
    print("First load (should cache)...")
    config1 = load_yaml_config(yaml_file, use_cache=True)
    print(f"Cache size after first load: {cache.size()}")

    # Second load - should use cache
    print("Second load (should use cache)...")
    config2 = load_yaml_config(yaml_file, use_cache=True)
    print(f"Cache size after second load: {cache.size()}")
    print(f"Same config object: {config1 is config2}")

    # Show cache stats
    stats = cache.get_stats()
    print(
        f"📊 Cache stats: {stats['active_entries']} active, TTL: {stats['ttl_seconds']}s"
    )

    print()


def demo_validation():
    """Demonstrate configuration validation."""
    print("✅ Configuration Validation")
    print("=" * 50)

    yaml_file = Path(__file__).parent / "simple_evaluation.yaml"

    print(f"Validating: {yaml_file.name}")

    # Validate the configuration
    config, report = validate_yaml_file(yaml_file)

    if report.is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has issues:")

    # Show warnings and suggestions
    if report.warnings:
        print(f"⚠️  Warnings: {len(report.warnings)}")
        for warning in report.warnings[:3]:  # Show first 3
            print(f"   • {warning}")

    if report.suggestions:
        print(f"💡 Suggestions: {len(report.suggestions)}")
        for suggestion in report.suggestions[:3]:  # Show first 3
            print(f"   • {suggestion}")

    print()


def demo_environment_info():
    """Show environment information."""
    print("🌍 Environment Information")
    print("=" * 50)

    env_info = get_environment_info()

    print(f"Total environment variables: {env_info['total_env_vars']}")
    print(f"MetaReason variables: {len(env_info['metareason_vars'])}")

    # Show some common variables
    common_vars = env_info["common_config_vars"]
    for var in ["USER", "HOME", "PATH"]:
        if var in common_vars:
            value = common_vars[var]
            if var == "PATH":
                value = f"{value[:50]}..." if len(value) > 50 else value
            print(f"  {var}: {value}")

    print()


def main():
    """Run all demonstrations."""
    print("🎯 MetaReason Configuration Loading Demo")
    print("=" * 60)
    print()

    # Run each demo
    demo_basic_loading()
    demo_caching()
    demo_validation()
    demo_environment_info()
    demo_advanced_features()
    demo_environment_variables()

    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("  • Try the CLI: metareason config validate examples/")
    print("  • Create your own configuration files")
    print("  • Experiment with environment variables and includes")


if __name__ == "__main__":
    main()
