"""Tests for enhanced configuration loader functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from metareason.config.cache import get_global_cache, set_global_cache
from metareason.config.environment import EnvironmentSubstitutionError
from metareason.config.loader import load_yaml_config, load_yaml_configs
from tests.fixtures.config_builders import ConfigBuilder


def create_valid_test_config(spec_id: str = "test_config") -> str:
    """Create a valid test configuration YAML string using ConfigBuilder."""
    config_builder = ConfigBuilder()
    config = (
        config_builder.spec_id(spec_id)
        .single_step(
            template="Hello {{name}}",
            adapter="openai",
            model="gpt-3.5-turbo",
            name=["Alice", "Bob"],
        )
        .with_oracle(
            "accuracy",
            lambda o: o.embedding_similarity(
                "This is a comprehensive test answer for validation", threshold=0.8
            ),
        )
        .to_yaml()
    )
    return config


class TestEnhancedLoader:
    """Test enhanced configuration loader functionality."""

    def setup_method(self):
        """Reset cache before each test."""
        set_global_cache(None)

    def test_load_yaml_config_with_caching(self):
        """Test configuration loading with caching enabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(create_valid_test_config("test_config"))
            temp_path = Path(f.name)

        try:
            # First load - should cache
            config1 = load_yaml_config(temp_path, use_cache=True)
            cache = get_global_cache()
            assert cache.size() == 1

            # Second load - should use cache
            config2 = load_yaml_config(temp_path, use_cache=True)
            assert config1.spec_id == config2.spec_id
            assert cache.size() == 1  # Still only one entry
        finally:
            temp_path.unlink()

    def test_load_yaml_config_without_caching(self):
        """Test configuration loading with caching disabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(create_valid_test_config("test_config"))
            temp_path = Path(f.name)

        try:
            # Load without caching
            config = load_yaml_config(temp_path, use_cache=False)
            cache = get_global_cache()
            assert cache.size() == 0  # Should not be cached
            assert config.spec_id == "test_config"
        finally:
            temp_path.unlink()

    def test_load_yaml_config_with_includes(self):
        """Test configuration loading with includes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create included file
            included_file = temp_path / "shared.yaml"
            included_file.write_text(
                """
accuracy:
  type: embedding_similarity
  canonical_answer: "This is a shared oracle configuration"
  threshold: 0.85
"""
            )

            # Create main file with include using proper pipeline format
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                f"""
spec_id: test_with_includes
pipeline:
  - template: "Hello {{{{name}}}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
oracles: !include {included_file.name}
"""
            )

            config = load_yaml_config(main_file, enable_includes=True)

            assert config.spec_id == "test_with_includes"
            assert config.oracles.accuracy.threshold == 0.85
            assert "shared oracle" in config.oracles.accuracy.canonical_answer

    def test_load_yaml_config_with_environment_substitution(self):
        """Test configuration loading with environment variable substitution."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: ${PROMPT_ID}
pipeline:
  - template: "Hello ${NAME:World}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      name:
        type: categorical
        values: ["${USER1}", "${USER2}"]
n_variants: ${VARIANTS:1000}
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: ${THRESHOLD:0.8}
"""
            )
            temp_path = Path(f.name)

        try:
            with patch.dict(
                os.environ,
                {
                    "PROMPT_ID": "env_test",
                    "USER1": "Alice",
                    "USER2": "Bob",
                    "VARIANTS": "500",
                },
            ):
                config = load_yaml_config(temp_path, enable_env_substitution=True)

                assert config.spec_id == "env_test"
                assert config.pipeline[0].template == "Hello World"  # Default used
                assert config.pipeline[0].axes["name"].values == ["Alice", "Bob"]
                assert config.n_variants == 500
                assert config.oracles.accuracy.threshold == 0.8  # Default used
        finally:
            temp_path.unlink()

    def test_load_yaml_config_env_strict_mode(self):
        """Test environment substitution in strict mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test_strict
pipeline:
  - template: "Missing: ${MISSING_VAR}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      name:
        type: categorical
        values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            with pytest.raises(EnvironmentSubstitutionError):
                load_yaml_config(
                    temp_path, enable_env_substitution=True, env_strict=True
                )
        finally:
            temp_path.unlink()

    def test_load_yaml_config_search_paths(self):
        """Test configuration loading with search paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "configs"
            config_dir.mkdir()

            # Create config in subdirectory
            config_file = config_dir / "test.yaml"
            config_file.write_text(create_valid_test_config("search_path_test"))

            # Load using search paths
            config = load_yaml_config("test.yaml", search_paths=[str(config_dir)])

            assert config.spec_id == "search_path_test"

    def test_load_yaml_config_invalid_extension(self):
        """Test loading file with invalid extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                load_yaml_config(temp_path)

            assert "Invalid file extension" in str(exc_info.value)
        finally:
            temp_path.unlink()

    def test_load_yaml_config_permission_error(self):
        """Test handling of permission errors."""
        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: content")
            temp_path = Path(f.name)

        try:
            # Remove read permissions
            os.chmod(temp_path, 0o000)

            with pytest.raises(PermissionError) as exc_info:
                load_yaml_config(temp_path)

            assert "Permission denied" in str(exc_info.value)
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            temp_path.unlink()

    def test_load_yaml_config_includes_fallback(self):
        """Test fallback to standard YAML loading when includes fail."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: fallback_test
pipeline:
  - template: "Hello {{name}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      name:
        type: categorical
        values: ["Alice", "Bob"]
broken_include: !include nonexistent.yaml
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )
            temp_path = Path(f.name)

        try:
            # Should fallback to standard loading (without processing includes)
            with pytest.raises(
                yaml.YAMLError
            ):  # Will fail on YAML parsing due to broken include
                load_yaml_config(temp_path, enable_includes=True)
        finally:
            temp_path.unlink()

    def test_load_yaml_config_yaml_error_with_line_numbers(self):
        """Test YAML error reporting with line numbers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
spec_id: test
invalid_yaml: [
missing_closing_bracket: true
"""
            )
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception) as exc_info:
                load_yaml_config(temp_path)

            # Should include error details
            error_msg = str(exc_info.value)
            assert "Failed to parse YAML file" in error_msg
        finally:
            temp_path.unlink()

    def test_load_yaml_configs_directory(self):
        """Test loading all configs from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple config files
            config1 = temp_path / "config1.yaml"
            config1.write_text(create_valid_test_config("config1"))

            config2 = temp_path / "config2.yml"
            config2.write_text(create_valid_test_config("config2"))

            # Load all configs
            configs = load_yaml_configs(temp_path)

            assert len(configs) == 2
            assert "config1" in configs
            assert "config2" in configs
            assert configs["config1"].spec_id == "config1"
            assert configs["config2"].spec_id == "config2"

    def test_load_yaml_configs_with_errors(self):
        """Test loading configs when some have errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid config
            valid_config = temp_path / "valid.yaml"
            valid_config.write_text(create_valid_test_config("valid_config"))

            # Create invalid config (missing required pipeline field)
            invalid_config = temp_path / "invalid.yaml"
            invalid_config.write_text(
                """
spec_id: ""  # Invalid empty spec_id
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
            )

            # Should still return valid configs
            configs = load_yaml_configs(temp_path)
            assert "valid" in configs
            # Invalid config should not be included
            assert len(configs) == 1


class TestPathResolution:
    """Test configuration file path resolution."""

    def test_absolute_path_resolution(self):
        """Test resolution of absolute paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(create_valid_test_config("absolute_path"))
            temp_path = Path(f.name)

        try:
            # Use absolute path
            config = load_yaml_config(temp_path.absolute())
            assert config.spec_id == "absolute_path"
        finally:
            temp_path.unlink()

    def test_relative_path_resolution(self):
        """Test resolution of relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            original_cwd = Path.cwd()

            try:
                # Change to temp directory
                os.chdir(temp_path)

                # Create config file
                config_file = temp_path / "relative.yaml"
                config_file.write_text(create_valid_test_config("relative_path"))

                # Load using relative path
                config = load_yaml_config("relative.yaml")
                assert config.spec_id == "relative_path"
            finally:
                os.chdir(original_cwd)

    def test_search_path_priority(self):
        """Test search path priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config in multiple locations
            search1 = temp_path / "search1"
            search1.mkdir()
            config1 = search1 / "test.yaml"
            config1.write_text(create_valid_test_config("from_search1"))

            search2 = temp_path / "search2"
            search2.mkdir()
            config2 = search2 / "test.yaml"
            config2.write_text(create_valid_test_config("from_search2"))

            # Should find first one in search path order
            config = load_yaml_config(
                "test.yaml", search_paths=[str(search1), str(search2)]
            )
            assert config.spec_id == "from_search1"

    def test_file_not_found_in_search_paths(self):
        """Test file not found in any search paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            search_dir = temp_path / "search"
            search_dir.mkdir()

            with pytest.raises(FileNotFoundError) as exc_info:
                load_yaml_config("nonexistent.yaml", search_paths=[str(search_dir)])

            assert "not found" in str(exc_info.value)
            assert "Searched in:" in str(exc_info.value)
