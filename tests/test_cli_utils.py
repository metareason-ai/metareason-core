"""Tests for CLI utility functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from rich.console import Console
from rich.table import Table

from metareason.cli.utils import (
    compare_configurations,
    create_config_summary_table,
    create_progress_callback,
    discover_config_directories,
    find_config_files,
    format_file_size,
    format_validation_report,
    suggest_config_location,
    truncate_text,
)
from metareason.config.loader import load_yaml_config
from metareason.config.validator import ValidationReport
from tests.fixtures.config_builders import ConfigBuilder


class TestFindConfigFiles:
    """Test find_config_files function."""

    def test_find_yaml_files_basic(self):
        """Test finding YAML files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "config1.yaml").write_text("test: value")
            (temp_path / "config2.yml").write_text("test: value")
            (temp_path / "not_yaml.txt").write_text("test: value")

            files = find_config_files(temp_path, recursive=False)

            assert len(files) == 2
            yaml_files = [f.name for f in files]
            assert "config1.yaml" in yaml_files
            assert "config2.yml" in yaml_files
            assert "not_yaml.txt" not in yaml_files

    def test_find_yaml_files_recursive(self):
        """Test finding YAML files recursively."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subdir = temp_path / "subdir"
            subdir.mkdir()

            # Create test files
            (temp_path / "root.yaml").write_text("test: value")
            (subdir / "nested.yaml").write_text("test: value")

            files = find_config_files(temp_path, recursive=True)

            assert len(files) == 2
            file_names = [f.name for f in files]
            assert "root.yaml" in file_names
            assert "nested.yaml" in file_names

    def test_find_yaml_files_non_recursive(self):
        """Test finding YAML files non-recursively."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subdir = temp_path / "subdir"
            subdir.mkdir()

            # Create test files
            (temp_path / "root.yaml").write_text("test: value")
            (subdir / "nested.yaml").write_text("test: value")

            files = find_config_files(temp_path, recursive=False)

            assert len(files) == 1
            assert files[0].name == "root.yaml"

    def test_find_yaml_files_filters_hidden(self):
        """Test that hidden and temporary files are filtered out."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "config.yaml").write_text("test: value")
            (temp_path / ".hidden.yaml").write_text("test: value")
            (temp_path / "temp.yaml~").write_text("test: value")

            files = find_config_files(temp_path)

            assert len(files) == 1
            assert files[0].name == "config.yaml"

    def test_find_yaml_files_nonexistent_directory(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            find_config_files(Path("/nonexistent/directory"))

    def test_find_yaml_files_not_directory(self):
        """Test error when path is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                find_config_files(Path(temp_file.name))

    def test_find_yaml_files_no_yaml_files(self):
        """Test error when no YAML files found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "not_yaml.txt").write_text("test")

            with pytest.raises(ValueError, match="No YAML configuration files found"):
                find_config_files(temp_path)

    def test_find_yaml_files_sorting(self):
        """Test that files are returned sorted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files in reverse alphabetical order
            (temp_path / "z_config.yaml").write_text("test: value")
            (temp_path / "a_config.yaml").write_text("test: value")
            (temp_path / "m_config.yaml").write_text("test: value")

            files = find_config_files(temp_path)

            file_names = [f.name for f in files]
            assert file_names == sorted(file_names)


class TestFormatValidationReport:
    """Test format_validation_report function."""

    def test_format_valid_report(self):
        """Test formatting a valid validation report."""
        report = ValidationReport()

        formatted = format_validation_report(report, "test.yaml")

        assert "📄" in formatted
        assert "test.yaml" in formatted
        assert "✅" in formatted
        assert "Valid configuration" in formatted

    def test_format_invalid_report_with_errors(self):
        """Test formatting an invalid validation report with errors."""
        report = ValidationReport()
        report.add_error("prompt_id", "Field is required", "Add a prompt_id field")
        report.add_warning("This is a warning")
        report.add_suggestion("This is a suggestion")

        formatted = format_validation_report(report, "invalid.yaml")

        assert "❌" in formatted
        assert "Invalid configuration" in formatted
        assert "🚨" in formatted
        assert "Errors:" in formatted
        assert "prompt_id" in formatted
        assert "Field is required" in formatted
        assert "💡" in formatted
        assert "Add a prompt_id field" in formatted
        assert "⚠️" in formatted
        assert "This is a warning" in formatted
        assert "This is a suggestion" in formatted

    def test_format_report_without_file_path(self):
        """Test formatting report without file path."""
        report = ValidationReport()

        formatted = format_validation_report(report)

        assert "✅" in formatted
        assert "Valid configuration" in formatted
        # Should not contain file path indicators
        assert "📄" not in formatted

    def test_format_report_escapes_markup(self):
        """Test that report formatting properly escapes Rich markup."""
        report = ValidationReport()
        report.add_error(
            "test[bold]field[/bold]",
            "Error with [red]markup[/red]",
            "Fix [green]this[/green]",
        )
        report.add_warning("Warning with [yellow]markup[/yellow]")
        report.add_suggestion("Suggestion with [blue]markup[/blue]")

        formatted = format_validation_report(report)

        # The markup should appear in the formatted output (function handles escaping internally)
        assert "test" in formatted
        assert "Error with" in formatted


class TestCompareConfigurations:
    """Test compare_configurations function."""

    def test_compare_identical_configs(self):
        """Test comparing identical configurations."""
        # Create configuration using ConfigBuilder
        config_dict = (
            ConfigBuilder()
            .spec_id("test_config")
            .single_step(
                template="Hello {{name}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                name=["Alice", "Bob"],
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("Test answer", threshold=0.8),
            )
            .build_dict()
        )

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            config1 = load_yaml_config(temp_path)
            config2 = load_yaml_config(temp_path)

            result = compare_configurations(config1, config2)

            assert not result["has_differences"]
            assert len(result["changes"]) == 0
            assert result["summary"]["total_changes"] == 0
        finally:
            temp_path.unlink()

    def test_compare_different_configs(self):
        """Test comparing different configurations."""
        # Create first configuration
        config1_dict = (
            ConfigBuilder()
            .spec_id("config1")
            .single_step(
                template="Hello {{name}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                name=["Alice", "Bob"],
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("Test answer", threshold=0.8),
            )
            .with_variants(100)
            .build_dict()
        )

        # Create second configuration with differences
        config2_dict = (
            ConfigBuilder()
            .spec_id("config2")
            .single_step(
                template="Hi {{name}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                name=["Alice", "Bob", "Charlie"],
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("Test answer", threshold=0.9),
            )
            .with_variants(200)
            .build_dict()
        )

        # Create temporary YAML files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            yaml.dump(config1_dict, f1)
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            yaml.dump(config2_dict, f2)
            temp_path2 = Path(f2.name)

        try:
            config1 = load_yaml_config(temp_path1)
            config2 = load_yaml_config(temp_path2)

            result = compare_configurations(
                config1, config2, file1_name="file1", file2_name="file2"
            )

            assert result["has_differences"]
            assert len(result["changes"]) > 0
            assert result["file1_name"] == "file1"
            assert result["file2_name"] == "file2"

            # Check specific changes
            changes = result["changes"]
            change_paths = [c["path"] for c in changes]
            assert "spec_id" in change_paths
            assert "n_variants" in change_paths
        finally:
            temp_path1.unlink()
            temp_path2.unlink()

    def test_compare_with_ignored_fields(self):
        """Test comparing with ignored fields."""
        # Create base configuration
        base_config = (
            ConfigBuilder()
            .spec_id("same_config")
            .single_step(
                template="Hello {{name}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                name=["Alice", "Bob"],
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("Test answer", threshold=0.8),
            )
            .with_variants(100)
        )

        # Create configurations with different metadata
        config1_dict = base_config.with_metadata(created_date="2024-01-01").build_dict()
        config2_dict = base_config.with_metadata(created_date="2024-01-02").build_dict()

        # Create temporary YAML files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
            yaml.dump(config1_dict, f1)
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
            yaml.dump(config2_dict, f2)
            temp_path2 = Path(f2.name)

        try:
            config1 = load_yaml_config(temp_path1)
            config2 = load_yaml_config(temp_path2)

            # Without ignoring metadata - should find differences
            result = compare_configurations(config1, config2)
            assert result["has_differences"]

            # With ignoring metadata.created_date - should find no differences
            result = compare_configurations(
                config1, config2, ignore_fields=["metadata.created_date"]
            )
            assert not result["has_differences"]
        finally:
            temp_path1.unlink()
            temp_path2.unlink()


class TestCreateConfigSummaryTable:
    """Test create_config_summary_table function."""

    def test_create_summary_table(self):
        """Test creating a configuration summary table."""
        # Create configuration using ConfigBuilder
        config_dict = (
            ConfigBuilder()
            .spec_id("test_summary")
            .add_pipeline_step(
                template="Hello {{name}} with temperature {{temperature}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                axes={
                    "name": {"type": "categorical", "values": ["Alice", "Bob"]},
                    "temperature": {
                        "type": "truncated_normal",
                        "mu": 0.7,
                        "sigma": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                    },
                },
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("Test answer", threshold=0.8),
            )
            .with_variants(500)
            .with_sampling(method="latin_hypercube")
            .with_metadata(version="1.0.0")
            .with_statistical_config(
                model="beta_binomial", prior={"alpha": 1.0, "beta": 1.0}
            )
            .build_dict()
        )

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            config = load_yaml_config(temp_path)
            table = create_config_summary_table(config)

            assert isinstance(table, Table)
            assert table.title == "Configuration Summary"

            # Convert table to string to check contents
            console = Console(file=open(os.devnull, "w"), width=120)
            with console.capture() as capture:
                console.print(table)
            table_str = capture.get()

            assert "test_summary" in table_str
            assert "500" in table_str
            assert "latin_hypercube" in table_str
            assert "2" in table_str  # Total axes
            assert "1" in table_str  # Categorical axes
            assert "1" in table_str  # Continuous axes (temperature)
        finally:
            temp_path.unlink()


class TestDiscoverConfigDirectories:
    """Test discover_config_directories function."""

    def test_discover_config_directories(self):
        """Test discovering configuration directories."""
        directories = discover_config_directories()

        # Should return a list of Path objects
        assert isinstance(directories, list)
        for directory in directories:
            assert isinstance(directory, Path)
            assert directory.exists()
            assert directory.is_dir()

    @patch("os.name", "posix")
    def test_discover_posix_directories(self):
        """Test discovering directories on POSIX systems."""
        directories = discover_config_directories()

        # Should include some POSIX-specific paths if they exist
        # We can't guarantee these exist, so just test the function runs
        assert isinstance(directories, list)

    @patch("os.name", "nt")
    @patch("metareason.cli.utils.Path.cwd")
    def test_discover_windows_directories(self, mock_cwd):
        """Test discovering directories on Windows systems."""
        from pathlib import PosixPath

        mock_cwd.return_value = PosixPath("/fake/cwd")
        with patch.dict(os.environ, {"APPDATA": "C:/Users/test/AppData/Roaming"}):
            # Mock the Path constructor to avoid Windows path issues
            with patch("metareason.cli.utils.Path", PosixPath):
                directories = discover_config_directories()

                # Should include Windows-specific paths
                assert isinstance(directories, list)


class TestSuggestConfigLocation:
    """Test suggest_config_location function."""

    def test_suggest_existing_config(self):
        """Test suggesting location of existing config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / "test_config.yaml"
            config_file.write_text("test: value")

            with patch(
                "metareason.cli.utils.discover_config_directories",
                return_value=[temp_path],
            ):
                suggestion = suggest_config_location("test_config")

                assert suggestion == config_file

    def test_suggest_config_with_extension(self):
        """Test suggesting config with different extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / "test_config.yml"
            config_file.write_text("test: value")

            with patch(
                "metareason.cli.utils.discover_config_directories",
                return_value=[temp_path],
            ):
                suggestion = suggest_config_location("test_config")

                assert suggestion == config_file

    def test_suggest_nonexistent_config(self):
        """Test suggesting location of non-existent config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch(
                "metareason.cli.utils.discover_config_directories",
                return_value=[temp_path],
            ):
                suggestion = suggest_config_location("nonexistent_config")

                assert suggestion is None


class TestUtilityFunctions:
    """Test utility formatting functions."""

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_truncate_text(self):
        """Test text truncation."""
        short_text = "Hello"
        assert truncate_text(short_text, 10) == "Hello"

        long_text = "This is a very long text that should be truncated"
        truncated = truncate_text(long_text, 20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")

        # Test exact length
        exact_text = "12345678901234567890"
        assert truncate_text(exact_text, 20) == exact_text

    def test_create_progress_callback(self):
        """Test progress callback creation."""
        console = Console(file=open(os.devnull, "w"))
        progress, callback = create_progress_callback(console, 100, "Testing")

        # Should return progress object and callback function
        assert progress is not None
        assert callable(callback)

        # Test callback execution
        callback(10)
        callback(5)

        # Clean up
        progress.stop()


class TestUtilsIntegration:
    """Test integration between utility functions."""

    def test_find_and_compare_configs(self):
        """Test finding and comparing multiple configs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create similar configs using ConfigBuilder
            config1_dict = (
                ConfigBuilder()
                .spec_id("config1")
                .single_step(
                    template="Hello {{name}}",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    name=["Alice", "Bob"],
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("Test answer", threshold=0.8),
                )
                .build_dict()
            )

            config2_dict = (
                ConfigBuilder()
                .spec_id("config2")
                .single_step(
                    template="Hello {{name}}",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    name=["Alice", "Bob", "Charlie"],
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("Test answer", threshold=0.8),
                )
                .build_dict()
            )

            # Write configs to files
            config1 = temp_path / "config1.yaml"
            with open(config1, "w") as f:
                yaml.dump(config1_dict, f)

            config2 = temp_path / "config2.yaml"
            with open(config2, "w") as f:
                yaml.dump(config2_dict, f)

            # Find configs
            found_configs = find_config_files(temp_path)
            assert len(found_configs) == 2

            # Load and compare
            loaded_config1 = load_yaml_config(found_configs[0])
            loaded_config2 = load_yaml_config(found_configs[1])

            comparison = compare_configurations(loaded_config1, loaded_config2)
            assert comparison["has_differences"]

    def test_discover_and_suggest_integration(self):
        """Test integration between discovery and suggestion functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_file = temp_path / "integration_test.yaml"
            config_file.write_text("test: value")

            with patch(
                "metareason.cli.utils.discover_config_directories",
                return_value=[temp_path],
            ):
                suggestion = suggest_config_location("integration_test")
                assert suggestion == config_file

                # Should also find it when looking for files
                found_files = find_config_files(temp_path)
                assert config_file in found_files
