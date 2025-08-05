"""Tests for configuration include/import functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from metareason.config.includes import (
    IncludeLoader,
    load_yaml_with_includes,
    merge_configs,
    process_includes_and_inheritance,
    resolve_inheritance,
)


class TestIncludeLoader:
    """Test IncludeLoader class."""

    def test_include_loader_basic(self):
        """Test basic include functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create included file
            included_file = temp_path / "included.yaml"
            included_file.write_text(
                """
shared_value: "from included file"
shared_dict:
  key1: "value1"
  key2: "value2"
"""
            )

            # Create main file with include
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                """
main_value: "from main file"
included_data: !include included.yaml
"""
            )

            # Load with includes
            data = load_yaml_with_includes(main_file)

            assert data["main_value"] == "from main file"
            assert data["included_data"]["shared_value"] == "from included file"
            assert data["included_data"]["shared_dict"]["key1"] == "value1"

    def test_include_relative_path(self):
        """Test include with relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create subdirectory with included file
            subdir = temp_path / "configs"
            subdir.mkdir()

            included_file = subdir / "shared.yaml"
            included_file.write_text("shared: true")

            # Create main file with relative include
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                """
main: true
shared_config: !include configs/shared.yaml
"""
            )

            data = load_yaml_with_includes(main_file)

            assert data["main"] is True
            assert data["shared_config"]["shared"] is True

    def test_include_absolute_path(self):
        """Test include with absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create included file
            included_file = temp_path / "absolute.yaml"
            included_file.write_text("absolute: true")

            # Create main file with absolute include
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                f"""
main: true
absolute_config: !include {included_file.absolute()}
"""
            )

            data = load_yaml_with_includes(main_file)

            assert data["main"] is True
            assert data["absolute_config"]["absolute"] is True

    def test_include_file_not_found(self):
        """Test include with non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            main_file = temp_path / "main.yaml"
            main_file.write_text(
                """
main: true
missing: !include nonexistent.yaml
"""
            )

            with pytest.raises(FileNotFoundError) as exc_info:
                load_yaml_with_includes(main_file)

            assert "Include file not found" in str(exc_info.value)
            assert "nonexistent.yaml" in str(exc_info.value)

    def test_include_circular_dependency(self):
        """Test detection of circular dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create circular includes
            file1 = temp_path / "file1.yaml"
            file1.write_text(
                """
file1: true
file2_data: !include file2.yaml
"""
            )

            file2 = temp_path / "file2.yaml"
            file2.write_text(
                """
file2: true
file1_data: !include file1.yaml
"""
            )

            with pytest.raises(yaml.YAMLError) as exc_info:
                load_yaml_with_includes(file1)

            assert "Circular dependency detected" in str(exc_info.value)

    def test_include_nested_includes(self):
        """Test nested includes (include within included file)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create deeply nested includes
            level3 = temp_path / "level3.yaml"
            level3.write_text("level3: true")

            level2 = temp_path / "level2.yaml"
            level2.write_text(
                """
level2: true
level3_data: !include level3.yaml
"""
            )

            level1 = temp_path / "level1.yaml"
            level1.write_text(
                """
level1: true
level2_data: !include level2.yaml
"""
            )

            data = load_yaml_with_includes(level1)

            assert data["level1"] is True
            assert data["level2_data"]["level2"] is True
            assert data["level2_data"]["level3_data"]["level3"] is True

    def test_include_invalid_yaml(self):
        """Test include with invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML file
            invalid_file = temp_path / "invalid.yaml"
            invalid_file.write_text("invalid: yaml: syntax: [")

            main_file = temp_path / "main.yaml"
            main_file.write_text(
                """
main: true
invalid_data: !include invalid.yaml
"""
            )

            with pytest.raises(yaml.YAMLError) as exc_info:
                load_yaml_with_includes(main_file)

            assert "Failed to parse included file" in str(exc_info.value)


class TestMergeConfigs:
    """Test configuration merging functionality."""

    def test_merge_simple_configs(self):
        """Test merging simple configurations."""
        base = {"key1": "base_value1", "key2": "base_value2"}

        override = {"key2": "override_value2", "key3": "override_value3"}

        result = merge_configs(base, override)

        assert result["key1"] == "base_value1"
        assert result["key2"] == "override_value2"  # Overridden
        assert result["key3"] == "override_value3"

    def test_merge_nested_configs(self):
        """Test merging nested configurations."""
        base = {
            "section1": {"key1": "base_value1", "key2": "base_value2"},
            "section2": {"nested": {"deep": "base_deep"}},
        }

        override = {
            "section1": {"key2": "override_value2", "key3": "override_value3"},
            "section2": {"nested": {"deep": "override_deep", "new": "override_new"}},
            "section3": {"new_section": True},
        }

        result = merge_configs(base, override)

        # Section1 should be merged
        assert result["section1"]["key1"] == "base_value1"
        assert result["section1"]["key2"] == "override_value2"
        assert result["section1"]["key3"] == "override_value3"

        # Section2 nested should be merged
        assert result["section2"]["nested"]["deep"] == "override_deep"
        assert result["section2"]["nested"]["new"] == "override_new"

        # Section3 should be added
        assert result["section3"]["new_section"] is True

    def test_merge_replace_types(self):
        """Test merging when value types change."""
        base = {"key1": "string_value", "key2": {"nested": "dict"}}

        override = {"key1": {"new": "dict"}, "key2": "string_value"}

        result = merge_configs(base, override)

        # Types should be replaced, not merged
        assert result["key1"] == {"new": "dict"}
        assert result["key2"] == "string_value"


class TestResolveInheritance:
    """Test configuration inheritance resolution."""

    def test_resolve_inheritance_basic(self):
        """Test basic inheritance resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create parent configuration
            parent_file = temp_path / "parent.yaml"
            parent_file.write_text(
                """
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
n_variants: 100
"""
            )

            # Create child configuration with inheritance
            child_config = {
                "inherits": str(parent_file),
                "n_variants": 500,  # Override
                "axes": {
                    "name": {
                        "type": "categorical",
                        "values": ["Alice", "Bob", "Charlie"],  # Override
                    },
                    "age": {  # Add new
                        "type": "categorical",
                        "values": ["young", "old"],
                    },
                },
            }

            result = resolve_inheritance(child_config)

            # Should inherit from parent
            assert result["prompt_template"] == "Hello {{name}}"

            # Should override parent values
            assert result["n_variants"] == 500
            assert result["axes"]["name"]["values"] == ["Alice", "Bob", "Charlie"]

            # Should add new values
            assert result["axes"]["age"]["values"] == ["young", "old"]

    def test_resolve_inheritance_extends_keyword(self):
        """Test inheritance with 'extends' keyword."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            parent_file = temp_path / "base.yaml"
            parent_file.write_text("base_value: true")

            child_config = {"extends": str(parent_file), "child_value": True}

            result = resolve_inheritance(child_config)

            assert result["base_value"] is True
            assert result["child_value"] is True
            assert "extends" not in result

    def test_resolve_inheritance_inherit_from_keyword(self):
        """Test inheritance with 'inherit_from' keyword."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            parent_file = temp_path / "base.yaml"
            parent_file.write_text("base_value: true")

            child_config = {"inherit_from": str(parent_file), "child_value": True}

            result = resolve_inheritance(child_config)

            assert result["base_value"] is True
            assert result["child_value"] is True
            assert "inherit_from" not in result

    def test_resolve_inheritance_no_inheritance(self):
        """Test configuration without inheritance."""
        config = {"prompt_id": "test", "value": True}

        result = resolve_inheritance(config)

        # Should return unchanged
        assert result == config

    def test_resolve_inheritance_parent_not_found(self):
        """Test inheritance with missing parent file."""
        child_config = {"inherits": "nonexistent.yaml", "child_value": True}

        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_inheritance(child_config)

        assert "Parent configuration file not found" in str(exc_info.value)

    def test_resolve_inheritance_recursive(self):
        """Test recursive inheritance resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create grandparent
            grandparent_file = temp_path / "grandparent.yaml"
            grandparent_file.write_text(
                """
grandparent: true
value: "grandparent"
"""
            )

            # Create parent that inherits from grandparent
            parent_file = temp_path / "parent.yaml"
            parent_file.write_text(
                f"""
inherits: {grandparent_file}
parent: true
value: "parent"
"""
            )

            # Create child that inherits from parent
            child_config = {
                "inherits": str(parent_file),
                "child": True,
                "value": "child",
            }

            result = resolve_inheritance(child_config)

            # Should have all inherited values
            assert result["grandparent"] is True
            assert result["parent"] is True
            assert result["child"] is True

            # Value should be overridden by child
            assert result["value"] == "child"

    def test_resolve_inheritance_invalid_parent_type(self):
        """Test inheritance with invalid parent configuration type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create parent file with non-dict content
            parent_file = temp_path / "parent.yaml"
            parent_file.write_text("- item1\n- item2")

            child_config = {"inherits": str(parent_file), "child": True}

            with pytest.raises(RuntimeError) as exc_info:
                resolve_inheritance(child_config)

            assert "must be a dictionary" in str(exc_info.value)


class TestProcessIncludesAndInheritance:
    """Test combined includes and inheritance processing."""

    def test_process_includes_and_inheritance_combined(self):
        """Test processing both includes and inheritance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create shared configuration
            shared_file = temp_path / "shared.yaml"
            shared_file.write_text(
                """
shared_axes:
  common:
    type: categorical
    values: ["shared1", "shared2"]
"""
            )

            # Create base configuration with includes
            base_file = temp_path / "base.yaml"
            base_file.write_text(
                f"""
prompt_template: "Base template"
includes: !include {shared_file}
n_variants: 100
"""
            )

            # Create main file with both includes and inheritance
            main_file = temp_path / "main.yaml"
            main_file.write_text(
                f"""
inherits: {base_file}
includes: !include {shared_file}
prompt_template: "Main template"
additional_data: !include {shared_file}
"""
            )

            result = process_includes_and_inheritance(main_file)

            # Should inherit from base
            assert result["n_variants"] == 100

            # Should override inherited values
            assert result["prompt_template"] == "Main template"

            # Should process includes
            assert result["includes"]["shared_axes"]["common"]["values"] == [
                "shared1",
                "shared2",
            ]
            assert result["additional_data"]["shared_axes"]["common"]["values"] == [
                "shared1",
                "shared2",
            ]

    def test_process_includes_and_inheritance_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            process_includes_and_inheritance("nonexistent.yaml")

    def test_process_includes_and_inheritance_invalid_root_type(self):
        """Test error handling for invalid root type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create file with non-dict root
            main_file = temp_path / "main.yaml"
            main_file.write_text("- item1\n- item2")

            with pytest.raises(ValueError) as exc_info:
                process_includes_and_inheritance(main_file)

            assert "must contain a dictionary at the root level" in str(exc_info.value)
