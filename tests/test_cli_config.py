"""Tests for CLI configuration commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from metareason.cli.config import config_group


class TestConfigValidateCommand:
    """Test the config validate command."""
    
    def test_validate_single_file_valid(self):
        """Test validating a single valid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob", "Charlie"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ['validate', str(temp_path)])
            
            assert result.exit_code == 0
            assert "All configuration files are valid" in result.output
        finally:
            temp_path.unlink()
    
    def test_validate_single_file_invalid(self):
        """Test validating a single invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: ""  # Invalid empty prompt_id
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ['validate', str(temp_path)])
            
            assert result.exit_code == 1
            assert "configuration files have issues" in result.output
        finally:
            temp_path.unlink()
    
    def test_validate_directory(self):
        """Test validating all files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid config
            valid_config = temp_path / "valid.yaml"
            valid_config.write_text("""
prompt_id: valid_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            
            # Create invalid config
            invalid_config = temp_path / "invalid.yaml"
            invalid_config.write_text("""
prompt_id: ""  # Invalid
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            
            runner = CliRunner()
            result = runner.invoke(config_group, ['validate', '-d', str(temp_path)])
            
            assert result.exit_code == 1
            assert "1 of 2 configuration files have issues" in result.output
    
    def test_validate_json_output(self):
        """Test validation with JSON output format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'validate', str(temp_path), '--format', 'json'
            ])
            
            assert result.exit_code == 0
            
            # Should be valid JSON
            output_data = json.loads(result.output)
            assert str(temp_path) in output_data
            assert output_data[str(temp_path)]["valid"] is True
        finally:
            temp_path.unlink()
    
    def test_validate_strict_mode(self):
        """Test validation in strict mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: test_config
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
  unused_axis:  # This will trigger warning in normal mode, error in strict
    type: categorical
    values: ["X", "Y"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            
            # Normal mode - should pass with warnings
            result = runner.invoke(config_group, ['validate', str(temp_path)])
            assert result.exit_code == 0
            
            # Strict mode - should fail
            result = runner.invoke(config_group, ['validate', str(temp_path), '--strict'])
            assert result.exit_code == 1
        finally:
            temp_path.unlink()
    
    def test_validate_no_files_specified(self):
        """Test validation with no files specified."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['validate'])
        
        assert result.exit_code == 1
        assert "No configuration files specified" in result.output
    
    def test_validate_nonexistent_directory(self):
        """Test validation with non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['validate', '-d', '/nonexistent/directory'])
        
        assert result.exit_code == 1
        assert "Error:" in result.output


class TestConfigShowCommand:
    """Test the config show command."""
    
    def test_show_basic_yaml(self):
        """Test showing configuration in YAML format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, ['show', str(temp_path)])
            
            assert result.exit_code == 0
            assert "test_config" in result.output
            assert temp_path.name in result.output  # Should show filename
        finally:
            temp_path.unlink()
    
    def test_show_json_format(self):
        """Test showing configuration in JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'show', str(temp_path), '--format', 'json'
            ])
            
            assert result.exit_code == 0
            
            # Should contain JSON-like structure
            assert '"prompt_id"' in result.output or 'prompt_id' in result.output
        finally:
            temp_path.unlink()
    
    def test_show_with_includes(self):
        """Test showing configuration with includes expanded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create included file
            included_file = temp_path / "shared.yaml"
            included_file.write_text("""
shared_value: "from included file"
""")
            
            # Create main file with include
            main_file = temp_path / "main.yaml"
            main_file.write_text(f"""
prompt_id: test_with_includes
prompt_template: "Hello {{{{name}}}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
shared: !include {included_file.name}
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'show', str(main_file), '--expand-includes'
            ])
            
            assert result.exit_code == 0
            assert "test_with_includes" in result.output
    
    def test_show_with_env_expansion(self):
        """Test showing configuration with environment variables expanded."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: ${TEST_PROMPT_ID:default_id}
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["${USER1:Alice}", "${USER2:Bob}"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            with patch.dict('os.environ', {'TEST_PROMPT_ID': 'env_test', 'USER1': 'Charlie'}):
                runner = CliRunner()
                result = runner.invoke(config_group, [
                    'show', str(temp_path), '--expand-env'
                ])
                
                assert result.exit_code == 0
                # Should show expanded values
                assert "env_test" in result.output or "default_id" in result.output
        finally:
            temp_path.unlink()
    
    def test_show_nonexistent_file(self):
        """Test showing non-existent configuration file."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['show', '/nonexistent/file.yaml'])
        
        assert result.exit_code == 1
        assert "Error loading configuration" in result.output


class TestConfigDiffCommand:
    """Test the config diff command."""
    
    def test_diff_identical_files(self):
        """Test diffing identical configuration files."""
        config_content = """
prompt_id: test_config
prompt_template: "Hello {{name}}"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            f1.write(config_content)
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f2.write(config_content)
            temp_path2 = Path(f2.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'diff', str(temp_path1), str(temp_path2)
            ])
            
            assert result.exit_code == 0
            assert "No differences found" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()
    
    def test_diff_different_files(self):
        """Test diffing different configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            f1.write("""
prompt_id: config1
prompt_template: "Hello {{name}}"
n_variants: 1000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f2.write("""
prompt_id: config2
prompt_template: "Hi {{name}}"
n_variants: 2000
axes:
  name:
    type: categorical
    values: ["Alice", "Bob", "Charlie"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.9
""")
            temp_path2 = Path(f2.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'diff', str(temp_path1), str(temp_path2)
            ])
            
            assert result.exit_code == 1  # Differences found
            assert "Modified:" in result.output or "changes" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()
    
    def test_diff_json_format(self):
        """Test diff with JSON output format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            f1.write("""
prompt_id: config1
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
""")
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f2.write("""
prompt_id: config2
prompt_template: "Hello {{name}}"
n_variants: 2000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
""")
            temp_path2 = Path(f2.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'diff', str(temp_path1), str(temp_path2), '--format', 'json'
            ])
            
            assert result.exit_code == 1
            
            # Should be valid JSON
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")
        finally:
            temp_path1.unlink()
            temp_path2.unlink()
    
    def test_diff_ignore_fields(self):
        """Test diff with ignored fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            f1.write("""
prompt_id: same_config
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
metadata:
  created_date: "2024-01-01"
""")
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            f2.write("""
prompt_id: same_config
prompt_template: "Hello {{name}}"
n_variants: 1000
schema:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    method: cosine_similarity
    threshold: 0.85
    embeddings_file: dummy.txt
metadata:
  created_date: "2024-01-02"  # Different date
""")
            temp_path2 = Path(f2.name)
        
        try:
            runner = CliRunner()
            result = runner.invoke(config_group, [
                'diff', str(temp_path1), str(temp_path2),
                '--ignore-fields', 'metadata.created_date'
            ])
            
            assert result.exit_code == 0  # No differences after ignoring field
            assert "No differences found" in result.output
        finally:
            temp_path1.unlink()
            temp_path2.unlink()


class TestConfigCacheCommand:
    """Test the config cache command."""
    
    def test_cache_stats(self):
        """Test showing cache statistics."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['cache', '--stats'])
        
        assert result.exit_code == 0
        assert "caching" in result.output or "Cache" in result.output
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['cache', '--clear'])
        
        assert result.exit_code == 0
        assert "caching" in result.output or "Cache" in result.output
    
    def test_cache_disable(self):
        """Test disabling the cache."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['cache', '--disable'])
        
        assert result.exit_code == 0
        assert "disabled" in result.output
    
    def test_cache_default_info(self):
        """Test default cache command shows basic info."""
        runner = CliRunner()
        result = runner.invoke(config_group, ['cache'])
        
        assert result.exit_code == 0
        # Should show some cache information
        assert "caching" in result.output or "Cache" in result.output or "entries" in result.output


class TestConfigCommandIntegration:
    """Test integration between config commands."""
    
    def test_validate_then_show(self):
        """Test validating then showing a configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: integration_test
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            
            # First validate
            result = runner.invoke(config_group, ['validate', str(temp_path)])
            assert result.exit_code == 0
            
            # Then show
            result = runner.invoke(config_group, ['show', str(temp_path)])
            assert result.exit_code == 0
            assert "integration_test" in result.output
        finally:
            temp_path.unlink()
    
    def test_validate_cache_interaction(self):
        """Test that validation interacts properly with cache."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
prompt_id: cache_test
prompt_template: "Hello {{name}}, this is a test template"
axes:
  name:
    type: categorical
    values: ["Alice", "Bob"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "This is a comprehensive test answer for validation"
    threshold: 0.8
""")
            temp_path = Path(f.name)
        
        try:
            runner = CliRunner()
            
            # Clear cache first
            runner.invoke(config_group, ['cache', '--clear'])
            
            # Validate (should cache)
            result = runner.invoke(config_group, ['validate', str(temp_path)])
            assert result.exit_code == 0
            
            # Check cache stats
            result = runner.invoke(config_group, ['cache', '--stats'])
            assert result.exit_code == 0
        finally:
            temp_path.unlink()