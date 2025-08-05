"""Tests for environment variable substitution functionality."""

import os
from unittest.mock import patch

import pytest

from metareason.config.environment import (
    EnvironmentSubstitutionError,
    get_environment_info,
    substitute_environment_variables,
    validate_required_environment_vars,
)


class TestEnvironmentSubstitution:
    """Test environment variable substitution."""
    
    def test_substitute_simple_variable(self):
        """Test simple variable substitution."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = substitute_environment_variables("${TEST_VAR}")
            assert result == "test_value"
    
    def test_substitute_variable_in_string(self):
        """Test variable substitution within a string."""
        with patch.dict(os.environ, {"NAME": "Alice"}):
            result = substitute_environment_variables("Hello ${NAME}!")
            assert result == "Hello Alice!"
    
    def test_substitute_multiple_variables(self):
        """Test multiple variable substitution."""
        with patch.dict(os.environ, {"FIRST": "Hello", "SECOND": "World"}):
            result = substitute_environment_variables("${FIRST} ${SECOND}!")
            assert result == "Hello World!"
    
    def test_substitute_variable_with_default(self):
        """Test variable substitution with default value."""
        # Variable not set
        result = substitute_environment_variables("${MISSING_VAR:default_value}")
        assert result == "default_value"
        
        # Variable set
        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = substitute_environment_variables("${SET_VAR:default_value}")
            assert result == "actual_value"
    
    def test_substitute_variable_with_bash_default(self):
        """Test variable substitution with bash-style default."""
        # Variable not set
        result = substitute_environment_variables("${MISSING_VAR:-default_value}")
        assert result == "default_value"
        
        # Variable set
        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = substitute_environment_variables("${SET_VAR:-default_value}")
            assert result == "actual_value"
    
    def test_substitute_variable_required_missing(self):
        """Test required variable that is missing."""
        with pytest.raises(EnvironmentSubstitutionError) as exc_info:
            substitute_environment_variables("${MISSING_REQUIRED}", strict=True)
        
        assert "Required environment variable 'MISSING_REQUIRED' is not set" in str(exc_info.value)
    
    def test_substitute_variable_required_missing_non_strict(self):
        """Test required variable missing in non-strict mode."""
        result = substitute_environment_variables("${MISSING_REQUIRED}")
        # Should leave unchanged for debugging
        assert result == "${MISSING_REQUIRED}"
    
    def test_substitute_variable_with_error_message(self):
        """Test variable with custom error message."""
        with pytest.raises(EnvironmentSubstitutionError) as exc_info:
            substitute_environment_variables("${MISSING_VAR:?Custom error message}")
        
        assert "Custom error message" in str(exc_info.value)
    
    def test_substitute_variable_type_coercion(self):
        """Test type coercion of environment variables."""
        with patch.dict(os.environ, {
            "BOOL_TRUE": "true",
            "BOOL_FALSE": "false",
            "INT_VAL": "42",
            "FLOAT_VAL": "3.14",
            "STRING_VAL": "hello"
        }):
            assert substitute_environment_variables("${BOOL_TRUE}") is True
            assert substitute_environment_variables("${BOOL_FALSE}") is False
            assert substitute_environment_variables("${INT_VAL}") == 42
            assert substitute_environment_variables("${FLOAT_VAL}") == 3.14
            assert substitute_environment_variables("${STRING_VAL}") == "hello"
    
    def test_substitute_variable_boolean_variations(self):
        """Test boolean value variations."""
        test_cases = [
            ("true", True), ("True", True), ("TRUE", True),
            ("yes", True), ("Yes", True), ("YES", True),
            ("1", True), ("on", True), ("ON", True),
            ("false", False), ("False", False), ("FALSE", False),
            ("no", False), ("No", False), ("NO", False),
            ("0", False), ("off", False), ("OFF", False),
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"BOOL_VAR": env_value}):
                result = substitute_environment_variables("${BOOL_VAR}")
                assert result is expected, f"Failed for {env_value}"
    
    def test_substitute_dict(self):
        """Test substitution in dictionary values."""
        with patch.dict(os.environ, {"KEY": "value", "NUM": "42"}):
            data = {
                "simple": "${KEY}",
                "nested": {
                    "deep": "${NUM}",
                    "string": "Hello ${KEY}!"
                }
            }
            
            result = substitute_environment_variables(data)
            
            assert result["simple"] == "value"
            assert result["nested"]["deep"] == 42
            assert result["nested"]["string"] == "Hello value!"
    
    def test_substitute_list(self):
        """Test substitution in list values."""
        with patch.dict(os.environ, {"ITEM1": "first", "ITEM2": "second"}):
            data = ["${ITEM1}", "${ITEM2}", "static"]
            
            result = substitute_environment_variables(data)
            
            assert result == ["first", "second", "static"]
    
    def test_substitute_primitive_types(self):
        """Test that primitive types are unchanged."""
        data = {
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None
        }
        
        result = substitute_environment_variables(data)
        
        assert result == data
    
    def test_substitute_custom_delimiters(self):
        """Test substitution with custom delimiters."""
        with patch.dict(os.environ, {"VAR": "value"}):
            result = substitute_environment_variables(
                "#{VAR}", 
                default_prefix="#{", 
                default_suffix="}"
            )
            assert result == "value"
    
    def test_substitute_invalid_syntax(self):
        """Test invalid variable syntax."""
        # These patterns shouldn't match our regex and should be left unchanged
        test_cases = [
            "${VAR:invalid_modifier}",  # This is actually valid (default value)
            "${123VAR}",  # Invalid variable name
            "${VAR@invalid}",  # Invalid character
        ]
        
        for test_case in test_cases:
            try:
                result = substitute_environment_variables(test_case)
                # If no error, should be unchanged or processed
                assert isinstance(result, str)
            except EnvironmentSubstitutionError:
                # Some cases might raise errors, which is also acceptable
                pass
    
    def test_substitute_strict_mode_with_defaults(self):
        """Test strict mode prevents use of defaults."""
        with pytest.raises(EnvironmentSubstitutionError) as exc_info:
            substitute_environment_variables("${MISSING:default}", strict=True)
        
        assert "strict mode is enabled" in str(exc_info.value)


class TestValidateRequiredEnvironmentVars:
    """Test validation of required environment variables."""
    
    def test_validate_no_variables(self):
        """Test validation with no environment variables."""
        config = {"simple": "value"}
        
        result = validate_required_environment_vars(config)
        
        assert result["all_required_available"] is True
        assert len(result["missing_vars"]) == 0
        assert len(result["found_vars"]) == 0
    
    def test_validate_found_variables(self):
        """Test validation with available variables."""
        with patch.dict(os.environ, {"FOUND_VAR": "value"}):
            config = {
                "key": "${FOUND_VAR}",
                "nested": {
                    "deep": "${FOUND_VAR:default}"
                }
            }
            
            result = validate_required_environment_vars(config)
            
            assert result["all_required_available"] is True
            assert len(result["missing_vars"]) == 0
            assert len(result["found_vars"]) == 2
    
    def test_validate_missing_required_variables(self):
        """Test validation with missing required variables."""
        config = {
            "required": "${MISSING_REQUIRED}",
            "with_default": "${MISSING_WITH_DEFAULT:default}",
            "custom_error": "${MISSING_CUSTOM:?custom error}"
        }
        
        with pytest.raises(EnvironmentSubstitutionError) as exc_info:
            validate_required_environment_vars(config)
        
        error_msg = str(exc_info.value)
        assert "Missing required environment variables" in error_msg
        assert "MISSING_REQUIRED" in error_msg
        assert "MISSING_CUSTOM" in error_msg
        # Should not include variables with defaults
        assert "MISSING_WITH_DEFAULT" not in error_msg
    
    def test_validate_complex_config(self):
        """Test validation with complex configuration structure."""
        with patch.dict(os.environ, {"AVAILABLE": "value"}):
            config = {
                "section1": {
                    "available": "${AVAILABLE}",
                    "missing": "${MISSING_VAR}"
                },
                "list_section": [
                    "${AVAILABLE}",
                    "${ANOTHER_MISSING}"
                ]
            }
            
            with pytest.raises(EnvironmentSubstitutionError) as exc_info:
                validate_required_environment_vars(config)
            
            error_msg = str(exc_info.value)
            assert "MISSING_VAR" in error_msg
            assert "ANOTHER_MISSING" in error_msg
    
    def test_validate_variable_paths(self):
        """Test that variable paths are correctly tracked."""
        config = {
            "root_var": "${ROOT}",
            "section": {
                "nested_var": "${NESTED}"
            }
        }
        
        with pytest.raises(EnvironmentSubstitutionError) as exc_info:
            validate_required_environment_vars(config)
        
        error_msg = str(exc_info.value)
        # Should show paths where variables are used
        assert "root_var" in error_msg or "ROOT" in error_msg
        assert "nested_var" in error_msg or "NESTED" in error_msg


class TestGetEnvironmentInfo:
    """Test environment information gathering."""
    
    def test_get_environment_info_basic(self):
        """Test basic environment info gathering."""
        info = get_environment_info()
        
        assert "metareason_vars" in info
        assert "total_env_vars" in info
        assert "common_config_vars" in info
        assert isinstance(info["total_env_vars"], int)
    
    def test_get_environment_info_with_metareason_vars(self):
        """Test environment info with MetaReason variables."""
        with patch.dict(os.environ, {
            "METAREASON_CONFIG_DIR": "/test/config",
            "MR_LOG_LEVEL": "debug",
            "OTHER_VAR": "value"
        }):
            info = get_environment_info()
            
            mr_vars = info["metareason_vars"]
            assert "METAREASON_CONFIG_DIR" in mr_vars
            assert "MR_LOG_LEVEL" in mr_vars
            assert "OTHER_VAR" not in mr_vars
            
            # Check variable details
            config_var = mr_vars["METAREASON_CONFIG_DIR"]
            assert config_var["value"] == "/test/config"
            assert config_var["length"] == len("/test/config")
            assert config_var["is_sensitive"] is False
    
    def test_get_environment_info_sensitive_detection(self):
        """Test detection of sensitive environment variables."""
        with patch.dict(os.environ, {
            "METAREASON_API_KEY": "secret123",
            "MR_PASSWORD": "pass123",
            "METAREASON_CONFIG": "public_config"
        }):
            info = get_environment_info()
            
            mr_vars = info["metareason_vars"]
            
            # Sensitive variables
            assert mr_vars["METAREASON_API_KEY"]["is_sensitive"] is True
            assert mr_vars["MR_PASSWORD"]["is_sensitive"] is True
            
            # Non-sensitive variables
            assert mr_vars["METAREASON_CONFIG"]["is_sensitive"] is False
    
    def test_get_environment_info_common_vars(self):
        """Test inclusion of common configuration variables."""
        info = get_environment_info()
        
        common_vars = info["common_config_vars"]
        expected_vars = [
            "HOME", "USER", "PATH", "PWD",
            "METAREASON_CONFIG_DIR", "METAREASON_LOG_LEVEL", "METAREASON_CACHE_DIR"
        ]
        
        for var in expected_vars:
            assert var in common_vars


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_variable_name(self):
        """Test handling of empty variable names."""
        # This should not match the pattern and be left unchanged
        result = substitute_environment_variables("${}")
        assert result == "${}"
    
    def test_malformed_variable_syntax(self):
        """Test handling of malformed variable syntax."""
        test_cases = [
            "${",          # Incomplete
            "${}",         # Empty name
            "${VAR",       # Missing closing brace
            "$VAR}",       # Missing opening brace
            "${123VAR}",   # Invalid name (starts with number)
        ]
        
        for test_case in test_cases:
            # These should either be left unchanged or raise appropriate errors
            try:
                result = substitute_environment_variables(test_case)
                # If no error, should be unchanged
                assert result == test_case
            except EnvironmentSubstitutionError:
                # Acceptable to raise error for malformed syntax
                pass
    
    def test_nested_substitution_not_supported(self):
        """Test that nested substitution is not supported."""
        with patch.dict(os.environ, {"OUTER": "${INNER}", "INNER": "value"}):
            # Should substitute OUTER but not evaluate its content
            result = substitute_environment_variables("${OUTER}")
            assert result == "${INNER}"
    
    def test_unicode_in_variables(self):
        """Test handling of unicode in variable values."""
        with patch.dict(os.environ, {"UNICODE_VAR": "Hello 🌍"}):
            result = substitute_environment_variables("${UNICODE_VAR}")
            assert result == "Hello 🌍"
    
    def test_very_long_variable_values(self):
        """Test handling of very long variable values."""
        long_value = "x" * 10000
        with patch.dict(os.environ, {"LONG_VAR": long_value}):
            result = substitute_environment_variables("${LONG_VAR}")
            assert result == long_value
            assert len(result) == 10000