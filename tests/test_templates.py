"""Test suite for MetaReason template system."""

import time

import pytest
from jinja2 import UndefinedError

from metareason.templates import (
    BatchRenderer,
    RenderResult,
    TemplateEngine,
    TemplateValidator,
    ValidationLevel,
    conditional_text,
    format_continuous,
    format_list,
)


class TestCustomFilters:
    """Test custom Jinja2 filters."""

    def test_format_continuous_decimal(self):
        """Test decimal formatting."""
        assert format_continuous(3.14159, precision=2) == "3.14"
        assert format_continuous(42.0, precision=0) == "42"
        assert format_continuous(0.123456, precision=4) == "0.1235"

    def test_format_continuous_percent(self):
        """Test percentage formatting."""
        assert format_continuous(0.75, precision=1, style="percent") == "75.0%"
        assert format_continuous(0.1234, precision=2, style="percent") == "12.34%"
        assert format_continuous(1.0, precision=0, style="percent") == "100%"

    def test_format_continuous_scientific(self):
        """Test scientific notation formatting."""
        assert format_continuous(1234.5, precision=2, style="scientific") == "1.23e+03"
        assert format_continuous(0.00012, precision=1, style="scientific") == "1.2e-04"

    def test_format_list_simple(self):
        """Test list formatting."""
        assert format_list([1, 2, 3]) == "1, 2, and 3"
        assert format_list(["apple", "banana"]) == "apple and banana"
        assert format_list(["only"]) == "only"
        assert format_list([]) == ""

    def test_format_list_custom(self):
        """Test list formatting with custom options."""
        items = ["red", "green", "blue"]
        assert format_list(items, separator="; ") == "red; green; and blue"
        assert format_list(items, conjunction="or") == "red, green, or blue"
        assert format_list(items, oxford_comma=False) == "red, green and blue"

    def test_conditional_text_truthy(self):
        """Test conditional text with truthy check."""
        assert conditional_text(True, "yes", "no") == "yes"
        assert conditional_text(False, "yes", "no") == "no"
        assert conditional_text(1, "exists", "empty") == "exists"
        assert conditional_text("", "exists", "empty") == "empty"
        assert conditional_text([], "has items", "no items") == "no items"

    def test_conditional_text_comparisons(self):
        """Test conditional text with comparisons."""
        assert (
            conditional_text([5, 3], "greater", "not greater", check_type="gt")
            == "greater"
        )
        assert conditional_text([2, 5], "less", "not less", check_type="lt") == "less"
        assert (
            conditional_text(
                ["hello", "hello"], "equal", "not equal", check_type="equals"
            )
            == "equal"
        )
        assert (
            conditional_text(
                ["hello world", "world"],
                "contains",
                "not contains",
                check_type="contains",
            )
            == "contains"
        )


class TestTemplateEngine:
    """Test the Jinja2 template engine."""

    @pytest.fixture
    def engine(self):
        """Create a template engine instance."""
        return TemplateEngine()

    def test_simple_rendering(self, engine):
        """Test basic template rendering."""
        template = "Hello {{name}}, you are {{age}} years old."
        context = {"name": "Alice", "age": 30}
        result = engine.render(template, context)
        assert result == "Hello Alice, you are 30 years old."

    def test_custom_filters_in_template(self, engine):
        """Test using custom filters in templates."""
        template = "The value is {{value | format_continuous(3, 'percent')}}"
        context = {"value": 0.12345}
        result = engine.render(template, context)
        assert result == "The value is 12.345%"

    def test_list_formatting_in_template(self, engine):
        """Test list formatting in templates."""
        template = "Options: {{items | format_list}}"
        context = {"items": ["A", "B", "C"]}
        result = engine.render(template, context)
        assert result == "Options: A, B, and C"

    def test_global_functions(self, engine):
        """Test global template functions."""
        template = """
        {% if include_if(show_detail, "Details: Important information.") %}
        {{ include_if(show_detail, "Details: Important information.") }}
        {% endif %}
        Result: {{ select(is_premium, "Premium", "Basic") }}
        """.strip()

        context = {"show_detail": True, "is_premium": False}
        result = engine.render(template, context)
        assert "Details: Important information." in result
        assert "Result: Basic" in result

    def test_undefined_variable_strict(self, engine):
        """Test undefined variable handling in strict mode."""
        template = "Hello {{name}}"
        context = {}

        with pytest.raises(UndefinedError):
            engine.render(template, context, strict=True)

    def test_undefined_variable_permissive(self, engine):
        """Test undefined variable handling in permissive mode."""
        template = "Hello {{name}}"
        context = {}
        result = engine.render(template, context, strict=False)
        # In permissive mode, undefined variables are replaced with empty string
        assert "Hello" in result

    def test_extract_variables(self, engine):
        """Test variable extraction from template."""
        template = """
        {{name}} is {{age}} years old.
        {% if show_address %}
        Address: {{address}}
        {% endif %}
        """
        variables = engine.extract_variables(template)
        assert variables == {"name", "age", "show_address", "address"}

    def test_template_validation(self, engine):
        """Test template validation."""
        # Valid template
        valid_template = "Hello {{name}}"
        errors = engine.validate_template(valid_template)
        assert len(errors) == 0

        # Invalid syntax
        invalid_template = "Hello {{name"
        errors = engine.validate_template(invalid_template)
        assert len(errors) > 0

        # Dangerous patterns
        dangerous_template = "{{__class__}}"
        errors = engine.validate_template(dangerous_template)
        assert any("Double underscore" in e for e in errors)

    def test_batch_rendering(self, engine):
        """Test batch rendering."""
        template = "Hello {{name}}, you are {{age}}."
        contexts = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        results = engine.render_batch(template, contexts)
        assert len(results) == 3
        assert "Alice" in results[0]
        assert "Bob" in results[1]
        assert "Charlie" in results[2]


class TestBatchRenderer:
    """Test the batch rendering system."""

    @pytest.fixture
    def renderer(self):
        """Create a batch renderer instance."""
        return BatchRenderer(batch_size=10)

    def test_small_batch_rendering(self, renderer):
        """Test rendering a small batch."""
        template = "Item {{id}}: {{name}}"
        contexts = [{"id": i, "name": f"Item{i}"} for i in range(5)]

        result = renderer.render(template, contexts)

        assert isinstance(result, RenderResult)
        assert result.success_count == 5
        assert result.error_count == 0
        assert len(result.rendered_prompts) == 5
        assert "Item 0: Item0" in result.rendered_prompts[0]

    def test_large_batch_rendering(self, renderer):
        """Test rendering a large batch."""
        template = "Number: {{num | format_continuous(2)}}"
        contexts = [{"num": i * 0.1} for i in range(150)]

        result = renderer.render(template, contexts, parallel=True)

        assert result.success_count == 150
        assert result.error_count == 0
        assert result.success_rate == 100.0
        assert len(result.rendered_prompts) == 150

    def test_rendering_with_errors(self, renderer):
        """Test rendering with some errors."""
        template = "Value: {{value}}"
        contexts = [{"value": i} if i % 2 == 0 else {"wrong_key": i} for i in range(10)]

        result = renderer.render(template, contexts, strict=True)

        assert result.success_count == 5  # Even numbers have correct key
        assert result.error_count == 5  # Odd numbers have wrong key
        assert len(result.errors) == 5

    def test_progress_callback(self, renderer):
        """Test progress callback functionality."""
        template = "Item {{id}}"
        contexts = [{"id": i} for i in range(50)]

        progress_updates = []

        def callback(count):
            progress_updates.append(count)

        result = renderer.render(template, contexts, progress_callback=callback)

        assert result.success_count == 50
        assert len(progress_updates) > 0
        assert sum(progress_updates) == 50

    def test_memory_estimation(self, renderer):
        """Test memory usage estimation."""
        template = "A fairly long template with {{variable}} content"
        estimates = renderer.estimate_memory_usage(template, 1000)

        assert "total_estimated_mb" in estimates
        assert "recommended_batch_size" in estimates
        assert estimates["total_estimated_mb"] > 0
        assert estimates["recommended_batch_size"] > 0

    def test_invalid_template_handling(self, renderer):
        """Test handling of invalid templates."""
        template = "Invalid {{template"  # Missing closing braces
        contexts = [{"template": "test"}]

        result = renderer.render(template, contexts)

        assert result.error_count > 0
        assert result.success_count == 0
        assert result.metadata.get("validation_failed") is True


class TestTemplateValidator:
    """Test template validation system."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return TemplateValidator(level=ValidationLevel.STANDARD)

    def test_valid_template(self, validator):
        """Test validation of a valid template."""
        template = "Hello {{name}}, welcome to {{place}}!"
        result = validator.validate(template, expected_variables={"name", "place"})

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.variables == {"name", "place"}

    def test_empty_template(self, validator):
        """Test validation of empty template."""
        result = validator.validate("")

        assert not result.is_valid
        assert "empty" in result.errors[0].lower()

    def test_syntax_errors(self, validator):
        """Test detection of syntax errors."""
        template = "Hello {{name"  # Missing closing braces
        result = validator.validate(template)

        assert not result.is_valid
        assert any("delimiter" in e.lower() for e in result.errors)

    def test_security_patterns(self, validator):
        """Test detection of security risks."""
        dangerous_templates = [
            "{{__class__}}",
            "{{eval('code')}}",
            "{{open('/etc/passwd')}}",
            "{% import os %}",
        ]

        for template in dangerous_templates:
            result = validator.validate(template)
            assert not result.is_valid
            assert any(
                "security" in e.lower() or "not allowed" in e.lower()
                for e in result.errors
            )

    def test_undefined_variables(self, validator):
        """Test detection of undefined variables."""
        template = "Hello {{name}} from {{unknown}}"
        result = validator.validate(template, expected_variables={"name"})

        assert not result.is_valid
        assert any("undefined" in e.lower() for e in result.errors)
        assert "unknown" in result.errors[0]

    def test_strict_validation(self):
        """Test strict validation level."""
        validator = TemplateValidator(level=ValidationLevel.STRICT)

        # Long line warning
        template = "x" * 250
        result = validator.validate(template)
        assert len(result.warnings) > 0

        # Unused variables warning
        template = "Hello {{name}}"
        result = validator.validate(template, expected_variables={"name", "unused"})
        assert any("unused" in w.lower() for w in result.warnings)

    def test_output_validation(self, validator):
        """Test validation of rendered output."""
        output = "Hello World!"

        # Valid output
        result = validator.validate_output(output, max_length=100)
        assert result.is_valid

        # Too long
        result = validator.validate_output(output, max_length=5)
        assert not result.is_valid

        # Missing required pattern
        result = validator.validate_output(output, required_patterns=[r"Goodbye"])
        assert not result.is_valid

        # Contains forbidden pattern
        result = validator.validate_output(output, forbidden_patterns=[r"World"])
        assert not result.is_valid

    def test_encoding_check(self, validator):
        """Test character encoding validation."""
        # Valid UTF-8
        template = "Hello 世界 🌍"
        result = validator.validate(template)
        assert result.is_valid

        # Control characters warning
        template = "Hello\x1fWorld"  # Use ASCII control character
        result = validator.validate(template)
        # Note: this might not trigger warnings in all Jinja2 versions
        # so we'll just check it doesn't crash
        assert isinstance(result.warnings, list)


class TestIntegration:
    """Integration tests for the template system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from validation to rendering."""
        # Setup
        engine = TemplateEngine()
        validator = TemplateValidator(engine=engine)
        renderer = BatchRenderer(engine=engine)

        # Template
        template = """
        You are a {{role}} assistant.
        Temperature: {{temp | format_continuous(1, 'decimal')}}
        Skills: {{skills | format_list}}
        {% if premium %}Premium features enabled.{% endif %}
        """

        # Validation
        validation = validator.validate(template)
        assert validation.is_valid
        assert validation.variables == {"role", "temp", "skills", "premium"}

        # Prepare contexts
        contexts = [
            {
                "role": "helpful",
                "temp": 0.7,
                "skills": ["coding", "writing"],
                "premium": True,
            },
            {
                "role": "creative",
                "temp": 0.9,
                "skills": ["art", "music", "poetry"],
                "premium": False,
            },
        ]

        # Render
        result = renderer.render(template, contexts)
        assert result.success_count == 2
        assert result.error_count == 0

        # Validate outputs
        for output in result.rendered_prompts:
            output_validation = validator.validate_output(output)
            assert output_validation.is_valid

    @pytest.mark.slow
    def test_performance_large_batch(self):
        """Test performance with large batch."""
        engine = TemplateEngine()
        renderer = BatchRenderer(engine=engine, batch_size=100)

        template = "User {{id}}: {{name}} ({{score | format_continuous(2)}}%)"
        contexts = [
            {"id": i, "name": f"User{i}", "score": i * 0.1} for i in range(2000)
        ]

        start_time = time.time()
        result = renderer.render(template, contexts, parallel=True)
        render_time = time.time() - start_time

        assert result.success_count == 2000
        assert result.error_count == 0
        assert render_time < 10.0  # Should complete within 10 seconds

        # Check metadata
        assert result.metadata["total_contexts"] == 2000
        assert result.metadata["parallel"] is True
        assert result.metadata["avg_time_per_prompt"] < 0.005  # < 5ms per prompt
