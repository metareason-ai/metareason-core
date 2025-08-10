"""Template validation and security checking for MetaReason."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import TemplateSyntaxError

from .engine import TemplateEngine


class ValidationLevel(Enum):
    """Validation strictness levels."""

    PERMISSIVE = "permissive"  # Basic syntax checking
    STANDARD = "standard"  # Standard validation with security checks
    STRICT = "strict"  # Strict validation with all checks


@dataclass
class ValidationResult:
    """Result of template validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    variables: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0


class TemplateValidator:
    """Comprehensive template validation with security checks."""

    # Security patterns to detect
    SECURITY_PATTERNS = [
        # Python introspection
        (r"__[^_]+__", "Double underscore attributes may access internals"),
        (r"\._[^_]", "Single underscore attributes may access private members"),
        # Code execution
        (r"\bimport\s+", "Import statements are not allowed"),
        (r"\bexec\s*\(", "Exec function is not allowed"),
        (r"\beval\s*\(", "Eval function is not allowed"),
        (r"\bcompile\s*\(", "Compile function is not allowed"),
        (r"\bglobals\s*\(", "Globals function is not allowed"),
        (r"\blocals\s*\(", "Locals function is not allowed"),
        # File system access
        (r"\bopen\s*\(", "File operations are not allowed"),
        (r"\bfile\s*\(", "File operations are not allowed"),
        (r"\.read\s*\(", "File read operations are not allowed"),
        (r"\.write\s*\(", "File write operations are not allowed"),
        # System access
        (r"\bos\.\w+", "OS module access is not allowed"),
        (r"\bsys\.\w+", "Sys module access is not allowed"),
        (r"\bsubprocess", "Subprocess operations are not allowed"),
        # Network access
        (r"\burlopen", "Network operations are not allowed"),
        (r"\brequests\.\w+", "Network requests are not allowed"),
    ]

    # Suspicious patterns (warnings)
    WARNING_PATTERNS = [
        (r"\|safe\b", "The 'safe' filter bypasses HTML escaping"),
        (r"loop\.index0", "Consider using loop.index for 1-based indexing"),
        (r"{%-?\s*raw\s*-?%}", "Raw blocks bypass template processing"),
        (r"recursive\s*\(", "Recursive calls may cause performance issues"),
    ]

    def __init__(
        self,
        engine: Optional[TemplateEngine] = None,
        level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> None:
        """Initialize the validator.

        Args:
            engine: Template engine to use for validation
            level: Validation strictness level
        """
        self.engine = engine or TemplateEngine()
        self.level = level

    def validate(
        self,
        template_string: str,
        expected_variables: Optional[Set[str]] = None,
        max_length: Optional[int] = None,
    ) -> ValidationResult:
        """Validate a template string.

        Args:
            template_string: Template to validate
            expected_variables: Expected variable names (if provided)
            max_length: Maximum allowed template length

        Returns:
            ValidationResult with errors, warnings, and metadata
        """
        errors = []
        warnings = []
        variables = set()
        metadata = {}

        # Basic checks
        if not template_string:
            errors.append("Template cannot be empty")
            return ValidationResult(
                is_valid=False, errors=errors, metadata={"empty": True}
            )

        # Length check
        if max_length and len(template_string) > max_length:
            errors.append(
                f"Template exceeds maximum length ({len(template_string)} > {max_length})"
            )

        metadata["length"] = len(template_string)
        metadata["lines"] = template_string.count("\n") + 1

        # Syntax validation
        syntax_errors = self._validate_syntax(template_string)
        errors.extend(syntax_errors)

        if not syntax_errors:
            # Extract variables only if syntax is valid
            variables = self._extract_variables(template_string)
            metadata["variable_count"] = len(variables)

            # Check for undefined variables
            if expected_variables:
                undefined = variables - expected_variables
                if undefined:
                    errors.append(
                        f"Template uses undefined variables: {', '.join(sorted(undefined))}"
                    )

                unused = expected_variables - variables
                if unused and self.level == ValidationLevel.STRICT:
                    warnings.append(
                        f"Template does not use all available variables: {', '.join(sorted(unused))}"
                    )

        # Security checks
        if self.level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            security_errors, security_warnings = self._security_check(template_string)
            errors.extend(security_errors)
            warnings.extend(security_warnings)

        # Additional strict checks
        if self.level == ValidationLevel.STRICT:
            strict_warnings = self._strict_checks(template_string)
            warnings.extend(strict_warnings)

        # Character encoding check
        encoding_issues = self._check_encoding(template_string)
        if encoding_issues:
            warnings.extend(encoding_issues)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            variables=variables,
            metadata=metadata,
        )

    def _validate_syntax(self, template_string: str) -> List[str]:
        """Validate template syntax."""
        errors = []

        try:
            # Try to compile the template
            self.engine.compile_template(template_string)
        except TemplateSyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.message}")
        except Exception as e:
            errors.append(f"Template compilation failed: {str(e)}")

        # Check for balanced delimiters
        delimiter_errors = self._check_delimiters(template_string)
        errors.extend(delimiter_errors)

        return errors

    def _check_delimiters(self, template_string: str) -> List[str]:
        """Check for balanced template delimiters."""
        errors = []

        # Count delimiters
        open_var = template_string.count("{{")
        close_var = template_string.count("}}")
        open_block = template_string.count("{%")
        close_block = template_string.count("%}")
        open_comment = template_string.count("{#")
        close_comment = template_string.count("#}")

        if open_var != close_var:
            errors.append(
                f"Unbalanced variable delimiters: {open_var} '{{{{' vs {close_var} '}}}}'"
            )

        if open_block != close_block:
            errors.append(
                f"Unbalanced block delimiters: {open_block} '{{%' vs {close_block} '%}}'"
            )

        if open_comment != close_comment:
            errors.append(
                f"Unbalanced comment delimiters: {open_comment} '{{#' vs {close_comment} '#}}'"
            )

        return errors

    def _extract_variables(self, template_string: str) -> Set[str]:
        """Extract variable names from template."""
        try:
            return self.engine.extract_variables(template_string)
        except Exception:
            # Fallback to regex
            pattern = r"\{\{[\s]*(\w+)"
            return set(re.findall(pattern, template_string))

    def _security_check(self, template_string: str) -> Tuple[List[str], List[str]]:
        """Perform security checks on template."""
        errors = []
        warnings = []

        # Check security patterns
        for pattern, message in self.SECURITY_PATTERNS:
            if re.search(pattern, template_string, re.IGNORECASE):
                errors.append(f"Security risk: {message}")

        # Check warning patterns
        for pattern, message in self.WARNING_PATTERNS:
            if re.search(pattern, template_string, re.IGNORECASE):
                warnings.append(f"Potential issue: {message}")

        return errors, warnings

    def _strict_checks(self, template_string: str) -> List[str]:
        """Perform strict validation checks."""
        warnings = []

        # Check for very long lines
        lines = template_string.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > 200:
                warnings.append(f"Line {i} is very long ({len(line)} chars)")

        # Check for excessive nesting
        nesting_level = 0
        max_nesting = 0
        for _match in re.finditer(r"{%\s*(if|for|block|macro)", template_string):
            nesting_level += 1
            max_nesting = max(max_nesting, nesting_level)
        for _match in re.finditer(r"{%\s*end(if|for|block|macro)", template_string):
            nesting_level -= 1

        if max_nesting > 3:
            warnings.append(f"Deep nesting detected (level {max_nesting})")

        # Check for complex expressions
        complex_patterns = [
            (
                r"\|\s*\w+\s*\|\s*\w+\s*\|\s*\w+",
                "Complex filter chains may be hard to maintain",
            ),
            (
                r"if\s+.{50,}\s*%}",
                "Complex conditional expressions may be hard to read",
            ),
        ]

        for pattern, message in complex_patterns:
            if re.search(pattern, template_string):
                warnings.append(message)

        return warnings

    def _check_encoding(self, template_string: str) -> List[str]:
        """Check for character encoding issues."""
        warnings = []

        # Check for non-ASCII characters
        non_ascii = []
        for i, char in enumerate(template_string):
            if ord(char) > 127:
                non_ascii.append((i, char, ord(char)))

        if non_ascii:
            # Check if they're valid UTF-8
            try:
                template_string.encode("utf-8")
            except UnicodeEncodeError:
                warnings.append("Template contains invalid UTF-8 characters")

            # Warn about control characters
            control_chars = [
                c
                for _, c, code in non_ascii
                if code < 32 or (code >= 127 and code < 160)
            ]
            if control_chars:
                warnings.append(
                    f"Template contains control characters: {control_chars}"
                )

        return warnings

    def validate_output(
        self,
        rendered_output: str,
        max_length: Optional[int] = None,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate rendered template output.

        Args:
            rendered_output: The rendered template string
            max_length: Maximum allowed output length
            required_patterns: Patterns that must be present
            forbidden_patterns: Patterns that must not be present

        Returns:
            ValidationResult for the output
        """
        errors = []
        warnings = []
        metadata = {"output_length": len(rendered_output)}

        # Length check
        if max_length and len(rendered_output) > max_length:
            errors.append(
                f"Output exceeds maximum length ({len(rendered_output)} > {max_length})"
            )

        # Required patterns
        if required_patterns:
            for pattern in required_patterns:
                if not re.search(pattern, rendered_output):
                    errors.append(f"Required pattern not found: {pattern}")

        # Forbidden patterns
        if forbidden_patterns:
            for pattern in forbidden_patterns:
                if re.search(pattern, rendered_output):
                    errors.append(f"Forbidden pattern found: {pattern}")

        # Check for template artifacts (unrendered variables)
        if "{{" in rendered_output or "{%" in rendered_output:
            warnings.append("Output contains unrendered template syntax")

        # Check for error markers
        if "ERROR:" in rendered_output or "undefined" in rendered_output.lower():
            warnings.append("Output may contain error messages")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )
