"""Jinja2 template engine for MetaReason."""

import re
from typing import Any, Dict, List, Optional, Set

from jinja2 import (
    Environment,
    Template,
    TemplateSyntaxError,
    UndefinedError,
    meta,
    sandbox,
    select_autoescape,
)
from jinja2.exceptions import SecurityError

from .filters import register_custom_filters


class TemplateEngine:
    """Secure Jinja2 template engine with custom filters and functions."""

    def __init__(
        self,
        enable_sandbox: bool = True,
        cache_size: int = 128,
        trim_blocks: bool = True,
        lstrip_blocks: bool = True,
        keep_trailing_newline: bool = True,
    ) -> None:
        """Initialize the template engine.

        Args:
            enable_sandbox: Enable sandboxed environment for security
            cache_size: Size of compiled template cache
            trim_blocks: Remove first newline after block
            lstrip_blocks: Remove leading spaces/tabs from line start
            keep_trailing_newline: Keep trailing newline in templates
        """
        from jinja2 import StrictUndefined

        # Use sandboxed environment for security
        if enable_sandbox:
            self.env = sandbox.SandboxedEnvironment(
                trim_blocks=trim_blocks,
                lstrip_blocks=lstrip_blocks,
                keep_trailing_newline=keep_trailing_newline,
                cache_size=cache_size,
                undefined=StrictUndefined,  # This makes undefined variables raise errors
                autoescape=select_autoescape(
                    disabled_extensions=("txt", "yaml", "yml"), default_for_string=False
                ),
            )
        else:
            self.env = Environment(
                trim_blocks=trim_blocks,
                lstrip_blocks=lstrip_blocks,
                keep_trailing_newline=keep_trailing_newline,
                cache_size=cache_size,
                undefined=StrictUndefined,
                autoescape=select_autoescape(
                    disabled_extensions=("txt", "yaml", "yml"), default_for_string=False
                ),
            )

        # Register custom filters
        register_custom_filters(self.env)

        # Add global functions
        self._register_global_functions()

        # Cache for compiled templates
        self._template_cache: Dict[str, Template] = {}
        self.cache_size = cache_size

    def _register_global_functions(self) -> None:
        """Register global functions available in templates."""

        # Conditional inclusion function
        def include_if(condition: Any, text: str) -> str:
            """Include text only if condition is truthy."""
            return text if condition else ""

        # Select function for choosing between options
        def select(condition: Any, true_val: Any, false_val: Any) -> Any:
            """Select value based on condition."""
            return true_val if condition else false_val

        # Range function for iteration
        def safe_range(
            start: int, stop: Optional[int] = None, step: int = 1
        ) -> List[int]:
            """Safe range function with limits."""
            if stop is None:
                stop = start
                start = 0

            # Limit range to prevent DOS
            max_items = 1000
            items = list(range(start, stop, step))
            return items[:max_items]

        self.env.globals["include_if"] = include_if
        self.env.globals["select"] = select
        self.env.globals["range"] = safe_range

    def compile_template(self, template_string: str) -> Template:
        """Compile a template string with caching.

        Args:
            template_string: The template string to compile

        Returns:
            Compiled Jinja2 template

        Raises:
            TemplateSyntaxError: If template syntax is invalid
        """
        # Check cache first
        if template_string in self._template_cache:
            return self._template_cache[template_string]

        try:
            template = self.env.from_string(template_string)

            # Cache the compiled template (with size limit)
            if len(self._template_cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._template_cache))
                del self._template_cache[oldest_key]

            self._template_cache[template_string] = template
            return template
        except TemplateSyntaxError as e:
            raise TemplateSyntaxError(
                f"Invalid template syntax at line {e.lineno}: {e.message}",
                lineno=e.lineno,
                name=e.name,
                filename=e.filename,
            )

    def render(
        self, template_string: str, context: Dict[str, Any], strict: bool = True
    ) -> str:
        """Render a template with the given context.

        Args:
            template_string: The template string
            context: Dictionary of template variables
            strict: Raise error on undefined variables

        Returns:
            Rendered template string

        Raises:
            UndefinedError: If strict and variable is undefined
            TemplateSyntaxError: If template syntax is invalid
            SecurityError: If template contains unsafe operations
        """
        template = self.compile_template(template_string)

        try:
            if strict:
                # Will raise UndefinedError if variable missing
                return template.render(context)
            else:
                # Silently ignore undefined variables
                from jinja2 import ChainableUndefined

                old_undefined = self.env.undefined
                self.env.undefined = ChainableUndefined
                try:
                    result = template.render(context)
                finally:
                    self.env.undefined = old_undefined
                return result
        except UndefinedError as e:
            raise UndefinedError(f"Template variable not found in context: {e.message}")
        except SecurityError as e:
            raise SecurityError(f"Template contains unsafe operations: {e.message}")

    def extract_variables(self, template_string: str) -> Set[str]:
        """Extract all variable names from a template.

        Args:
            template_string: The template string to analyze

        Returns:
            Set of variable names used in template
        """
        try:
            ast = self.env.parse(template_string)
            return meta.find_undeclared_variables(ast)
        except TemplateSyntaxError:
            # Fallback to regex if parsing fails
            pattern = r"\{\{[\s]*(\w+)"
            return set(re.findall(pattern, template_string))

    def validate_template(self, template_string: str) -> List[str]:
        """Validate template syntax and return any errors.

        Args:
            template_string: The template to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for basic syntax
        if not template_string or not template_string.strip():
            errors.append("Template cannot be empty")
            return errors

        # Try to compile the template
        try:
            self.compile_template(template_string)
        except TemplateSyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.message}")

        # Check for potentially dangerous patterns
        dangerous_patterns = [
            (r"__[^_]+__", "Double underscore attributes are not allowed"),
            (r"import\s+", "Import statements are not allowed"),
            (r"exec\s*\(", "Exec function is not allowed"),
            (r"eval\s*\(", "Eval function is not allowed"),
            (r"compile\s*\(", "Compile function is not allowed"),
            (r"open\s*\(", "File operations are not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, template_string, re.IGNORECASE):
                errors.append(message)

        return errors

    def render_batch(
        self, template_string: str, contexts: List[Dict[str, Any]], strict: bool = True
    ) -> List[str]:
        """Render a template with multiple contexts.

        Args:
            template_string: The template string
            contexts: List of context dictionaries
            strict: Raise error on undefined variables

        Returns:
            List of rendered strings
        """
        template = self.compile_template(template_string)
        results = []

        for context in contexts:
            try:
                if strict:
                    results.append(template.render(context))
                else:
                    from jinja2 import ChainableUndefined

                    old_undefined = self.env.undefined
                    self.env.undefined = ChainableUndefined
                    try:
                        results.append(template.render(context))
                    finally:
                        self.env.undefined = old_undefined
            except (UndefinedError, SecurityError) as e:
                # Store error message in result
                results.append(f"ERROR: {str(e)}")

        return results
