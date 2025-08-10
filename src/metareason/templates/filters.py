"""Custom Jinja2 filters for MetaReason template rendering."""

from typing import Any, List, Optional, Union


def format_continuous(value: float, precision: int = 2, style: str = "decimal") -> str:
    """Format continuous numerical values.

    Args:
        value: The numerical value to format
        precision: Number of decimal places
        style: Formatting style ('decimal', 'percent', 'scientific')

    Returns:
        Formatted string representation
    """
    if style == "percent":
        return f"{value * 100:.{precision}f}%"
    elif style == "scientific":
        return f"{value:.{precision}e}"
    else:  # decimal
        return f"{value:.{precision}f}"


def format_list(
    items: List[Any],
    separator: str = ", ",
    conjunction: str = "and",
    oxford_comma: bool = True,
) -> str:
    """Format a list of items with proper conjunction.

    Args:
        items: List of items to format
        separator: Separator between items
        conjunction: Conjunction word for last item
        oxford_comma: Whether to use Oxford comma

    Returns:
        Formatted string representation
    """
    if not items:
        return ""

    items_str = [str(item) for item in items]

    if len(items_str) == 1:
        return items_str[0]
    elif len(items_str) == 2:
        return f"{items_str[0]} {conjunction} {items_str[1]}"
    else:
        if oxford_comma:
            return f"{separator.join(items_str[:-1])}{separator}{conjunction} {items_str[-1]}"
        else:
            return f"{separator.join(items_str[:-1])} {conjunction} {items_str[-1]}"


def conditional_text(
    condition: Any,
    true_text: str = "",
    false_text: str = "",
    check_type: str = "truthy",
) -> str:
    """Conditionally include text based on parameter value.

    Args:
        condition: The value to check
        true_text: Text to return if condition is met
        false_text: Text to return if condition is not met
        check_type: Type of check ('truthy', 'equals', 'contains', 'gt', 'lt')

    Returns:
        Selected text based on condition
    """
    result = False

    if check_type == "truthy":
        result = bool(condition)
    elif check_type == "equals":
        # For equals, we expect condition to be a tuple (value, compare_to)
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            result = condition[0] == condition[1]
    elif check_type == "contains":
        # Check if first item contains second
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            result = condition[1] in str(condition[0])
    elif check_type == "gt":
        # Greater than comparison
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            try:
                result = float(condition[0]) > float(condition[1])
            except (ValueError, TypeError):
                result = False
    elif check_type == "lt":
        # Less than comparison
        if isinstance(condition, (list, tuple)) and len(condition) == 2:
            try:
                result = float(condition[0]) < float(condition[1])
            except (ValueError, TypeError):
                result = False

    return true_text if result else false_text


def round_to_precision(value: float, precision: int = 2) -> float:
    """Round a number to specified precision.

    Args:
        value: The value to round
        precision: Number of decimal places

    Returns:
        Rounded value
    """
    return round(value, precision)


def capitalize_first(text: str) -> str:
    """Capitalize only the first letter of text.

    Args:
        text: Text to capitalize

    Returns:
        Text with first letter capitalized
    """
    if not text:
        return text
    return text[0].upper() + text[1:]


def pluralize(
    count: Union[int, float], singular: str, plural: Optional[str] = None
) -> str:
    """Pluralize a word based on count.

    Args:
        count: The count to check
        singular: Singular form of the word
        plural: Plural form (if None, adds 's' to singular)

    Returns:
        Appropriate form based on count
    """
    if plural is None:
        plural = f"{singular}s"

    return singular if abs(count) == 1 else plural


def truncate(text: str, length: int = 50, suffix: str = "...") -> str:
    """Truncate text to specified length.

    Args:
        text: Text to truncate
        length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= length:
        return text

    return text[: length - len(suffix)] + suffix


def register_custom_filters(env: Any) -> None:
    """Register all custom filters with a Jinja2 environment.

    Args:
        env: Jinja2 Environment instance
    """
    env.filters["format_continuous"] = format_continuous
    env.filters["format_list"] = format_list
    env.filters["conditional_text"] = conditional_text
    env.filters["round_to_precision"] = round_to_precision
    env.filters["capitalize_first"] = capitalize_first
    env.filters["pluralize"] = pluralize
    env.filters["truncate"] = truncate

    # Aliases for convenience
    env.filters["fmt_num"] = format_continuous
    env.filters["fmt_list"] = format_list
    env.filters["if_text"] = conditional_text
