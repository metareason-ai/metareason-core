"""Regex Oracle for evaluating LLM responses via pattern matching.

This module provides the RegexOracle implementation, which evaluates LLM
responses by checking them against one or more regular expression patterns.
No external LLM call is needed - evaluation is purely local.

The oracle scores responses on a 1-5 scale based on the proportion of
patterns that match the response text.

Classes:
    RegexOracle: Oracle that evaluates responses using regex pattern matching.
"""

import re
from typing import List, Tuple

from ..config import OracleConfig
from .oracle_base import (
    EvaluationContext,
    EvaluationResult,
    OracleBase,
    OracleException,
)


class RegexOracle(OracleBase):
    """Regex-based oracle for evaluating LLM responses via pattern matching.

    This oracle checks the LLM response against one or more regex patterns
    and scores based on the proportion of patterns that match. Useful for
    format validation, keyword presence checking, and structural evaluation
    without requiring an LLM API call.

    Scoring: The score is linearly interpolated between 1.0 and 5.0 based
    on the fraction of patterns matched. All patterns match = 5.0, none = 1.0.

    Attributes:
        compiled_patterns: List of (pattern_string, compiled_regex) tuples.

    Raises:
        ValueError: If no patterns are provided or a pattern is invalid regex.

    Example:
        >>> config = OracleConfig(
        ...     type="regex",
        ...     patterns=["\\\\d+", "[A-Z]"],
        ... )
        >>> oracle = RegexOracle(config)
        >>> context = EvaluationContext(
        ...     prompt="Give me a number",
        ...     response="The answer is 42"
        ... )
        >>> result = await oracle.evaluate(context)
        >>> print(result.score)  # 5.0 (both patterns match)
    """

    def __init__(self, config: OracleConfig):
        super().__init__(config)

        if not self.config.patterns:
            raise ValueError("At least one pattern is required for regex oracle.")

        self.compiled_patterns: List[Tuple[str, re.Pattern]] = []
        for pattern in self.config.patterns:
            try:
                self.compiled_patterns.append((pattern, re.compile(pattern)))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    async def evaluate(self, request: EvaluationContext) -> EvaluationResult:
        """Evaluate an LLM response by checking regex patterns against it.

        Args:
            request: Context containing the prompt and response to evaluate.

        Returns:
            EvaluationResult with score (1.0-5.0) based on pattern match ratio,
            and an explanation listing which patterns matched/failed.

        Raises:
            OracleException: If evaluation fails unexpectedly.
        """
        try:
            matched = []
            failed = []

            for pattern_str, compiled in self.compiled_patterns:
                if compiled.search(request.response):
                    matched.append(pattern_str)
                else:
                    failed.append(pattern_str)

            total = len(self.compiled_patterns)
            match_ratio = len(matched) / total

            # Linear interpolation: 0% matched -> 1.0, 100% matched -> 5.0
            score = 1.0 + 4.0 * match_ratio

            parts = []
            if matched:
                parts.append(f"Matched {len(matched)}/{total} patterns: {matched}")
            if failed:
                parts.append(f"Failed {len(failed)}/{total} patterns: {failed}")
            explanation = ". ".join(parts)

            return EvaluationResult(score=score, explanation=explanation)

        except Exception as e:
            raise OracleException(f"Regex oracle evaluation failed: {e}") from e
