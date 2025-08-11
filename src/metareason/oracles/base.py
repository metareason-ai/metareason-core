"""Base oracle classes for evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class OracleResult(BaseModel):
    """Base result from oracle evaluation."""

    score: float
    metadata: Dict[str, Any] = {}


class BaseOracle(ABC):
    """Abstract base class for all oracles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize oracle with configuration.

        Args:
            config: Oracle-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    async def evaluate(self, response: str, context: Dict[str, Any]) -> OracleResult:
        """Evaluate a response against oracle criteria.

        Args:
            response: The LLM response to evaluate
            context: Additional context including prompt, expected answer, etc.

        Returns:
            OracleResult with score and metadata

        Raises:
            OracleError: On evaluation failure
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get oracle name for identification."""
        pass


class OracleError(Exception):
    """Base exception for oracle errors."""

    pass
