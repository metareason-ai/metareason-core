from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..config import OracleConfig


class OracleException(Exception):
    """Base class for Oracle exceptions."""

    pass


class EvaluationContext(BaseModel):
    """Request to an oracle for evaluation."""

    prompt: str  # the original prompt
    response: str  # the response from the LLM to evaluate
    criteria: Optional[str] = None  # the optional criteria/instructions/rubric to use
    metadata: Dict[str, Any] = Field(default_factory=dict)  # sample params


class EvaluationResult(BaseModel):
    """Response from an oracle evaluation."""

    score: float = Field(ge=1.0, le=5.0)
    explanation: Optional[str] = None


class OracleBase(ABC):
    """Abstract base class for all oracle implementations."""

    def __init__(self, config: OracleConfig):
        self.config = config

    @abstractmethod
    def evaluate(self, request: EvaluationContext) -> EvaluationResult:
        """Evaluate an LLM response.

        Args:
            request: The oracle request containing prompt and response

        Returns:
            EvaluationResult with score and optional reasoning.

        Raises:
            OracleException: If evaluation fails
        """
        ...
