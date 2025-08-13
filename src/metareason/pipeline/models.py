"""Data models for pipeline execution results."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..adapters.base import CompletionResponse
from ..analysis.bayesian import MultiOracleResult
from ..config.models import EvaluationConfig
from ..oracles.base import OracleResult
from ..sampling.base import SampleResult

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from executing a single pipeline step."""

    step_index: int
    step_name: str
    prompts: List[str]
    responses: List[CompletionResponse]
    errors: List[Exception] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this step."""
        total = len(self.prompts)
        if total == 0:
            return 0.0
        successful = len(self.responses)
        return successful / total

    @property
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return self.success_rate > 0.95  # 95% success threshold


@dataclass
class PipelineResult:
    """Complete result from pipeline execution."""

    # Core data
    config: EvaluationConfig
    samples: SampleResult
    step_results: List[StepResult]
    oracle_results: Dict[str, List[OracleResult]]
    bayesian_results: Optional[MultiOracleResult] = None

    # Metadata
    execution_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization to calculate derived fields."""
        if self.end_time and self.total_duration is None:
            self.total_duration = (self.end_time - self.start_time).total_seconds()

    @property
    def is_successful(self) -> bool:
        """Check if entire pipeline completed successfully."""
        return all(step.is_successful for step in self.step_results)

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate across all steps."""
        if not self.step_results:
            return 0.0
        return sum(step.success_rate for step in self.step_results) / len(
            self.step_results
        )

    @property
    def total_responses(self) -> int:
        """Total number of LLM responses across all steps."""
        return sum(len(step.responses) for step in self.step_results)

    @property
    def total_errors(self) -> int:
        """Total number of errors across all steps."""
        return sum(len(step.errors) for step in self.step_results)

    def get_step_outputs(self, step_index: int) -> List[str]:
        """Get output content from a specific step.

        Args:
            step_index: Index of the step (0-based)

        Returns:
            List of response content strings
        """
        if step_index >= len(self.step_results):
            return []

        step = self.step_results[step_index]
        return [response.content for response in step.responses]

    def finalize(self, end_time: Optional[datetime] = None) -> None:
        """Mark pipeline as complete and calculate final metrics.

        Args:
            end_time: Optional end time (defaults to now)
        """
        self.end_time = end_time or datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()

        # Log completion
        logger.info(
            f"Pipeline {self.execution_id} completed in {self.total_duration:.2f}s "
            f"with {self.overall_success_rate:.1%} success rate"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline execution.

        Returns:
            Dictionary with key metrics and status
        """
        return {
            "execution_id": self.execution_id,
            "spec_id": self.config.spec_id,
            "status": "success" if self.is_successful else "partial_failure",
            "overall_success_rate": self.overall_success_rate,
            "total_samples": self.samples.metadata.get("n_samples", 0),
            "pipeline_steps": len(self.step_results),
            "total_responses": self.total_responses,
            "total_errors": self.total_errors,
            "execution_time": self.total_duration,
            "oracles_evaluated": len(self.oracle_results),
            "bayesian_analysis": self.bayesian_results is not None,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class ExecutionPlan:
    """Plan for pipeline execution (used in dry-run mode)."""

    config: EvaluationConfig
    estimated_samples: int
    estimated_prompts: int
    estimated_api_calls: int
    estimated_cost: Optional[float] = None
    estimated_duration: Optional[float] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_step_plan(
        self,
        step_index: int,
        step_name: str,
        adapter: str,
        model: str,
        estimated_calls: int,
        estimated_cost: Optional[float] = None,
    ) -> None:
        """Add a step to the execution plan.

        Args:
            step_index: Index of the step
            step_name: Human-readable step name
            adapter: Adapter type
            model: Model name
            estimated_calls: Estimated number of API calls
            estimated_cost: Estimated cost for this step
        """
        self.steps.append(
            {
                "step_index": step_index,
                "step_name": step_name,
                "adapter": adapter,
                "model": model,
                "estimated_calls": estimated_calls,
                "estimated_cost": estimated_cost,
            }
        )

    def add_warning(self, warning: str) -> None:
        """Add a warning to the execution plan.

        Args:
            warning: Warning message
        """
        self.warnings.append(warning)
