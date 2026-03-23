"""Auto-calibration loop for iterative judge rubric optimization.

Orchestrates the evaluate → check → optimize cycle, tracking history
and the best-performing rubric across iterations.
"""

from dataclasses import dataclass, field

from metareason.adapters.adapter_factory import get_adapter
from metareason.analysis.analyzer import BayesianAnalyzer
from metareason.calibration.convergence import ConvergenceChecker, ConvergenceResult
from metareason.calibration.optimizer import RubricOptimizer
from metareason.config.models import CalibrateConfig
from metareason.oracles.llm_judge import LLMJudge
from metareason.oracles.oracle_base import EvaluationContext
from metareason.pipeline.runner import SampleResult


@dataclass
class AutoCalibrationResult:
    """Outcome of an auto-calibration run.

    Attributes:
        converged: Whether the judge truly converged (not just hit max iterations).
        iterations: Total number of iterations executed.
        final_rubric: The rubric used in the last iteration.
        original_rubric: The rubric from the initial config, before any optimization.
        convergence_result: The ConvergenceResult from the final iteration.
        history: Per-iteration records with rubric, scores, and analysis.
        best_rubric: The rubric that produced the lowest absolute bias.
        best_cal_result: The calibration result dict from the best iteration.
    """

    converged: bool
    iterations: int
    final_rubric: str
    original_rubric: str
    convergence_result: ConvergenceResult
    history: list[dict] = field(default_factory=list)
    best_rubric: str = ""
    best_cal_result: dict = field(default_factory=dict)


async def _run_single_calibration(
    config: CalibrateConfig,
) -> tuple[list[float], dict]:
    """Run one round of judge evaluation + Bayesian analysis.

    This is the inner loop extracted from the CLI calibrate command.
    Runs repeated evaluations with the judge, then fits the Bayesian
    calibration model to estimate bias and noise.

    Args:
        config: Calibration config with oracle, prompt, response, and analysis settings.

    Returns:
        Tuple of (scores list, calibration result dict).
    """
    judge = LLMJudge(config.oracle)
    eval_context = EvaluationContext(
        prompt=config.prompt,
        response=config.response,
    )

    eval_results = []
    for _ in range(config.repeats):
        try:
            result = await judge.evaluate(eval_context)
            eval_results.append(result)
        except Exception:  # nosec B112
            # Individual eval failures are expected (e.g. JSON parse errors
            # from the judge). We continue collecting what we can.
            continue

    scores = [r.score for r in eval_results]

    oracle_name = config.oracle.model
    sample_results = [
        SampleResult(
            sample_params={},
            original_prompt=config.prompt,
            final_response=config.response,
            evaluations={oracle_name: r},
        )
        for r in eval_results
    ]

    analyzer = BayesianAnalyzer(sample_results, analysis_config=config.analysis)
    hdi_prob = config.analysis.hdi_probability if config.analysis else 0.94
    cal_result = analyzer.estimate_judge_calibration(
        oracle_name,
        expected_score=config.expected_score,
        hdi_prob=hdi_prob,
    )

    return scores, cal_result


def _build_optimizer(config: CalibrateConfig) -> RubricOptimizer:
    """Construct a RubricOptimizer from the auto_calibration config."""
    auto_config = config.auto_calibration
    adapter = get_adapter(
        auto_config.optimizer_adapter.name, **auto_config.optimizer_adapter.params
    )
    return RubricOptimizer(model=auto_config.optimizer_model, adapter=adapter)


class AutoCalibrationLoop:
    """Orchestrates the iterative rubric optimization loop.

    Each iteration:
    1. Runs the judge against the fixed prompt/response (N repeats)
    2. Runs Bayesian analysis to measure bias and noise
    3. Checks convergence (HDI contains target + mean within tolerance)
    4. If not converged, asks the optimizer LLM to revise the rubric
    5. Repeats with the new rubric

    Tracks the best rubric (lowest absolute bias) across all iterations,
    since later iterations may regress.

    Args:
        config: CalibrateConfig with auto_calibration settings.
    """

    def __init__(self, config: CalibrateConfig):
        self.config = config
        auto = config.auto_calibration
        self.checker = ConvergenceChecker(
            expected_score=config.expected_score,
            tolerance=auto.tolerance,
            max_iterations=auto.max_iterations,
        )
        self.optimizer = _build_optimizer(config)

    async def run(self) -> AutoCalibrationResult:
        """Execute the auto-calibration loop.

        Returns:
            AutoCalibrationResult with convergence status, final/best rubrics,
            and per-iteration history.
        """
        original_rubric = self.config.oracle.rubric
        current_config = self.config
        history = []

        best_rubric = original_rubric
        best_cal_result = {}
        best_abs_bias = float("inf")

        convergence = None

        for iteration in range(1, self.checker.max_iterations + 1):
            scores, cal_result = await _run_single_calibration(current_config)

            convergence = self.checker.check(cal_result, iteration)

            # Track the best-performing rubric
            abs_bias = abs(cal_result["bias_mean"])
            if abs_bias < best_abs_bias:
                best_abs_bias = abs_bias
                best_rubric = current_config.oracle.rubric
                best_cal_result = cal_result

            history.append(
                {
                    "iteration": iteration,
                    "rubric": current_config.oracle.rubric,
                    "scores": scores,
                    "cal_result": cal_result,
                }
            )

            if convergence.converged:
                break

            # Optimize the rubric for the next iteration
            new_rubric = await self.optimizer.optimize(
                current_rubric=current_config.oracle.rubric,
                expected_score=self.config.expected_score,
                cal_result=cal_result,
                iteration_history=history,
            )

            # Create updated config with the new rubric
            new_oracle = current_config.oracle.model_copy(update={"rubric": new_rubric})
            current_config = current_config.model_copy(update={"oracle": new_oracle})

        return AutoCalibrationResult(
            converged=convergence.reason == "converged",
            iterations=convergence.iteration,
            final_rubric=current_config.oracle.rubric,
            original_rubric=original_rubric,
            convergence_result=convergence,
            history=history,
            best_rubric=best_rubric,
            best_cal_result=best_cal_result,
        )
