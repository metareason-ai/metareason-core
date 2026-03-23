"""Convergence checking for auto-calibration.

Decides whether the judge has converged on the expected score based on
Bayesian analysis results (HDI and mean score).
"""

from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Outcome of a convergence check for a single iteration.

    Attributes:
        converged: Whether the loop should stop (either truly converged or at limit).
        reason: Why the loop stopped — "converged", "max_iterations", or "not_converged".
        hdi_contains_target: Whether the score HDI contains the expected score.
        mean_within_tolerance: Whether the raw mean is close enough to the target.
        current_mean: The judge's raw mean score this iteration.
        current_hdi: The score HDI (not bias HDI) for this iteration.
        iteration: Which iteration this check was performed on.
    """

    converged: bool
    reason: str
    hdi_contains_target: bool
    mean_within_tolerance: bool
    current_mean: float
    current_hdi: tuple[float, float]
    iteration: int


class ConvergenceChecker:
    """Determines whether a judge has converged on the expected score.

    Convergence requires BOTH:
    - The score HDI contains the expected score (statistical guarantee)
    - The raw mean score is within tolerance (practical accuracy)

    This dual condition prevents false convergence from wide-but-biased posteriors.

    Args:
        expected_score: The target score the judge should converge toward.
        tolerance: Maximum acceptable distance between mean score and target.
        max_iterations: Hard stop — the loop ends here regardless of convergence.
    """

    def __init__(self, expected_score: float, tolerance: float, max_iterations: int):
        self.expected_score = expected_score
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def check(self, cal_result: dict, iteration: int) -> ConvergenceResult:
        """Evaluate whether the judge has converged this iteration.

        Args:
            cal_result: Dict from BayesianAnalyzer.estimate_judge_calibration().
                Must contain 'bias_hdi' (tuple) and 'raw_score_mean' (float).
            iteration: Current iteration number (1-indexed).

        Returns:
            ConvergenceResult with the verdict and diagnostic details.
        """
        # Convert bias HDI to score HDI: the actual range of scores the judge produces
        bias_hdi = cal_result["bias_hdi"]
        score_hdi = (
            self.expected_score + bias_hdi[0],
            self.expected_score + bias_hdi[1],
        )
        raw_score_mean = cal_result["raw_score_mean"]

        hdi_ok = score_hdi[0] <= self.expected_score <= score_hdi[1]
        mean_ok = abs(raw_score_mean - self.expected_score) <= self.tolerance
        at_limit = iteration >= self.max_iterations

        # True convergence takes priority — even at the iteration limit,
        # report "converged" if conditions are met
        if hdi_ok and mean_ok:
            reason = "converged"
        elif at_limit:
            reason = "max_iterations"
        else:
            reason = "not_converged"

        return ConvergenceResult(
            converged=reason != "not_converged",
            reason=reason,
            hdi_contains_target=hdi_ok,
            mean_within_tolerance=mean_ok,
            current_mean=raw_score_mean,
            current_hdi=score_hdi,
            iteration=iteration,
        )
