"""Tests specifying the behavior of the auto-calibration system.

The auto-calibration loop iteratively improves a judge's rubric by:
1. Running repeated evaluations + Bayesian analysis (existing infrastructure)
2. Checking if the judge has converged on the expected score
3. If not, asking an optimizer LLM to revise the rubric
4. Repeating until convergence or max iterations
"""

from unittest.mock import AsyncMock, patch

import pytest

from metareason.adapters.adapter_base import AdapterBase
from metareason.config.models import (
    AdapterConfig,
    AutoCalibrationConfig,
    BayesianAnalysisConfig,
    CalibrateConfig,
    OracleConfig,
)

# --- Helpers ---


def make_oracle(**overrides):
    defaults = dict(
        type="llm_judge",
        model="gemma3:27b",
        adapter=AdapterConfig(name="ollama"),
        rubric='Score 1-5 on clarity. Return: {"score": X, "explanation": "..."}',
    )
    defaults.update(overrides)
    return OracleConfig(**defaults)


def make_calibrate_config(**overrides):
    defaults = dict(
        spec_id="auto-cal-test",
        prompt="Explain quantum entanglement.",
        response="Quantum entanglement is when two particles become linked.",
        expected_score=4.0,
        repeats=10,
        oracle=make_oracle(),
        analysis=BayesianAnalysisConfig(mcmc_draws=500, mcmc_chains=2),
        auto_calibration=AutoCalibrationConfig(
            optimizer_model="claude-sonnet-4-6",
            optimizer_adapter=AdapterConfig(name="anthropic"),
            max_iterations=5,
            tolerance=0.3,
        ),
    )
    defaults.update(overrides)
    return CalibrateConfig(**defaults)


def make_cal_result(
    expected_score=4.0,
    bias_mean=0.0,
    bias_hdi=(-0.2, 0.2),
    noise_mean=0.3,
    raw_score_mean=4.0,
):
    """Build a calibration result dict matching BayesianAnalyzer.estimate_judge_calibration output."""
    return {
        "expected_score": expected_score,
        "bias_mean": bias_mean,
        "bias_median": bias_mean,
        "bias_hdi": bias_hdi,
        "noise_mean": noise_mean,
        "noise_hdi": (0.1, 0.5),
        "n_samples": 10,
        "hdi_prob": 0.94,
        "raw_score_mean": raw_score_mean,
        "raw_score_std": 0.4,
        "bias_posterior_samples": [bias_mean] * 100,
        "noise_posterior_samples": [noise_mean] * 100,
    }


# =============================================================================
# ConvergenceChecker: decides whether to stop iterating
# =============================================================================


class TestConvergenceChecker:
    """Convergence checker decides whether to stop iterating.

    Convergence requires BOTH:
    - The score HDI (expected_score + bias_hdi) contains expected_score
    - The raw mean score is within tolerance of expected_score

    This dual condition prevents false convergence from wide-but-biased posteriors.
    """

    def test_converged_when_hdi_contains_target_and_mean_within_tolerance(self):
        """Judge is accurate and precise -- stop iterating."""
        from metareason.calibration.convergence import ConvergenceChecker

        checker = ConvergenceChecker(
            expected_score=4.0, tolerance=0.3, max_iterations=10
        )
        cal_result = make_cal_result(
            bias_mean=0.1, bias_hdi=(-0.2, 0.3), raw_score_mean=4.1
        )

        result = checker.check(cal_result, iteration=1)

        assert result.converged is True
        assert result.reason == "converged"
        assert result.hdi_contains_target is True
        assert result.mean_within_tolerance is True

    def test_not_converged_when_hdi_excludes_target(self):
        """Bias HDI is entirely above zero -- judge consistently scores too high."""
        from metareason.calibration.convergence import ConvergenceChecker

        checker = ConvergenceChecker(
            expected_score=4.0, tolerance=0.3, max_iterations=10
        )
        # bias_hdi (0.5, 1.2) -> score HDI (4.5, 5.2) -- doesn't contain 4.0
        cal_result = make_cal_result(
            bias_mean=0.8, bias_hdi=(0.5, 1.2), raw_score_mean=4.8
        )

        result = checker.check(cal_result, iteration=1)

        assert result.converged is False
        assert result.hdi_contains_target is False

    def test_not_converged_when_mean_outside_tolerance(self):
        """HDI is wide enough to contain target, but mean is far off."""
        from metareason.calibration.convergence import ConvergenceChecker

        checker = ConvergenceChecker(
            expected_score=4.0, tolerance=0.3, max_iterations=10
        )
        # bias_hdi spans zero but mean is -0.8 -> raw_score_mean = 3.2
        cal_result = make_cal_result(
            bias_mean=-0.8, bias_hdi=(-1.5, 0.1), raw_score_mean=3.2
        )

        result = checker.check(cal_result, iteration=1)

        assert result.converged is False
        assert result.hdi_contains_target is True
        assert result.mean_within_tolerance is False

    def test_max_iterations_stops_loop(self):
        """Even if not converged, stop at max iterations."""
        from metareason.calibration.convergence import ConvergenceChecker

        checker = ConvergenceChecker(
            expected_score=4.0, tolerance=0.3, max_iterations=5
        )
        cal_result = make_cal_result(
            bias_mean=0.8, bias_hdi=(0.5, 1.2), raw_score_mean=4.8
        )

        result = checker.check(cal_result, iteration=5)

        assert result.converged is True
        assert result.reason == "max_iterations"
        assert result.hdi_contains_target is False

    def test_reports_current_score_hdi(self):
        """Result includes the score HDI (not bias HDI) for display/logging."""
        from metareason.calibration.convergence import ConvergenceChecker

        checker = ConvergenceChecker(
            expected_score=4.0, tolerance=0.3, max_iterations=10
        )
        cal_result = make_cal_result(bias_hdi=(-0.5, 0.3))

        result = checker.check(cal_result, iteration=1)

        # Score HDI = expected_score + bias_hdi = (3.5, 4.3)
        assert result.current_hdi == (3.5, 4.3)


# =============================================================================
# RubricOptimizer: rewrites the rubric to reduce judge bias
# =============================================================================


class TestRubricOptimizer:
    """Rubric optimizer rewrites rubrics to reduce judge bias."""

    @pytest.mark.asyncio
    async def test_sends_rubric_and_gap_to_optimizer_llm(self):
        """Prompt includes current rubric, expected score, actual mean, and bias."""
        from metareason.calibration.optimizer import RubricOptimizer

        mock_adapter: AdapterBase = AsyncMock(spec=AdapterBase)
        mock_adapter.send_request.return_value = AsyncMock(
            response_text="Revised rubric: evaluate clarity 1-5."
        )

        optimizer = RubricOptimizer(model="claude-sonnet-4-6", adapter=mock_adapter)
        cal_result = make_cal_result(bias_mean=-0.8, raw_score_mean=3.2)

        await optimizer.optimize(
            current_rubric="Score 1-5 on clarity.",
            expected_score=4.0,
            cal_result=cal_result,
            iteration_history=[],
        )

        prompt_sent = mock_adapter.send_request.call_args[0][0].user_prompt
        assert "Score 1-5 on clarity." in prompt_sent
        assert "4.0" in prompt_sent
        assert "3.2" in prompt_sent

    @pytest.mark.asyncio
    async def test_returns_revised_rubric_text(self):
        """The optimizer extracts and returns just the rubric, not wrapper text."""
        from metareason.calibration.optimizer import RubricOptimizer

        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AsyncMock(
            response_text="Evaluate response clarity and accuracy on a 1-5 scale."
        )

        optimizer = RubricOptimizer(model="claude-sonnet-4-6", adapter=mock_adapter)
        cal_result = make_cal_result()

        result = await optimizer.optimize(
            current_rubric="old rubric",
            expected_score=4.0,
            cal_result=cal_result,
            iteration_history=[],
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        """LLMs often wrap output in markdown fences -- strip them."""
        from metareason.calibration.optimizer import RubricOptimizer

        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AsyncMock(
            response_text="```\nRevised rubric here.\n```"
        )

        optimizer = RubricOptimizer(model="claude-sonnet-4-6", adapter=mock_adapter)
        cal_result = make_cal_result()

        result = await optimizer.optimize(
            current_rubric="old",
            expected_score=4.0,
            cal_result=cal_result,
            iteration_history=[],
        )

        assert "```" not in result
        assert "Revised rubric here." in result

    @pytest.mark.asyncio
    async def test_includes_iteration_history_to_prevent_oscillation(self):
        """When previous iterations exist, the optimizer sees what was already tried."""
        from metareason.calibration.optimizer import RubricOptimizer

        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AsyncMock(
            response_text="Third attempt rubric."
        )

        optimizer = RubricOptimizer(model="claude-sonnet-4-6", adapter=mock_adapter)
        cal_result = make_cal_result()
        history = [
            {
                "iteration": 1,
                "rubric": "First rubric",
                "cal_result": {"bias_mean": -0.8},
            },
            {
                "iteration": 2,
                "rubric": "Second rubric",
                "cal_result": {"bias_mean": 0.5},
            },
        ]

        await optimizer.optimize(
            current_rubric="Second rubric",
            expected_score=4.0,
            cal_result=cal_result,
            iteration_history=history,
        )

        prompt_sent = mock_adapter.send_request.call_args[0][0].user_prompt
        assert "First rubric" in prompt_sent
        assert "Second rubric" in prompt_sent

    @pytest.mark.asyncio
    async def test_raises_on_adapter_failure(self):
        """Adapter errors propagate -- the loop decides how to handle them."""
        from metareason.calibration.optimizer import RubricOptimizer

        mock_adapter = AsyncMock()
        mock_adapter.send_request.side_effect = RuntimeError("API timeout")

        optimizer = RubricOptimizer(model="claude-sonnet-4-6", adapter=mock_adapter)
        cal_result = make_cal_result()

        with pytest.raises(RuntimeError, match="API timeout"):
            await optimizer.optimize(
                current_rubric="rubric",
                expected_score=4.0,
                cal_result=cal_result,
                iteration_history=[],
            )


# =============================================================================
# AutoCalibrationLoop: orchestrates the evaluate -> check -> optimize cycle
# =============================================================================


class TestAutoCalibrationLoop:
    """Auto-calibration loop orchestrates evaluate, check, and optimize cycle."""

    @pytest.mark.asyncio
    async def test_stops_immediately_when_already_converged(self):
        """If the judge is already accurate, no optimization happens."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config()
        converged_result = make_cal_result(
            bias_mean=0.1,
            bias_hdi=(-0.2, 0.3),
            raw_score_mean=4.1,
        )

        with patch(
            "metareason.calibration.loop._run_single_calibration",
            return_value=([4.0, 4.1, 3.9, 4.2], converged_result),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert result.converged is True
        assert result.iterations == 1
        assert result.final_rubric == config.oracle.rubric

    @pytest.mark.asyncio
    async def test_optimizes_rubric_when_not_converged(self):
        """When the judge is biased, the loop calls the optimizer and tries again."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config()
        biased_result = make_cal_result(
            bias_mean=-0.8,
            bias_hdi=(-1.2, -0.4),
            raw_score_mean=3.2,
        )
        converged_result = make_cal_result(
            bias_mean=0.1,
            bias_hdi=(-0.2, 0.3),
            raw_score_mean=4.1,
        )

        call_count = 0

        async def mock_calibration(cfg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [3.0, 3.2, 3.4], biased_result
            return [4.0, 4.1, 3.9], converged_result

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "Improved rubric text"

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                side_effect=mock_calibration,
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert result.converged is True
        assert result.iterations == 2
        mock_optimizer.optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_rubric_changes_between_iterations(self):
        """Each iteration after optimization uses the new rubric, not the original."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config()
        biased = make_cal_result(
            bias_mean=-0.8,
            bias_hdi=(-1.2, -0.4),
            raw_score_mean=3.2,
        )
        converged = make_cal_result(
            bias_mean=0.1,
            bias_hdi=(-0.2, 0.3),
            raw_score_mean=4.1,
        )

        configs_received = []

        async def mock_calibration(cfg):
            configs_received.append(cfg)
            if len(configs_received) == 1:
                return [3.0, 3.2], biased
            return [4.0, 4.1], converged

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "Better rubric"

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                side_effect=mock_calibration,
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            await loop.run()

        assert configs_received[0].oracle.rubric == config.oracle.rubric
        assert configs_received[1].oracle.rubric == "Better rubric"

    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self):
        """Never exceeds max_iterations, even if never converged."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config(
            auto_calibration=AutoCalibrationConfig(
                optimizer_model="claude-sonnet-4-6",
                optimizer_adapter=AdapterConfig(name="anthropic"),
                max_iterations=3,
                tolerance=0.3,
            ),
        )
        biased = make_cal_result(
            bias_mean=-0.8,
            bias_hdi=(-1.2, -0.4),
            raw_score_mean=3.2,
        )

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "Attempt N"

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                return_value=([3.0, 3.2], biased),
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert result.iterations == 3
        assert result.convergence_result.reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_returns_best_rubric_not_last(self):
        """When max iterations hit, return the rubric with lowest absolute bias."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config(
            auto_calibration=AutoCalibrationConfig(
                optimizer_model="claude-sonnet-4-6",
                optimizer_adapter=AdapterConfig(name="anthropic"),
                max_iterations=3,
                tolerance=0.1,
            ),
        )

        results_sequence = [
            make_cal_result(bias_mean=-0.8, bias_hdi=(-1.2, -0.4), raw_score_mean=3.2),
            make_cal_result(bias_mean=0.2, bias_hdi=(-0.1, 0.5), raw_score_mean=4.2),
            make_cal_result(bias_mean=-0.5, bias_hdi=(-0.9, -0.1), raw_score_mean=3.5),
        ]
        call_count = 0

        async def mock_calibration(cfg):
            nonlocal call_count
            r = results_sequence[call_count]
            call_count += 1
            return [3.0, 3.2], r

        # Iter 1: original rubric → bias -0.8 → optimizer produces "Rubric A"
        # Iter 2: "Rubric A"    → bias  0.2 → optimizer produces "Rubric B"
        # Iter 3: "Rubric B"    → bias -0.5 → max iterations
        # Best result is iter 2, which used "Rubric A"
        rubrics = iter(["Rubric A", "Rubric B", "Rubric C"])

        async def mock_optimize(*args, **kwargs):
            return next(rubrics)

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.side_effect = mock_optimize

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                side_effect=mock_calibration,
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert result.best_rubric == "Rubric A"
        assert abs(result.best_cal_result["bias_mean"]) < abs(
            results_sequence[0]["bias_mean"]
        )

    @pytest.mark.asyncio
    async def test_history_tracks_each_iteration(self):
        """Every iteration's rubric, scores, and analysis are recorded."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config(
            auto_calibration=AutoCalibrationConfig(
                optimizer_model="claude-sonnet-4-6",
                optimizer_adapter=AdapterConfig(name="anthropic"),
                max_iterations=2,
                tolerance=0.3,
            ),
        )
        biased = make_cal_result(
            bias_mean=-0.8,
            bias_hdi=(-1.2, -0.4),
            raw_score_mean=3.2,
        )

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "New rubric"

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                return_value=([3.0, 3.2], biased),
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert len(result.history) == 2
        assert result.history[0]["iteration"] == 1
        assert result.history[1]["iteration"] == 2
        assert "rubric" in result.history[0]
        assert "scores" in result.history[0]
        assert "cal_result" in result.history[0]

    @pytest.mark.asyncio
    async def test_preserves_original_rubric(self):
        """The original rubric is always accessible, regardless of outcome."""
        from metareason.calibration.loop import AutoCalibrationLoop

        config = make_calibrate_config()
        original_rubric = config.oracle.rubric
        biased = make_cal_result(
            bias_mean=-0.8,
            bias_hdi=(-1.2, -0.4),
            raw_score_mean=3.2,
        )
        converged = make_cal_result(
            bias_mean=0.1,
            bias_hdi=(-0.2, 0.3),
            raw_score_mean=4.1,
        )

        call_count = 0

        async def mock_calibration(cfg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [3.0], biased
            return [4.0], converged

        mock_optimizer = AsyncMock()
        mock_optimizer.optimize.return_value = "Changed rubric"

        with (
            patch(
                "metareason.calibration.loop._run_single_calibration",
                side_effect=mock_calibration,
            ),
            patch(
                "metareason.calibration.loop._build_optimizer",
                return_value=mock_optimizer,
            ),
        ):
            loop = AutoCalibrationLoop(config)
            result = await loop.run()

        assert result.original_rubric == original_rubric
        assert result.final_rubric == "Changed rubric"
