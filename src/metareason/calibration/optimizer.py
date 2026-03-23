"""Rubric optimizer for auto-calibration.

Uses an LLM to revise a judge's rubric based on gap analysis between
the judge's actual scores and the expected score.
"""

import re

from metareason.adapters.adapter_base import AdapterBase, AdapterRequest


class RubricOptimizer:
    """Rewrites a judge's rubric to reduce scoring bias.

    Sends the current rubric, calibration results, and iteration history
    to an optimizer LLM, which returns a revised rubric that should bring
    the judge's scores closer to the expected score.

    Args:
        model: The model identifier for the optimizer LLM.
        adapter: An initialized adapter for communicating with the optimizer LLM.
    """

    SYSTEM_PROMPT = (
        "You are a rubric optimization expert. Your job is to revise evaluation "
        "rubrics so that an LLM judge's scores align with a target score.\n\n"
        "Output ONLY the revised rubric text. No explanations, no commentary, "
        "no markdown fences."
    )

    def __init__(self, model: str, adapter: AdapterBase):
        self.model = model
        self.adapter = adapter

    async def optimize(
        self,
        current_rubric: str,
        expected_score: float,
        cal_result: dict,
        iteration_history: list[dict],
    ) -> str:
        """Generate a revised rubric based on calibration gap analysis.

        Args:
            current_rubric: The rubric text currently used by the judge.
            expected_score: The target score the judge should produce.
            cal_result: Dict from BayesianAnalyzer.estimate_judge_calibration().
            iteration_history: Previous iterations' rubrics and results,
                used to prevent oscillation and repeated failures.

        Returns:
            The revised rubric text.

        Raises:
            Any exception from the adapter (e.g. API timeout, auth error).
        """
        prompt = self._build_prompt(
            current_rubric, expected_score, cal_result, iteration_history
        )
        request = AdapterRequest(
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2000,
        )
        response = await self.adapter.send_request(request)
        return self._strip_fences(response.response_text)

    def _build_prompt(
        self,
        current_rubric: str,
        expected_score: float,
        cal_result: dict,
        iteration_history: list[dict],
    ) -> str:
        """Build the optimization prompt with rubric, gap analysis, and history."""
        raw_mean = cal_result["raw_score_mean"]
        bias_mean = cal_result["bias_mean"]

        direction = "higher" if bias_mean > 0 else "lower"
        magnitude = abs(bias_mean)

        prompt = (
            f"## Current Rubric\n{current_rubric}\n\n"
            f"## Gap Analysis\n"
            f"- Target score: {expected_score}\n"
            f"- Actual mean score: {raw_mean}\n"
            f"- Bias: {bias_mean:+.2f} (judge scores {direction} than target by {magnitude:.2f})\n"
        )

        if iteration_history:
            prompt += "\n## Previous Attempts\n"
            for entry in iteration_history:
                prompt += (
                    f"\n### Iteration {entry['iteration']}\n"
                    f"Rubric: {entry['rubric']}\n"
                    f"Bias: {entry['cal_result']['bias_mean']:+.2f}\n"
                )

        prompt += (
            f"\nRevise the rubric so the judge's scores center on {expected_score}. "
            f"Output ONLY the revised rubric."
        )

        return prompt

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences that LLMs often wrap around output."""
        stripped = re.sub(r"^```\w*\n?", "", text.strip())
        stripped = re.sub(r"\n?```$", "", stripped)
        return stripped.strip()
