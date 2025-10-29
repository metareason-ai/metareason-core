"""LLM Judge Oracle for evaluating LLM responses.

This module provides the LlmJudge oracle implementation, which uses a Language
Model as a judge to evaluate other LLM outputs. The judge model receives the
original prompt, the LLM's response, and a rubric, then provides a structured
evaluation with a score and explanation.

The implementation uses configurable adapters for communication with the judge
model and expects JSON-formatted responses containing evaluation results.

Classes:
    LLMJudge: Oracle that uses an LLM to judge other LLM responses based on a rubric.
"""

import json

from metareason.adapters.adapter_base import AdapterResponse
from metareason.adapters.adapter_factory import get_adapter

from ..adapters import AdapterRequest
from ..config import OracleConfig
from .oracle_base import (
    EvaluationContext,
    EvaluationResult,
    OracleBase,
    OracleException,
)


class LLMJudge(OracleBase):
    """LLM-based judge oracle for evaluating LLM responses using another LLM.

    This oracle uses a Language Model as a judge to evaluate LLM outputs based on
    a provided rubric. The judge LLM analyzes both the original prompt and the
    response, then provides a score and explanation according to the rubric criteria.

    The evaluation is performed by a configurable adapter that sends a structured
    prompt to the judge model, expecting a JSON response with score and explanation.

    Attributes:
        adapter (AdapterBase): The adapter used to communicate with the judge LLM.
        sys_prompt (str): The system prompt that includes evaluation instructions and rubric.

    Raises:
        ValueError: If the config doesn't contain a required rubric.

    Example:
        >>> config = OracleConfig(
        ...     rubric="Score 1-5 based on accuracy and clarity",
        ...     model="llama2",
        ...     temperature=0.7
        ... )
        >>> judge = LlmJudge(config)
        >>> context = EvaluationContext(
        ...     prompt="What is 2+2?",
        ...     response="The answer is 4"
        ... )
        >>> result = await judge.evaluate(context)
        >>> print(result.score, result.explanation)
    """

    def __init__(self, config: OracleConfig):
        """Initialize the LLM judge oracle.

        Args:
            config (OracleConfig): Configuration object containing rubric, model,
                temperature, and other evaluation parameters.

        Raises:
            ValueError: If the config doesn't contain a rubric.
        """
        super().__init__(config)

        if not self.config.rubric:
            raise (ValueError("Rubric is required for LLM judge."))

        self.adapter = get_adapter(config.adapter.name, **config.adapter.params)
        self.sys_prompt = f"""
            You are an expert evaluator of LLM responses. Use the following rubric to evaluate the prompt and response from the LLM.
            Respond with a JSON object that contains the score and an explanation.

            Example Response:
            {{
                "score": 5,
                "explanation": "The response is perfect and meets all criteria."
            }}

            Rubric:
            {self.config.rubric}
            """

    async def evaluate(self, request: EvaluationContext) -> EvaluationResult:
        """Evaluate an LLM response using the configured judge model and rubric.

        This method sends the original prompt and LLM response to a judge model,
        which evaluates the quality based on the configured rubric. The judge
        returns a structured JSON response containing a numerical score and
        a text explanation of the evaluation.

        Args:
            request (EvaluationContext): Context containing the original prompt
                and the LLM response to be evaluated.

        Returns:
            EvaluationResult: An object containing the numerical score and
                explanation from the judge model.

        Raises:
            OracleException: If the evaluation fails due to adapter errors,
                JSON parsing issues, or missing fields in the response.

        Example:
            >>> context = EvaluationContext(
            ...     prompt="Explain photosynthesis",
            ...     response="Plants use sunlight to make food"
            ... )
            >>> result = await judge.evaluate(context)
            >>> print(f"Score: {result.score}, Reason: {result.explanation}")
        """
        try:
            user_prompt = f"""
            Original Prompt: {request.prompt}

            LLM Response: {request.response}
            """

            adapter_request = AdapterRequest(
                model=self.config.model,
                system_prompt=self.sys_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
                top_p=1.0,  # Use default value for judge oracle
                max_tokens=self.config.max_tokens,
            )

            adapter_resposne: AdapterResponse = await self.adapter.send_request(
                adapter_request
            )
            cleaned = adapter_resposne.response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0]

            response = json.loads(cleaned)

            if "score" not in response:
                raise OracleException("Judge response missing 'score' field")

            return EvaluationResult(
                score=float(response["score"]),
                explanation=response.get("explanation", "No explanation provided"),
            )

        except json.JSONDecodeError as e:
            raise OracleException(f"Invalid JSON from judge: {cleaned[:200]}", e) from e
        except Exception as e:
            raise OracleException(f"LLM judge evaluation failed: {e}", e) from e
