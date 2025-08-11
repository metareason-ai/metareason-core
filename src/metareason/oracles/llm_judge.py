"""LLM-based judge oracle implementation."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from ..adapters.base import CompletionRequest, LLMAdapter, Message, MessageRole
from ..adapters.registry import AdapterFactory
from ..config.adapters import AdapterConfigType
from ..config.oracles import LLMJudgeOracle as LLMJudgeConfig
from .base import BaseOracle, OracleError, OracleResult
from .judge_response import (
    BinaryJudgeResponse,
    JudgeResponseType,
    JudgeResult,
    NumericJudgeResponse,
    StructuredJudgeResponse,
)

logger = logging.getLogger(__name__)


class LLMJudgeOracle(BaseOracle):
    """LLM-based judge oracle for evaluating responses."""

    def __init__(
        self, config: LLMJudgeConfig, adapter_config: Optional[AdapterConfigType] = None
    ):
        """Initialize LLM judge oracle.

        Args:
            config: LLM judge configuration
            adapter_config: LLM adapter configuration
        """
        super().__init__()
        self.config = config
        self.adapter_config = adapter_config
        self.adapter: Optional[LLMAdapter] = None
        self._judges: List[str] = []
        self._current_judge_index = 0

        # Initialize judge models list (support for multiple judges via config extension)
        self._judges = [config.judge_model]

    async def initialize(self) -> None:
        """Initialize the LLM adapter."""
        if not self.adapter:
            # Use primary adapter configuration if available
            if self.adapter_config:
                self.adapter = AdapterFactory.create(self.adapter_config)
                await self.adapter.initialize()

    async def cleanup(self) -> None:
        """Cleanup adapter resources."""
        if self.adapter:
            await self.adapter.cleanup()

    def add_judge(self, judge_model: str) -> None:
        """Add additional judge model for rotation.

        Args:
            judge_model: Additional judge model identifier
        """
        if judge_model not in self._judges:
            self._judges.append(judge_model)

    def get_name(self) -> str:
        """Get oracle name."""
        return f"llm_judge_{self.config.judge_model}"

    async def evaluate(self, response: str, context: Dict[str, Any]) -> OracleResult:
        """Evaluate response using LLM judge.

        Args:
            response: Response to evaluate
            context: Context including prompt, rubric, etc.

        Returns:
            Oracle result with judge score
        """
        if not self.adapter:
            raise OracleError("LLM adapter not initialized")

        # Get current judge model (with rotation support)
        judge_model = self._get_current_judge()

        # Build judge prompt
        judge_prompt = self._build_judge_prompt(response, context)

        # Create completion request with optimized temperature
        request = CompletionRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=self._get_system_prompt()),
                Message(role=MessageRole.USER, content=judge_prompt),
            ],
            model=judge_model,
            temperature=self.config.temperature,
            max_tokens=1000,
        )

        try:
            # Get judge response
            completion = await self.adapter.complete(request)
            raw_response = completion.content.strip()

            # Parse judge response
            judge_result = self._parse_judge_response(raw_response, judge_model)

            # Create oracle result
            oracle_result = OracleResult(
                score=judge_result.response.score,
                metadata={
                    "judge_result": judge_result.model_dump(),
                    "output_format": self.config.output_format,
                    "reasoning": judge_result.response.reasoning,
                },
            )

            # Rotate judge if multiple judges available
            self._rotate_judge()

            return oracle_result

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            raise OracleError(f"Judge evaluation failed: {e}") from e

    def _get_current_judge(self) -> str:
        """Get current judge model with rotation support."""
        return self._judges[self._current_judge_index]

    def _rotate_judge(self) -> None:
        """Rotate to next judge model for bias reduction."""
        if len(self._judges) > 1:
            self._current_judge_index = (self._current_judge_index + 1) % len(
                self._judges
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for judge."""
        format_instructions = self._get_format_instructions()

        return f"""You are an expert evaluator tasked with judging the quality of responses based on specific criteria.

Your job is to:
1. Carefully read the evaluation rubric
2. Analyze the response against each criterion
3. Provide your judgment in the exact JSON format specified
4. Include clear reasoning for your decision

Be consistent, fair, and thorough in your evaluation. Focus on the specific criteria provided.

{format_instructions}"""

    def _get_format_instructions(self) -> str:
        """Get format-specific instructions for judge output."""
        if self.config.output_format == "binary":
            return """
OUTPUT FORMAT:
Provide your response as valid JSON with this exact structure:
{
    "score": 1,  // 1 for pass, 0 for fail
    "reasoning": "Your detailed explanation here"
}"""
        elif self.config.output_format == "score":
            return """
OUTPUT FORMAT:
Provide your response as valid JSON with this exact structure:
{
    "score": 0.85,  // Float between 0.0 and 1.0
    "reasoning": "Your detailed explanation here"
}"""
        else:  # structured
            return """
OUTPUT FORMAT:
Provide your response as valid JSON with this exact structure:
{
    "score": 0.85,  // Overall float score 0.0-1.0
    "reasoning": "Your overall explanation",
    "dimensions": {
        "criterion_1": 0.9,  // Individual scores 0.0-1.0
        "criterion_2": 0.8
    },
    "details": {  // Optional additional structured details
        "strengths": ["strength 1", "strength 2"],
        "weaknesses": ["weakness 1"]
    }
}"""

    def _build_judge_prompt(self, response: str, context: Dict[str, Any]) -> str:
        """Build judge prompt with rubric and few-shot examples."""
        prompt_parts = []

        # Add rubric
        prompt_parts.append("EVALUATION RUBRIC:")
        prompt_parts.append(self.config.rubric)
        prompt_parts.append("")

        # Add original prompt if available
        if "original_prompt" in context:
            prompt_parts.append("ORIGINAL PROMPT:")
            prompt_parts.append(context["original_prompt"])
            prompt_parts.append("")

        # Add few-shot examples if configured
        examples = self._get_few_shot_examples()
        if examples:
            prompt_parts.append("EXAMPLES:")
            prompt_parts.extend(examples)
            prompt_parts.append("")

        # Add chain-of-thought instruction
        if self._should_use_chain_of_thought():
            prompt_parts.append("Think through your evaluation step by step:")
            prompt_parts.append("1. Identify the key aspects to evaluate")
            prompt_parts.append("2. Analyze the response against each criterion")
            prompt_parts.append("3. Consider strengths and weaknesses")
            prompt_parts.append("4. Determine final score and reasoning")
            prompt_parts.append("")

        # Add response to evaluate
        prompt_parts.append("RESPONSE TO EVALUATE:")
        prompt_parts.append(f'"""{response}"""')
        prompt_parts.append("")

        prompt_parts.append("Provide your evaluation in the specified JSON format:")

        return "\n".join(prompt_parts)

    def _get_few_shot_examples(self) -> List[str]:
        """Get few-shot examples for consistency."""
        # Default examples based on output format
        examples = []

        if self.config.output_format == "binary":
            examples.extend(
                [
                    "Example Response: 'The capital of France is Paris.'",
                    "Example Evaluation:",
                    '{"score": 1, "reasoning": "Provides accurate, direct answer to the question."}',
                    "",
                    "Example Response: 'France has many cities and Paris is one of them.'",
                    "Example Evaluation:",
                    '{"score": 0, "reasoning": "Does not directly answer the question asked."}',
                    "",
                ]
            )
        elif self.config.output_format == "score":
            examples.extend(
                [
                    "Example Response: 'The capital of France is Paris, known for the Eiffel Tower.'",
                    "Example Evaluation:",
                    '{"score": 0.95, "reasoning": "Accurate answer with helpful additional context."}',
                    "",
                    "Example Response: 'I think Paris might be the capital of France.'",
                    "Example Evaluation:",
                    '{"score": 0.4, "reasoning": "Correct information but expressed with unnecessary uncertainty."}',
                    "",
                ]
            )

        return examples

    def _should_use_chain_of_thought(self) -> bool:
        """Determine if chain-of-thought reasoning should be used."""
        # Use CoT for complex rubrics or structured outputs
        rubric_lines = len(self.config.rubric.split("\n"))
        return rubric_lines > 3 or self.config.output_format == "structured"

    def _parse_judge_response(self, raw_response: str, judge_model: str) -> JudgeResult:
        """Parse judge response with robust JSON extraction and fallbacks."""
        # Try to extract JSON from response
        parsed_response = self._extract_json_from_response(raw_response)

        if not parsed_response:
            # Fallback parsing for malformed responses
            parsed_response = self._fallback_parse_response(raw_response)

        # Validate and create appropriate response object
        judge_response = self._create_judge_response(parsed_response)

        return JudgeResult(
            response=judge_response,
            raw_response=raw_response,
            judge_model=judge_model,
            temperature=self.config.temperature,
            metadata={"parsing_method": "json" if parsed_response else "fallback"},
        )

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with robust parsing."""
        # Try JSON code block first
        json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object with proper brace matching
        def find_json_object(text: str) -> Optional[str]:
            for i, char in enumerate(text):
                if char == "{":
                    brace_count = 0
                    for j in range(i, len(text)):
                        if text[j] == "{":
                            brace_count += 1
                        elif text[j] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                return text[i : j + 1]
            return None

        json_str = find_json_object(response)
        if json_str:
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                pass

        return None

    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for malformed judge responses."""
        logger.warning(f"Using fallback parsing for response: {response[:100]}...")

        # Extract score using various patterns
        score = self._extract_score_fallback(response)

        # Extract reasoning
        reasoning = self._extract_reasoning_fallback(response)

        return {
            "score": score,
            "reasoning": reasoning or "Could not extract clear reasoning from response",
        }

    def _extract_score_fallback(self, response: str) -> Union[int, float]:
        """Extract score using fallback patterns."""
        # Binary patterns
        if self.config.output_format == "binary":
            if any(
                word in response.lower() for word in ["pass", "yes", "correct", "meets"]
            ):
                return 1
            if any(
                word in response.lower()
                for word in ["fail", "no", "incorrect", "does not meet"]
            ):
                return 0

            # Look for numeric 1 or 0
            score_match = re.search(r"\b[01]\b", response)
            if score_match:
                return int(score_match.group())

            # Default to fail for unclear responses
            return 0

        # Numeric score patterns
        score_patterns = [
            r"score[:\s]*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*/\s*1",
            r"([0-9]*\.?[0-9]+)\s*out\s*of\s*1",
            r"rating[:\s]*([0-9]*\.?[0-9]+)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))  # Clamp to [0,1]
                except ValueError:
                    continue

        # Default to neutral score
        return 0.5

    def _extract_reasoning_fallback(self, response: str) -> Optional[str]:
        """Extract reasoning using fallback patterns."""
        # Look for reasoning indicators
        reasoning_patterns = [
            r"reasoning[:\s]*(.*?)(?:\n|$)",
            r"because[:\s]*(.*?)(?:\n|$)",
            r"explanation[:\s]*(.*?)(?:\n|$)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 10:  # Ensure meaningful reasoning
                    return reasoning

        # Return cleaned response if no specific reasoning found
        cleaned = re.sub(r'[{}"\[\]]', "", response).strip()
        return cleaned if len(cleaned) > 10 else None

    def _create_judge_response(
        self, response_data: Dict[str, Any]
    ) -> JudgeResponseType:
        """Create appropriate judge response object based on format."""
        try:
            if self.config.output_format == "binary":
                return BinaryJudgeResponse(**response_data)
            elif self.config.output_format == "score":
                return NumericJudgeResponse(**response_data)
            else:  # structured
                return StructuredJudgeResponse(**response_data)
        except Exception as e:
            logger.warning(f"Failed to create structured response: {e}")
            # Fallback to basic response
            score = response_data.get("score", 0.5)
            reasoning = response_data.get("reasoning", "No reasoning provided")

            if self.config.output_format == "binary":
                return BinaryJudgeResponse(
                    score=1 if score > 0.5 else 0, reasoning=reasoning
                )
            else:
                return NumericJudgeResponse(score=float(score), reasoning=reasoning)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
