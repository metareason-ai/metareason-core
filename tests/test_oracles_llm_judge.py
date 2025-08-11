"""Tests for LLM judge oracle implementation."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from metareason.adapters.base import CompletionResponse
from metareason.config.oracles import LLMJudgeOracle as LLMJudgeConfig
from metareason.oracles.base import OracleError
from metareason.oracles.judge_response import BinaryJudgeResponse, NumericJudgeResponse
from metareason.oracles.llm_judge import LLMJudgeOracle


class TestLLMJudgeOracle:
    """Test LLM judge oracle implementation."""

    @pytest.fixture
    def binary_config(self):
        """Create binary judge configuration."""
        return LLMJudgeConfig(
            rubric="1. Answers the question directly\n2. Uses appropriate tone",
            judge_model="gpt-4",
            temperature=0.0,
            output_format="binary",
        )

    @pytest.fixture
    def score_config(self):
        """Create score judge configuration."""
        return LLMJudgeConfig(
            rubric=(
                "1. Response accuracy and factual correctness\n"
                "2. Clarity and comprehensibility of explanation\n"
                "3. Appropriate level of detail and completeness"
            ),
            judge_model="gpt-4",
            temperature=0.1,
            output_format="score",
        )

    @pytest.fixture
    def structured_config(self):
        """Create structured judge configuration."""
        return LLMJudgeConfig(
            rubric=(
                "Evaluate response across multiple dimensions:\n"
                "1. Accuracy: Factual correctness and reliability\n"
                "2. Clarity: Clear expression and comprehensibility\n"
                "3. Completeness: Adequate coverage of the topic"
            ),
            judge_model="gpt-4",
            temperature=0.0,
            output_format="structured",
        )

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter."""
        adapter = AsyncMock()
        adapter.initialize = AsyncMock()
        adapter.cleanup = AsyncMock()
        return adapter

    def test_oracle_initialization(self, binary_config):
        """Test oracle initialization."""
        oracle = LLMJudgeOracle(binary_config)

        assert oracle.config == binary_config
        assert oracle.get_name() == "llm_judge_gpt-4"
        assert oracle._judges == ["gpt-4"]
        assert oracle._current_judge_index == 0

    def test_add_judge_functionality(self):
        """Test adding additional judges."""
        config = LLMJudgeConfig(
            rubric=(
                "1. Response answers the question directly\n"
                "2. Uses appropriate professional tone\n"
                "3. Provides sufficient detail"
            ),
            judge_model="gpt-4",
            output_format="binary",
        )

        oracle = LLMJudgeOracle(config)
        assert oracle._judges == ["gpt-4"]

        # Add additional judges
        oracle.add_judge("claude-3")
        oracle.add_judge("gpt-3.5-turbo")
        assert oracle._judges == ["gpt-4", "claude-3", "gpt-3.5-turbo"]

        # Don't add duplicates
        oracle.add_judge("gpt-4")
        assert oracle._judges == ["gpt-4", "claude-3", "gpt-3.5-turbo"]

    def test_judge_rotation(self, binary_config):
        """Test judge rotation for bias reduction."""
        oracle = LLMJudgeOracle(binary_config)
        oracle.add_judge("claude-3")

        # Initial judge
        assert oracle._get_current_judge() == "gpt-4"

        # After rotation
        oracle._rotate_judge()
        assert oracle._get_current_judge() == "claude-3"

        # Back to first judge
        oracle._rotate_judge()
        assert oracle._get_current_judge() == "gpt-4"

    def test_single_judge_no_rotation(self, binary_config):
        """Test no rotation with single judge."""
        oracle = LLMJudgeOracle(binary_config)

        initial_judge = oracle._get_current_judge()
        oracle._rotate_judge()
        assert oracle._get_current_judge() == initial_judge

    @pytest.mark.asyncio
    async def test_evaluate_binary_success(self, binary_config, mock_adapter):
        """Test successful binary evaluation."""
        # Mock adapter response
        mock_response = CompletionResponse(
            content='{"score": 1, "reasoning": "Response answers directly and uses appropriate tone."}',
            model="gpt-4",
        )
        mock_adapter.complete = AsyncMock(return_value=mock_response)

        oracle = LLMJudgeOracle(binary_config)
        oracle.adapter = mock_adapter

        result = await oracle.evaluate(
            "The capital of France is Paris.",
            {"original_prompt": "What is the capital of France?"},
        )

        assert result.score == 1
        assert "judge_result" in result.metadata
        assert result.metadata["output_format"] == "binary"
        assert "Response answers directly" in result.metadata["reasoning"]

    @pytest.mark.asyncio
    async def test_evaluate_score_success(self, score_config, mock_adapter):
        """Test successful score evaluation."""
        mock_response = CompletionResponse(
            content='{"score": 0.85, "reasoning": "Good response with minor issues."}',
            model="gpt-4",
        )
        mock_adapter.complete = AsyncMock(return_value=mock_response)

        oracle = LLMJudgeOracle(score_config)
        oracle.adapter = mock_adapter

        result = await oracle.evaluate(
            "Paris is the capital of France and known for its cuisine.",
            {"original_prompt": "What is the capital of France?"},
        )

        assert result.score == 0.85
        assert result.metadata["output_format"] == "score"

    @pytest.mark.asyncio
    async def test_evaluate_structured_success(self, structured_config, mock_adapter):
        """Test successful structured evaluation."""
        mock_response = CompletionResponse(
            content=json.dumps(
                {
                    "score": 0.8,
                    "reasoning": "Overall good response",
                    "dimensions": {
                        "accuracy": 0.9,
                        "clarity": 0.7,
                        "completeness": 0.8,
                    },
                    "details": {
                        "strengths": ["accurate", "complete"],
                        "improvements": ["could be clearer"],
                    },
                }
            ),
            model="gpt-4",
        )
        mock_adapter.complete = AsyncMock(return_value=mock_response)

        oracle = LLMJudgeOracle(structured_config)
        oracle.adapter = mock_adapter

        result = await oracle.evaluate(
            "Test response", {"original_prompt": "Test prompt"}
        )

        assert result.score == 0.8
        assert result.metadata["output_format"] == "structured"

    @pytest.mark.asyncio
    async def test_evaluate_no_adapter_error(self, binary_config):
        """Test evaluation without initialized adapter."""
        oracle = LLMJudgeOracle(binary_config)

        with pytest.raises(OracleError, match="LLM adapter not initialized"):
            await oracle.evaluate("test", {})

    @pytest.mark.asyncio
    async def test_evaluate_adapter_error(self, binary_config, mock_adapter):
        """Test evaluation with adapter error."""
        mock_adapter.complete = AsyncMock(side_effect=Exception("API Error"))

        oracle = LLMJudgeOracle(binary_config)
        oracle.adapter = mock_adapter

        with pytest.raises(OracleError, match="Judge evaluation failed"):
            await oracle.evaluate("test", {})

    def test_system_prompt_generation(self, binary_config):
        """Test system prompt generation."""
        oracle = LLMJudgeOracle(binary_config)
        system_prompt = oracle._get_system_prompt()

        assert "expert evaluator" in system_prompt
        assert "JSON format" in system_prompt
        assert '"score": 1' in system_prompt  # Binary format instruction

    def test_format_instructions_binary(self, binary_config):
        """Test format instructions for binary output."""
        oracle = LLMJudgeOracle(binary_config)
        instructions = oracle._get_format_instructions()

        assert '"score": 1' in instructions
        assert "1 for pass, 0 for fail" in instructions

    def test_format_instructions_score(self, score_config):
        """Test format instructions for score output."""
        oracle = LLMJudgeOracle(score_config)
        instructions = oracle._get_format_instructions()

        assert '"score": 0.85' in instructions
        assert "Float between 0.0 and 1.0" in instructions

    def test_format_instructions_structured(self, structured_config):
        """Test format instructions for structured output."""
        oracle = LLMJudgeOracle(structured_config)
        instructions = oracle._get_format_instructions()

        assert '"dimensions"' in instructions
        assert '"details"' in instructions
        assert "Overall float score" in instructions

    def test_judge_prompt_building(self, binary_config):
        """Test judge prompt building."""
        oracle = LLMJudgeOracle(binary_config)

        context = {"original_prompt": "What is the capital of France?"}

        prompt = oracle._build_judge_prompt("Paris is the capital.", context)

        assert "EVALUATION RUBRIC:" in prompt
        assert binary_config.rubric in prompt
        assert "ORIGINAL PROMPT:" in prompt
        assert "What is the capital of France?" in prompt
        assert "RESPONSE TO EVALUATE:" in prompt
        assert "Paris is the capital." in prompt

    def test_few_shot_examples_binary(self, binary_config):
        """Test few-shot examples for binary format."""
        oracle = LLMJudgeOracle(binary_config)
        examples = oracle._get_few_shot_examples()

        assert len(examples) > 0
        assert any('score": 1' in example for example in examples)
        assert any('score": 0' in example for example in examples)

    def test_few_shot_examples_score(self, score_config):
        """Test few-shot examples for score format."""
        oracle = LLMJudgeOracle(score_config)
        examples = oracle._get_few_shot_examples()

        assert len(examples) > 0
        assert any("0.95" in example for example in examples)
        assert any("0.4" in example for example in examples)

    def test_chain_of_thought_decision(self, binary_config, structured_config):
        """Test chain-of-thought usage decision."""
        # Simple binary config should not use CoT
        binary_oracle = LLMJudgeOracle(binary_config)
        assert not binary_oracle._should_use_chain_of_thought()

        # Structured config should use CoT
        structured_oracle = LLMJudgeOracle(structured_config)
        assert structured_oracle._should_use_chain_of_thought()

        # Complex rubric should use CoT
        complex_config = LLMJudgeConfig(
            rubric=(
                "Comprehensive evaluation criteria:\n"
                "1. Accuracy: Factual correctness and reliability\n"
                "2. Clarity: Clear expression and comprehensibility\n"
                "3. Completeness: Adequate coverage of all aspects\n"
                "4. Relevance: Direct relation to the question asked\n"
                "5. Grammar: Proper language usage and structure"
            ),
            judge_model="gpt-4",
            output_format="binary",
        )
        complex_oracle = LLMJudgeOracle(complex_config)
        assert complex_oracle._should_use_chain_of_thought()

    def test_json_extraction_success(self, binary_config):
        """Test successful JSON extraction."""
        oracle = LLMJudgeOracle(binary_config)

        # Direct JSON
        response1 = '{"score": 1, "reasoning": "Good response"}'
        result1 = oracle._extract_json_from_response(response1)
        assert result1 == {"score": 1, "reasoning": "Good response"}

        # JSON in code block
        response2 = (
            'Here is my evaluation:\n```json\n{"score": 0, "reasoning": "Poor"}\n```'
        )
        result2 = oracle._extract_json_from_response(response2)
        assert result2 == {"score": 0, "reasoning": "Poor"}

    def test_json_extraction_failure(self, binary_config):
        """Test JSON extraction failure."""
        oracle = LLMJudgeOracle(binary_config)

        # Malformed JSON
        response = "This is not JSON at all"
        result = oracle._extract_json_from_response(response)
        assert result is None

    def test_fallback_parsing_binary(self, binary_config):
        """Test fallback parsing for binary format."""
        oracle = LLMJudgeOracle(binary_config)

        # Test pass indicators
        pass_response = "This response passes all criteria and meets requirements"
        result = oracle._fallback_parse_response(pass_response)
        assert result["score"] == 1

        # Test fail indicators
        fail_response = "This response fails to meet the standards"
        result = oracle._fallback_parse_response(fail_response)
        assert result["score"] == 0

    def test_fallback_parsing_score(self, score_config):
        """Test fallback parsing for score format."""
        oracle = LLMJudgeOracle(score_config)

        # Extract score from text
        score_response = "The response gets a score of 0.75 out of 1.0"
        result = oracle._fallback_parse_response(score_response)
        assert result["score"] == 0.75

    def test_score_extraction_patterns(self, score_config):
        """Test various score extraction patterns."""
        oracle = LLMJudgeOracle(score_config)

        patterns = [
            ("score: 0.8", 0.8),
            ("0.9/1", 0.9),
            ("0.6 out of 1", 0.6),
            ("rating: 0.7", 0.7),
        ]

        for text, expected in patterns:
            score = oracle._extract_score_fallback(text)
            assert score == expected

    def test_reasoning_extraction(self, binary_config):
        """Test reasoning extraction from responses."""
        oracle = LLMJudgeOracle(binary_config)

        # Test reasoning patterns
        patterns = [
            ("reasoning: This is the explanation", "This is the explanation"),
            ("because: It meets criteria", "It meets criteria"),
            ("explanation: Very clear response", "Very clear response"),
        ]

        for text, expected in patterns:
            reasoning = oracle._extract_reasoning_fallback(text)
            assert expected in reasoning

    def test_create_judge_response_types(
        self, binary_config, score_config, structured_config
    ):
        """Test creating different judge response types."""
        # Binary response
        binary_oracle = LLMJudgeOracle(binary_config)
        binary_data = {"score": 1, "reasoning": "Pass"}
        binary_response = binary_oracle._create_judge_response(binary_data)
        assert isinstance(binary_response, BinaryJudgeResponse)

        # Score response
        score_oracle = LLMJudgeOracle(score_config)
        score_data = {"score": 0.8, "reasoning": "Good"}
        score_response = score_oracle._create_judge_response(score_data)
        assert isinstance(score_response, NumericJudgeResponse)

    def test_create_judge_response_fallback(self, binary_config):
        """Test judge response creation with invalid data."""
        oracle = LLMJudgeOracle(binary_config)

        # Invalid data should fallback gracefully
        invalid_data = {"invalid": "data"}
        response = oracle._create_judge_response(invalid_data)

        assert isinstance(response, BinaryJudgeResponse)
        assert response.score in (0, 1)
        assert response.reasoning == "No reasoning provided"

    @pytest.mark.asyncio
    async def test_context_manager(self, binary_config):
        """Test async context manager functionality."""
        with patch("metareason.oracles.llm_judge.AdapterFactory") as mock_factory:
            mock_adapter = AsyncMock()
            mock_factory.create.return_value = mock_adapter

            oracle = LLMJudgeOracle(binary_config, adapter_config={"test": "config"})

            async with oracle:
                assert oracle.adapter == mock_adapter
                mock_adapter.initialize.assert_called_once()

            mock_adapter.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_and_cleanup(self, binary_config):
        """Test manual initialization and cleanup."""
        with patch("metareason.oracles.llm_judge.AdapterFactory") as mock_factory:
            mock_adapter = AsyncMock()
            mock_factory.create.return_value = mock_adapter

            oracle = LLMJudgeOracle(binary_config, adapter_config={"test": "config"})

            await oracle.initialize()
            assert oracle.adapter == mock_adapter
            mock_adapter.initialize.assert_called_once()

            await oracle.cleanup()
            mock_adapter.cleanup.assert_called_once()

    def test_prompt_building_with_chain_of_thought(self, structured_config):
        """Test prompt building includes chain-of-thought for complex rubrics."""
        oracle = LLMJudgeOracle(structured_config)

        prompt = oracle._build_judge_prompt("test response", {})

        assert "Think through your evaluation step by step:" in prompt
        assert "1. Identify the key aspects" in prompt
        assert "2. Analyze the response" in prompt

    def test_prompt_building_without_chain_of_thought(self, binary_config):
        """Test prompt building without chain-of-thought for simple rubrics."""
        oracle = LLMJudgeOracle(binary_config)

        prompt = oracle._build_judge_prompt("test response", {})

        assert "Think through your evaluation step by step:" not in prompt
