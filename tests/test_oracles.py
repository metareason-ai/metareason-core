import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metareason.adapters.adapter_base import AdapterResponse
from metareason.config.models import AdapterConfig, OracleConfig
from metareason.oracles.llm_judge import LLMJudge
from metareason.oracles.oracle_base import (
    EvaluationContext,
    EvaluationResult,
    OracleBase,
    OracleException,
)


class TestOracleBaseContract:
    def test_evaluate_is_coroutine_function(self):
        """OracleBase.evaluate must be declared as an async method."""
        assert inspect.iscoroutinefunction(OracleBase.evaluate)


def make_oracle_config(**overrides):
    defaults = dict(
        type="llm_judge",
        model="test-model",
        adapter=AdapterConfig(name="ollama"),
        rubric="Score 1-5 on quality",
    )
    defaults.update(overrides)
    return OracleConfig(**defaults)


@pytest.fixture
def eval_context():
    return EvaluationContext(
        prompt="What is 2+2?",
        response="The answer is 4.",
    )


class TestLLMJudgeInit:
    @patch("metareason.oracles.llm_judge.get_adapter")
    def test_init_requires_rubric(self, mock_get_adapter):
        with pytest.raises(ValueError, match="Rubric is required"):
            LLMJudge(make_oracle_config(rubric=None))

    @patch("metareason.oracles.llm_judge.get_adapter")
    def test_init_creates_adapter(self, mock_get_adapter):
        mock_get_adapter.return_value = MagicMock()
        config = make_oracle_config()
        judge = LLMJudge(config)

        mock_get_adapter.assert_called_once_with("ollama")
        assert judge.adapter is mock_get_adapter.return_value
        assert "Score 1-5 on quality" in judge.sys_prompt


class TestLLMJudgeEvaluate:
    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_success(self, mock_get_adapter, eval_context):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text=json.dumps({"score": 4, "explanation": "good"})
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        result = await judge.evaluate(eval_context)

        assert isinstance(result, EvaluationResult)
        assert result.score == 4.0
        assert result.explanation == "good"

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_with_markdown_code_block(
        self, mock_get_adapter, eval_context
    ):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text='```json\n{"score": 3, "explanation": "ok"}\n```'
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        result = await judge.evaluate(eval_context)

        assert result.score == 3.0
        assert result.explanation == "ok"

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_with_text_before_code_block(
        self, mock_get_adapter, eval_context
    ):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text='Here is my evaluation:\n```json\n{"score": 5, "explanation": "excellent"}\n```'
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        result = await judge.evaluate(eval_context)

        assert result.score == 5.0
        assert result.explanation == "excellent"

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_with_code_block_no_lang_tag(
        self, mock_get_adapter, eval_context
    ):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text='Some preamble\n```\n{"score": 2, "explanation": "poor"}\n```'
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        result = await judge.evaluate(eval_context)

        assert result.score == 2.0
        assert result.explanation == "poor"

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_missing_score_field(self, mock_get_adapter, eval_context):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text=json.dumps({"explanation": "no score"})
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        with pytest.raises(OracleException, match="missing 'score' field"):
            await judge.evaluate(eval_context)

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_invalid_json(self, mock_get_adapter, eval_context):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text="this is not json"
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        with pytest.raises(OracleException, match="Invalid JSON"):
            await judge.evaluate(eval_context)

    @patch("metareason.oracles.llm_judge.get_adapter")
    @pytest.mark.asyncio
    async def test_evaluate_no_explanation_uses_default(
        self, mock_get_adapter, eval_context
    ):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text=json.dumps({"score": 5})
        )
        mock_get_adapter.return_value = mock_adapter

        judge = LLMJudge(make_oracle_config())
        result = await judge.evaluate(eval_context)

        assert result.score == 5.0
        assert result.explanation == "No explanation provided"
