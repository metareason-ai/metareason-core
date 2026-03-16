import pytest

from metareason.config.models import OracleConfig
from metareason.oracles.oracle_base import (
    EvaluationContext,
    EvaluationResult,
    OracleException,
)
from metareason.oracles.regex_oracle import RegexOracle


def make_regex_config(**overrides):
    defaults = dict(
        type="regex",
        patterns=["\\d+"],
    )
    defaults.update(overrides)
    return OracleConfig(**defaults)


@pytest.fixture
def eval_context():
    return EvaluationContext(
        prompt="Give me a number",
        response="The answer is 42.",
    )


class TestRegexOracleInit:
    def test_init_requires_patterns(self):
        with pytest.raises(ValueError, match="At least one pattern"):
            RegexOracle(make_regex_config(patterns=None))

    def test_init_empty_patterns(self):
        with pytest.raises(ValueError, match="At least one pattern"):
            RegexOracle(make_regex_config(patterns=[]))

    def test_init_invalid_regex(self):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            RegexOracle(make_regex_config(patterns=["[invalid"]))

    def test_init_compiles_patterns(self):
        oracle = RegexOracle(make_regex_config(patterns=["\\d+", "[A-Z]"]))
        assert len(oracle.compiled_patterns) == 2

    def test_init_multiple_patterns_with_one_invalid(self):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            RegexOracle(make_regex_config(patterns=["\\d+", "(unclosed"]))


class TestRegexOracleEvaluate:
    @pytest.mark.asyncio
    async def test_single_pattern_match(self, eval_context):
        oracle = RegexOracle(make_regex_config(patterns=["\\d+"]))
        result = await oracle.evaluate(eval_context)

        assert isinstance(result, EvaluationResult)
        assert result.score == 5.0
        assert "Matched 1/1" in result.explanation

    @pytest.mark.asyncio
    async def test_single_pattern_no_match(self, eval_context):
        oracle = RegexOracle(make_regex_config(patterns=["xyz_not_present"]))
        result = await oracle.evaluate(eval_context)

        assert result.score == 1.0
        assert "Failed 1/1" in result.explanation

    @pytest.mark.asyncio
    async def test_all_patterns_match(self, eval_context):
        oracle = RegexOracle(
            make_regex_config(patterns=["\\d+", "[A-Z]", "answer"])
        )
        result = await oracle.evaluate(eval_context)

        assert result.score == 5.0
        assert "Matched 3/3" in result.explanation

    @pytest.mark.asyncio
    async def test_no_patterns_match(self, eval_context):
        oracle = RegexOracle(
            make_regex_config(patterns=["xyz", "abc123nothere"])
        )
        result = await oracle.evaluate(eval_context)

        assert result.score == 1.0
        assert "Failed 2/2" in result.explanation

    @pytest.mark.asyncio
    async def test_partial_match_scoring(self, eval_context):
        # 2 of 4 patterns match -> ratio 0.5 -> score 1.0 + 4.0*0.5 = 3.0
        oracle = RegexOracle(
            make_regex_config(patterns=["\\d+", "answer", "xyz", "nothere"])
        )
        result = await oracle.evaluate(eval_context)

        assert result.score == 3.0
        assert "Matched 2/4" in result.explanation
        assert "Failed 2/4" in result.explanation

    @pytest.mark.asyncio
    async def test_one_of_three_match(self, eval_context):
        # 1 of 3 -> ratio 1/3 -> score 1.0 + 4.0*(1/3) ≈ 2.333
        oracle = RegexOracle(
            make_regex_config(patterns=["\\d+", "xyz", "nothere"])
        )
        result = await oracle.evaluate(eval_context)

        assert abs(result.score - (1.0 + 4.0 / 3.0)) < 0.01
        assert "Matched 1/3" in result.explanation

    @pytest.mark.asyncio
    async def test_case_sensitive_by_default(self):
        context = EvaluationContext(
            prompt="test",
            response="hello world",
        )
        oracle = RegexOracle(make_regex_config(patterns=["Hello"]))
        result = await oracle.evaluate(context)

        assert result.score == 1.0  # No match - case sensitive

    @pytest.mark.asyncio
    async def test_case_insensitive_with_flag(self):
        context = EvaluationContext(
            prompt="test",
            response="hello world",
        )
        # Use inline regex flag for case-insensitive matching
        oracle = RegexOracle(make_regex_config(patterns=["(?i)Hello"]))
        result = await oracle.evaluate(context)

        assert result.score == 5.0  # Matches with inline flag

    @pytest.mark.asyncio
    async def test_multiline_response(self):
        context = EvaluationContext(
            prompt="test",
            response="Line 1\nLine 2\nLine 3",
        )
        oracle = RegexOracle(make_regex_config(patterns=["Line \\d"]))
        result = await oracle.evaluate(context)

        assert result.score == 5.0

    @pytest.mark.asyncio
    async def test_empty_response(self):
        context = EvaluationContext(
            prompt="test",
            response="",
        )
        oracle = RegexOracle(make_regex_config(patterns=["\\d+"]))
        result = await oracle.evaluate(context)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_explanation_lists_patterns(self, eval_context):
        oracle = RegexOracle(
            make_regex_config(patterns=["\\d+", "xyz"])
        )
        result = await oracle.evaluate(eval_context)

        assert "\\d+" in result.explanation
        assert "xyz" in result.explanation
