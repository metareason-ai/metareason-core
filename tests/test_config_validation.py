"""Comprehensive tests for configuration validation system."""

from datetime import date

import pytest
from pydantic import ValidationError

from metareason.config import (
    BetaDistributionConfig,
    CategoricalAxis,
    ContinuousAxis,
    CustomDistributionConfig,
    CustomOracle,
    EmbeddingSimilarityOracle,
    LLMJudgeOracle,
    Metadata,
    OracleConfig,
    SamplingConfig,
    StatisticalCalibrationOracle,
    TruncatedNormalConfig,
    UniformDistributionConfig,
)
from tests.fixtures.config_builders import ConfigBuilder


class TestEvaluationConfig:
    """Test main EvaluationConfig validation."""

    def test_valid_evaluation_config(self) -> None:
        """Test that a valid configuration passes validation."""
        config = (
            ConfigBuilder()
            .spec_id("test_prompt")
            .add_pipeline_step(
                template="Hello {{name}}, please respond to: {{question}}",
                adapter="openai",
                model="gpt-3.5-turbo",
                axes={
                    "name": {
                        "type": "categorical",
                        "values": ["Alice", "Bob", "Charlie"],
                    },
                    "question": {
                        "type": "categorical",
                        "values": ["What is 2+2?", "Explain quantum physics"],
                    },
                },
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "The answer should be comprehensive and accurate", threshold=0.85
                ),
            )
            .build()
        )

        assert config.spec_id == "test_prompt"
        assert len(config.pipeline) == 1
        assert len(config.pipeline[0].axes) == 2
        assert config.n_variants == 1000  # default value

    def test_empty_spec_id_fails(self) -> None:
        """Test that empty spec_id fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("")  # Empty spec_id
                .single_step(template="Hello world", var=["a"])
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        assert "Spec ID cannot be empty" in str(exc_info.value)

    def test_invalid_spec_id_format_fails(self) -> None:
        """Test that invalid spec_id format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("Invalid-Format!")  # Invalid format
                .single_step(template="Hello world", var=["a"])
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "lowercase letters, numbers, and underscores only" in error_msg
        assert "iso_42001_compliance_check" in error_msg

    def test_long_spec_id_fails(self) -> None:
        """Test that overly long spec_id fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("a" * 101)  # 101 characters
                .single_step(template="Hello world", var=["a"])
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        assert "too long" in str(exc_info.value)

    def test_empty_template_fails(self) -> None:
        """Test that empty template fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="",  # Empty template
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={"var": {"type": "categorical", "values": ["a"]}},
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "Template cannot be empty" in error_msg
        assert "Jinja2 template" in error_msg

    def test_short_template_fails(self) -> None:
        """Test that overly short template fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hi",  # Too short
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={"var": {"type": "categorical", "values": ["a"]}},
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        assert "too short" in str(exc_info.value)

    def test_empty_pipeline_fails(self) -> None:
        """Test that empty pipeline fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .with_params(pipeline=[])  # Empty pipeline
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "Pipeline cannot be empty" in error_msg
        assert "at least one pipeline step" in error_msg

    def test_template_with_no_axes_fails(self) -> None:
        """Test that template with variables but no axes fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hello {{missing_var}}, this is a longer template",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={},  # No axes defined
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "not defined in axes" in error_msg
        assert "missing_var" in error_msg

    def test_low_n_variants_fails(self) -> None:
        """Test that too few variants fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .single_step(
                    template="Hello {{name}}, this is a longer template", name=["a"]
                )
                .with_variants(50)  # Too low
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "greater than or equal to 100" in error_msg

    def test_high_n_variants_fails(self) -> None:
        """Test that too many variants fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .single_step(
                    template="Hello {{name}}, this is a longer template", name=["a"]
                )
                .with_variants(15000)  # Too high
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "less than or equal to 10000" in error_msg


class TestTemplateVariableValidation:
    """Test template variable validation."""

    def test_missing_template_variables_fails(self) -> None:
        """Test that missing template variables fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hello {{name}} and {{age}}, this is a longer template",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={
                        "name": {"type": "categorical", "values": ["Alice"]}
                    },  # missing 'age'
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "not defined in axes" in error_msg
        assert "age" in error_msg

    def test_unused_axes_variables_allowed(self) -> None:
        """Test that unused axes variables are allowed (just warnings)."""
        # Should not raise an error
        config = (
            ConfigBuilder()
            .spec_id("test_prompt")
            .add_pipeline_step(
                template="Hello {{name}}, this is a longer template",
                adapter="openai",
                model="gpt-3.5-turbo",
                axes={
                    "name": {"type": "categorical", "values": ["Alice"]},
                    "unused_var": {"type": "categorical", "values": ["value"]},
                },
            )
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity("test answer", threshold=0.85),
            )
            .build()
        )

        assert config.spec_id == "test_prompt"


class TestCategoricalAxis:
    """Test CategoricalAxis validation."""

    def test_valid_categorical_axis(self) -> None:
        """Test valid categorical axis creation."""
        axis = CategoricalAxis(values=["A", "B", "C"])
        assert axis.type == "categorical"
        assert axis.values == ["A", "B", "C"]
        assert axis.weights is None

    def test_valid_categorical_axis_with_weights(self) -> None:
        """Test valid categorical axis with weights."""
        axis = CategoricalAxis(values=["A", "B", "C"], weights=[0.5, 0.3, 0.2])
        assert axis.weights == [0.5, 0.3, 0.2]

    def test_empty_values_fails(self) -> None:
        """Test that empty values list fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalAxis(values=[])

        error_msg = str(exc_info.value)
        # Check for Pydantic built-in validation message or custom validation
        assert "at least one value" in error_msg or "at least 1 item" in error_msg

    def test_duplicate_values_fails(self) -> None:
        """Test that duplicate values fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalAxis(values=["A", "B", "A"])

        error_msg = str(exc_info.value)
        assert "must be unique" in error_msg
        assert "duplicates: ['A']" in error_msg

    def test_negative_weights_fails(self) -> None:
        """Test that negative weights fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalAxis(values=["A", "B"], weights=[0.5, -0.3])

        error_msg = str(exc_info.value)
        assert "non-negative" in error_msg
        assert "[-0.3]" in error_msg

    def test_mismatched_weights_length_fails(self) -> None:
        """Test that mismatched weights length fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalAxis(values=["A", "B", "C"], weights=[0.5, 0.5])

        error_msg = str(exc_info.value)
        assert "Number of weights (2) must match number of values (3)" in error_msg

    def test_weights_not_sum_to_one_fails(self) -> None:
        """Test that weights not summing to 1.0 fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalAxis(values=["A", "B"], weights=[0.4, 0.4])

        error_msg = str(exc_info.value)
        assert "must sum to 1.0" in error_msg
        assert "got 0.800000" in error_msg


class TestContinuousAxis:
    """Test ContinuousAxis validation."""

    def test_valid_truncated_normal(self) -> None:
        """Test valid truncated normal axis."""
        axis = ContinuousAxis(
            type="truncated_normal", mu=0.5, sigma=0.1, min=0.0, max=1.0
        )
        assert axis.type == "truncated_normal"
        assert axis.mu == 0.5

    def test_truncated_normal_missing_parameters_fails(self) -> None:
        """Test that missing required parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="truncated_normal", mu=0.5)

        error_msg = str(exc_info.value)
        assert "requires: sigma, min, max" in error_msg

    def test_truncated_normal_negative_sigma_fails(self) -> None:
        """Test that negative sigma fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(
                type="truncated_normal", mu=0.5, sigma=-0.1, min=0.0, max=1.0
            )

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg or "greater than 0" in error_msg

    def test_truncated_normal_invalid_bounds_fails(self) -> None:
        """Test that invalid bounds fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="truncated_normal", mu=0.5, sigma=0.1, min=1.0, max=0.0)

        error_msg = str(exc_info.value)
        assert "Minimum (1.0) must be less than maximum (0.0)" in error_msg

    def test_valid_beta_distribution(self) -> None:
        """Test valid beta distribution axis."""
        axis = ContinuousAxis(type="beta", alpha=2.0, beta=5.0)
        assert axis.type == "beta"
        assert axis.alpha == 2.0

    def test_beta_missing_parameters_fails(self) -> None:
        """Test that missing beta parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="beta", alpha=2.0)

        error_msg = str(exc_info.value)
        assert "requires: beta" in error_msg

    def test_beta_negative_parameters_fail(self) -> None:
        """Test that negative beta parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="beta", alpha=-1.0, beta=2.0)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg or "greater than 0" in error_msg

    def test_valid_uniform_distribution(self) -> None:
        """Test valid uniform distribution axis."""
        axis = ContinuousAxis(type="uniform", min=0.0, max=1.0)
        assert axis.type == "uniform"

    def test_uniform_missing_parameters_fails(self) -> None:
        """Test that missing uniform parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="uniform", min=0.0)

        error_msg = str(exc_info.value)
        assert "requires: max" in error_msg

    def test_uniform_invalid_bounds_fails(self) -> None:
        """Test that invalid uniform bounds fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="uniform", min=1.0, max=0.0)

        error_msg = str(exc_info.value)
        assert "must be less than maximum" in error_msg

    def test_valid_custom_distribution(self) -> None:
        """Test valid custom distribution axis."""
        axis = ContinuousAxis(
            type="custom",
            module="metareason.distributions.custom",
            class_name="CustomDistribution",
            config={"param": "value"},
        )
        assert axis.type == "custom"

    def test_custom_missing_parameters_fails(self) -> None:
        """Test that missing custom parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContinuousAxis(type="custom", module="test")

        error_msg = str(exc_info.value)
        assert "requires: class_name" in error_msg


class TestDistributionModels:
    """Test distribution configuration models."""

    def test_valid_truncated_normal_config(self) -> None:
        """Test valid truncated normal configuration."""
        config = TruncatedNormalConfig(mu=0.5, sigma=0.1, min=0.0, max=1.0)
        assert config.type == "truncated_normal"

    def test_truncated_normal_negative_sigma_fails(self) -> None:
        """Test that negative sigma fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            TruncatedNormalConfig(mu=0.5, sigma=-0.1, min=0.0, max=1.0)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg or "greater than 0" in error_msg

    def test_truncated_normal_invalid_bounds_fails(self) -> None:
        """Test that invalid bounds fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            TruncatedNormalConfig(mu=0.5, sigma=0.1, min=1.0, max=0.0)

        error_msg = str(exc_info.value)
        assert "must be less than maximum" in error_msg

    def test_valid_beta_distribution_config(self) -> None:
        """Test valid beta distribution configuration."""
        config = BetaDistributionConfig(alpha=2.0, beta=5.0)
        assert config.type == "beta"

    def test_beta_negative_parameters_fail(self) -> None:
        """Test that negative beta parameters fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            BetaDistributionConfig(alpha=-1.0, beta=2.0)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg or "greater than 0" in error_msg

    def test_valid_uniform_distribution_config(self) -> None:
        """Test valid uniform distribution configuration."""
        config = UniformDistributionConfig(min=0.0, max=1.0)
        assert config.type == "uniform"

    def test_uniform_invalid_bounds_fails(self) -> None:
        """Test that invalid uniform bounds fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            UniformDistributionConfig(min=1.0, max=0.0)

        error_msg = str(exc_info.value)
        assert "must be less than maximum" in error_msg

    def test_valid_custom_distribution_config(self) -> None:
        """Test valid custom distribution configuration."""
        config = CustomDistributionConfig(
            module="metareason.distributions.custom", class_name="CustomDistribution"
        )
        assert config.type == "custom"

    def test_custom_invalid_module_fails(self) -> None:
        """Test that invalid module format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CustomDistributionConfig(module="invalid", class_name="Test")

        error_msg = str(exc_info.value)
        assert "appears invalid" in error_msg
        assert "metareason.distributions.custom" in error_msg

    def test_custom_invalid_class_name_fails(self) -> None:
        """Test that invalid class name fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CustomDistributionConfig(
                module="metareason.distributions.custom", class_name="lowercase"
            )

        error_msg = str(exc_info.value)
        assert "naming conventions" in error_msg
        assert "PascalCase" in error_msg


class TestOracleModels:
    """Test oracle configuration models."""

    def test_valid_embedding_similarity_oracle(self) -> None:
        """Test valid embedding similarity oracle."""
        oracle = EmbeddingSimilarityOracle(
            canonical_answer="This is a comprehensive test answer for validation",
            threshold=0.85,
        )
        assert oracle.type == "embedding_similarity"

    def test_embedding_empty_canonical_answer_fails(self) -> None:
        """Test that empty canonical answer fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSimilarityOracle(canonical_answer="", threshold=0.85)

        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg
        assert "comprehensive expected answer" in error_msg

    def test_embedding_short_canonical_answer_fails(self) -> None:
        """Test that short canonical answer fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSimilarityOracle(canonical_answer="Short", threshold=0.85)

        error_msg = str(exc_info.value)
        assert "too short" in error_msg
        assert "more detailed" in error_msg

    def test_embedding_invalid_threshold_fails(self) -> None:
        """Test that invalid threshold fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSimilarityOracle(
                canonical_answer="This is a comprehensive test answer for validation",
                threshold=1.5,
            )

        error_msg = str(exc_info.value)
        assert (
            "between 0.0 and 1.0" in error_msg or "less than or equal to 1" in error_msg
        )

    def test_valid_llm_judge_oracle(self) -> None:
        """Test valid LLM judge oracle."""
        oracle = LLMJudgeOracle(
            rubric="1. Check accuracy\n2. Verify completeness\n3. Assess clarity"
        )
        assert oracle.type == "llm_judge"

    def test_llm_judge_empty_rubric_fails(self) -> None:
        """Test that empty rubric fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMJudgeOracle(rubric="")

        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg
        assert "clear evaluation criteria" in error_msg

    def test_llm_judge_brief_rubric_fails(self) -> None:
        """Test that overly brief rubric fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMJudgeOracle(rubric="Check quality")

        error_msg = str(exc_info.value)
        assert "too brief" in error_msg
        assert "numbered (1., 2., etc.)" in error_msg

    def test_valid_statistical_calibration_oracle(self) -> None:
        """Test valid statistical calibration oracle."""
        oracle = StatisticalCalibrationOracle(expected_confidence=0.85)
        assert oracle.type == "statistical_calibration"

    def test_statistical_invalid_confidence_fails(self) -> None:
        """Test that invalid confidence fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            StatisticalCalibrationOracle(expected_confidence=1.5)

        error_msg = str(exc_info.value)
        assert (
            "between 0.0 and 1.0" in error_msg or "less than or equal to 1" in error_msg
        )

    def test_statistical_invalid_tolerance_fails(self) -> None:
        """Test that invalid tolerance fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            StatisticalCalibrationOracle(expected_confidence=0.85, tolerance=0.0)

        error_msg = str(exc_info.value)
        assert "must be positive" in error_msg or "greater than 0" in error_msg

    def test_statistical_large_tolerance_fails(self) -> None:
        """Test that overly large tolerance fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            StatisticalCalibrationOracle(expected_confidence=0.85, tolerance=0.6)

        error_msg = str(exc_info.value)
        assert "too large" in error_msg or "less than or equal to 0.5" in error_msg

    def test_valid_custom_oracle(self) -> None:
        """Test valid custom oracle."""
        oracle = CustomOracle(
            module="metareason.oracles.custom", class_name="CustomOracle"
        )
        assert oracle.type == "custom"

    def test_custom_oracle_invalid_module_fails(self) -> None:
        """Test that invalid module fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CustomOracle(module="invalid", class_name="CustomOracle")

        error_msg = str(exc_info.value)
        assert "appears invalid" in error_msg
        assert "metareason.oracles.custom" in error_msg

    def test_custom_oracle_invalid_class_name_fails(self) -> None:
        """Test that invalid class name fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            CustomOracle(module="metareason.oracles.custom", class_name="lowercase")

        error_msg = str(exc_info.value)
        assert "naming conventions" in error_msg
        assert "PascalCase" in error_msg


class TestOracleConfig:
    """Test oracle configuration container."""

    def test_valid_oracle_config(self) -> None:
        """Test valid oracle configuration."""
        oracle_config = OracleConfig(
            accuracy=EmbeddingSimilarityOracle(
                canonical_answer="This is a comprehensive test answer", threshold=0.85
            )
        )
        assert oracle_config.accuracy is not None

    def test_invalid_custom_oracle_name_fails(self) -> None:
        """Test that invalid custom oracle names fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            OracleConfig(
                custom_oracles={
                    "invalid-name!": CustomOracle(
                        module="metareason.oracles.custom", class_name="CustomOracle"
                    )
                }
            )

        error_msg = str(exc_info.value)
        assert "is invalid" in error_msg
        assert "regulatory_compliance" in error_msg


class TestSamplingConfig:
    """Test sampling configuration validation."""

    def test_valid_sampling_config(self) -> None:
        """Test valid sampling configuration."""
        config = SamplingConfig()
        assert config.method == "latin_hypercube"
        assert config.random_seed == 42

    def test_negative_random_seed_fails(self) -> None:
        """Test that negative random seed fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            SamplingConfig(random_seed=-1)

        error_msg = str(exc_info.value)
        assert "non-negative" in error_msg

    def test_empty_stratified_by_fails(self) -> None:
        """Test that empty stratified_by list fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            SamplingConfig(stratified_by=[])

        error_msg = str(exc_info.value)
        assert "at least one axis name" in error_msg

    def test_duplicate_stratified_by_fails(self) -> None:
        """Test that duplicate stratified_by names fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            SamplingConfig(stratified_by=["axis1", "axis2", "axis1"])

        error_msg = str(exc_info.value)
        assert "must be unique" in error_msg
        assert "['axis1']" in error_msg


class TestCrossFieldValidation:
    """Test cross-field validation rules."""

    def test_stratified_sampling_validation_passes(self) -> None:
        """Test that valid stratified sampling passes validation."""
        config = (
            ConfigBuilder()
            .spec_id("test_prompt")
            .add_pipeline_step(
                template="Hello {{category}}, this is a longer template",
                adapter="openai",
                model="gpt-3.5-turbo",
                axes={"category": {"type": "categorical", "values": ["A", "B", "C"]}},
            )
            .with_sampling(method="latin_hypercube", stratified_by=["category"])
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "This is a comprehensive test answer", threshold=0.85
                ),
            )
            .build()
        )

        assert config.sampling.stratified_by == ["category"]

    def test_stratified_sampling_nonexistent_axis_fails(self) -> None:
        """Test that stratified sampling with nonexistent axis fails."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hello {{category}}, this is a longer template",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={
                        "category": {"type": "categorical", "values": ["A", "B", "C"]}
                    },
                )
                .with_sampling(
                    method="latin_hypercube", stratified_by=["nonexistent_axis"]
                )
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity(
                        "This is a comprehensive test answer", threshold=0.85
                    ),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "not found" in error_msg

    def test_stratified_sampling_continuous_axis_fails(self) -> None:
        """Test that stratified sampling with continuous axis fails."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hello world, this is a longer template with {{temperature}}",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={"temperature": {"type": "uniform", "min": 0.0, "max": 1.0}},
                )
                .with_sampling(method="latin_hypercube", stratified_by=["temperature"])
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity(
                        "This is a comprehensive test answer", threshold=0.85
                    ),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        assert "must be categorical" in error_msg

    def test_no_oracles_configured_fails(self) -> None:
        """Test that configuration with no oracles fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .single_step(
                    template="Hello {{name}}, this is a longer template", name=["Alice"]
                )
                .with_params(oracles={})  # No oracles
                .build()
            )

        error_msg = str(exc_info.value)
        assert "At least one oracle must be configured" in error_msg

    def test_statistical_requirements_validation_passes(self) -> None:
        """Test that valid statistical requirements pass validation."""
        config = (
            ConfigBuilder()
            .spec_id("test_prompt")
            .add_pipeline_step(
                template="Hello {{category}}, this is a longer template",
                adapter="openai",
                model="gpt-3.5-turbo",
                axes={
                    "category": {
                        "type": "categorical",
                        "values": ["A", "B"],
                    }  # 2 combinations
                },
            )
            .with_variants(100)  # 100 > 2*10 minimum requirement
            .with_oracle(
                "accuracy",
                lambda o: o.embedding_similarity(
                    "This is a comprehensive test answer", threshold=0.85
                ),
            )
            .build()
        )

        assert config.n_variants == 100

    def test_statistical_requirements_validation_fails(self) -> None:
        """Test that insufficient statistical requirements fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("test_prompt")
                .add_pipeline_step(
                    template="Hello {{category}} {{type}}, this is a longer template",
                    adapter="openai",
                    model="gpt-3.5-turbo",
                    axes={
                        "category": {"type": "categorical", "values": ["A", "B", "C"]},
                        "type": {
                            "type": "categorical",
                            "values": ["X", "Y"],
                        },  # 3*2 = 6 combinations
                    },
                )
                .with_variants(50)  # This should fail the ge=100 constraint first
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity(
                        "This is a comprehensive test answer", threshold=0.85
                    ),
                )
                .build()
            )

        error_msg = str(exc_info.value)
        # This will fail on the ge=100 constraint first
        assert "greater than or equal to 100" in error_msg


class TestMetadataValidation:
    """Test metadata validation."""

    def test_valid_metadata(self) -> None:
        """Test valid metadata configuration."""
        metadata = Metadata(
            version="1.2.3", created_by="user@example.com", created_date=date.today()
        )
        assert metadata.version == "1.2.3"

    def test_invalid_version_format_fails(self) -> None:
        """Test that invalid version format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(version="1.0")

        error_msg = str(exc_info.value)
        assert "semantic versioning" in error_msg
        assert "major.minor.patch" in error_msg

    def test_invalid_email_format_fails(self) -> None:
        """Test that invalid email format fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            Metadata(created_by="invalid@email")

        error_msg = str(exc_info.value)
        assert "invalid email format" in error_msg


class TestErrorMessageQuality:
    """Test the quality and helpfulness of error messages."""

    def test_error_messages_contain_suggestions(self) -> None:
        """Test that error messages contain helpful suggestions."""
        test_cases = [
            # (builder_func, expected_keywords)
            (
                lambda: (
                    ConfigBuilder()
                    .spec_id("Invalid-Name")
                    .single_step(template="Test template", param=["a"])
                    .with_oracle(
                        "accuracy",
                        lambda o: o.embedding_similarity("test answer", threshold=0.85),
                    )
                ),
                ["lowercase letters"],  # Custom validation
            ),
            (
                lambda: (ConfigBuilder().minimal().with_variants(50)),
                ["greater than or equal to 100"],  # Built-in Pydantic validation
            ),
        ]

        for builder_func, expected_keywords in test_cases:
            with pytest.raises(ValidationError) as exc_info:
                builder_func().build()

            error_msg = str(exc_info.value)
            for keyword in expected_keywords:
                assert (
                    keyword in error_msg
                ), f"Missing '{keyword}' in error: {error_msg}"

    def test_field_specific_error_reporting(self) -> None:
        """Test that validation errors are reported for specific fields."""
        with pytest.raises(ValidationError) as exc_info:
            (
                ConfigBuilder()
                .spec_id("")  # Invalid empty spec_id
                .single_step(template="Test template", param=["a"])
                .with_oracle(
                    "accuracy",
                    lambda o: o.embedding_similarity("test answer", threshold=0.85),
                )
                .build()
            )

        # Check that error is specifically attributed to spec_id field
        error_details = exc_info.value.errors()
        assert len(error_details) >= 1
        assert any("spec_id" in str(error.get("loc", [])) for error in error_details)
