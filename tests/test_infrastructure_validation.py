"""Tests to validate the new test infrastructure.

This test file ensures that our new ConfigBuilder, factories, and fixtures
work correctly before we start the major refactoring effort.
"""

from pathlib import Path

from metareason.config.models import EvaluationConfig

from .factories.evaluation_factory import YamlFileFactory
from .fixtures.config_builders import AxisBuilder, ConfigBuilder, OracleBuilder


class TestConfigBuilder:
    """Test the ConfigBuilder infrastructure."""

    def test_minimal_config_builder(self, config_builder):
        """Test creating a minimal configuration with builder."""
        config = config_builder.minimal().build()

        assert isinstance(config, EvaluationConfig)
        assert config.spec_id == "minimal_test"
        assert len(config.pipeline) == 1
        step = config.pipeline[0]
        assert "{{param}}" in step.template
        assert "param" in step.axes
        assert config.oracles.accuracy is not None
        assert step.adapter == "openai"
        assert step.model == "gpt-3.5-turbo"

    def test_comprehensive_config_builder(self, config_builder):
        """Test creating a comprehensive configuration with builder."""
        config = config_builder.comprehensive().build()

        assert isinstance(config, EvaluationConfig)
        assert config.spec_id == "comprehensive_test"
        assert len(config.pipeline) == 1
        step = config.pipeline[0]
        assert len(step.axes) >= 2
        assert config.oracles.accuracy is not None
        # Check if we have more than just accuracy oracle
        oracle_count = sum(
            1
            for attr in ["accuracy", "explainability", "confidence_calibration"]
            if getattr(config.oracles, attr) is not None
        )
        assert oracle_count >= 1  # At least accuracy should be set
        assert config.n_variants == 1000
        assert config.sampling is not None

    def test_fluent_api_chaining(self, config_builder):
        """Test that the fluent API works correctly."""
        config = (
            config_builder.spec_id("fluent_test")
            .single_step(
                template="Test {{name}} with {{style}}",
                name=["Alice", "Bob"],
                style=["formal", "casual"],
            )
            .with_oracle(
                "explainability",
                lambda o: o.llm_judge(
                    "Rate the response quality from 1-5 based on: 1. Clarity of explanation, "
                    "2. Accuracy of content, 3. Completeness of answer"
                ),
            )
            .with_variants(500)
            .build()
        )

        assert config.spec_id == "fluent_test"
        step = config.pipeline[0]
        assert step.axes["name"].values == ["Alice", "Bob"]
        assert step.axes["style"].values == ["formal", "casual"]
        assert config.oracles.explainability is not None
        assert config.n_variants == 500

    def test_yaml_generation(self, config_builder):
        """Test generating YAML from builder."""
        yaml_content = config_builder.minimal().to_yaml()

        assert "spec_id: minimal_test" in yaml_content
        assert "pipeline:" in yaml_content
        assert "adapter: openai" in yaml_content
        assert "axes:" in yaml_content
        assert "oracles:" in yaml_content


class TestAxisBuilder:
    """Test the AxisBuilder helper class."""

    def test_categorical_axis(self):
        """Test creating categorical axis."""
        axis = AxisBuilder("test").categorical(["A", "B", "C"], [0.5, 0.3, 0.2]).build()

        assert axis["type"] == "categorical"
        assert axis["values"] == ["A", "B", "C"]
        assert axis["weights"] == [0.5, 0.3, 0.2]

    def test_uniform_axis(self):
        """Test creating uniform distribution axis."""
        axis = AxisBuilder("test").uniform(0.0, 1.0).build()

        assert axis["type"] == "uniform"
        assert axis["min"] == 0.0
        assert axis["max"] == 1.0

    def test_truncated_normal_axis(self):
        """Test creating truncated normal axis."""
        axis = AxisBuilder("test").truncated_normal(0.7, 0.1, 0.3, 0.9).build()

        assert axis["type"] == "truncated_normal"
        assert axis["mu"] == 0.7
        assert axis["sigma"] == 0.1
        assert axis["min"] == 0.3
        assert axis["max"] == 0.9


class TestOracleBuilder:
    """Test the OracleBuilder helper class."""

    def test_embedding_similarity_oracle(self):
        """Test creating embedding similarity oracle."""
        oracle = OracleBuilder("test").embedding_similarity("Test answer", 0.85).build()

        assert oracle["type"] == "embedding_similarity"
        assert oracle["canonical_answer"] == "Test answer"
        assert oracle["threshold"] == 0.85

    def test_llm_judge_oracle(self):
        """Test creating LLM judge oracle."""
        oracle = OracleBuilder("test").llm_judge("Rate from 1-5", "gpt-4").build()

        assert oracle["type"] == "llm_judge"
        assert oracle["rubric"] == "Rate from 1-5"
        assert oracle["judge_model"] == "gpt-4"


class TestEvaluationFactory:
    """Test the EvaluationFactory."""

    def test_minimal_factory(self, evaluation_factory):
        """Test creating minimal config with factory."""
        config = evaluation_factory.minimal()

        assert isinstance(config, EvaluationConfig)
        assert len(config.pipeline) == 1
        assert config.pipeline[0].adapter is not None

    def test_single_axis_factory(self, evaluation_factory):
        """Test creating config with single axis."""
        config = evaluation_factory.with_single_axis("topic", ["AI", "ML"])

        step = config.pipeline[0]
        assert "topic" in step.axes
        assert step.axes["topic"].values == ["AI", "ML"]
        assert "{{topic}}" in step.template

    def test_multiple_axes_factory(self, evaluation_factory):
        """Test creating config with multiple axes."""
        axes_config = {"name": ["Alice", "Bob"], "style": ["formal", "casual"]}
        config = evaluation_factory.with_multiple_axes(axes_config)

        step = config.pipeline[0]
        assert len(step.axes) == 2
        assert step.axes["name"].values == ["Alice", "Bob"]
        assert step.axes["style"].values == ["formal", "casual"]

    def test_invalid_config_factory(self, evaluation_factory):
        """Test creating invalid configurations."""
        missing_oracles = evaluation_factory.invalid_missing_field("oracles")
        assert "oracles" not in missing_oracles

        empty_spec_id = evaluation_factory.invalid_empty_field("spec_id")
        assert empty_spec_id["spec_id"] == ""


class TestYamlFileFactory:
    """Test the YamlFileFactory for creating temporary files."""

    def test_create_temp_file(self, evaluation_factory):
        """Test creating temporary YAML file."""
        config = evaluation_factory.minimal()
        temp_path = YamlFileFactory.create_temp_file(config)

        try:
            assert temp_path.exists()
            assert temp_path.suffix == ".yaml"

            content = temp_path.read_text()
            assert "spec_id:" in content
            assert "pipeline:" in content
            assert "axes:" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_create_minimal_file(self):
        """Test creating minimal config file."""
        temp_path = YamlFileFactory.create_minimal_file(spec_id="custom_test")

        try:
            assert temp_path.exists()
            content = temp_path.read_text()
            assert "spec_id: custom_test" in content
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestFixtures:
    """Test that pytest fixtures work correctly."""

    def test_config_builder_fixture(self, config_builder):
        """Test config_builder fixture."""
        assert isinstance(config_builder, ConfigBuilder)
        config = config_builder.minimal().build()
        assert isinstance(config, EvaluationConfig)

    def test_minimal_config_fixture(self, minimal_config):
        """Test minimal_config fixture."""
        assert isinstance(minimal_config, EvaluationConfig)
        assert minimal_config.spec_id == "minimal_test"

    def test_temp_config_file_fixture(self, temp_config_file):
        """Test temp_config_file fixture."""
        temp_path = temp_config_file()

        assert isinstance(temp_path, Path)
        assert temp_path.exists()
        assert temp_path.suffix == ".yaml"

        content = temp_path.read_text()
        assert "spec_id:" in content
        assert "pipeline:" in content

    def test_yaml_template_fixture(self, yaml_template):
        """Test yaml_template fixture."""
        rendered = yaml_template.render(yaml_template.MINIMAL, spec_id="fixture_test")
        assert "spec_id: fixture_test" in rendered

    def test_common_axes_fixture(self, common_axes):
        """Test common_axes fixture."""
        assert "name" in common_axes
        assert isinstance(common_axes["name"], list)

    def test_common_oracles_fixture(self, common_oracles):
        """Test common_oracles fixture."""
        assert "accuracy" in common_oracles
        assert common_oracles["accuracy"]["type"] == "embedding_similarity"
        assert "explainability" in common_oracles
        assert common_oracles["explainability"]["type"] == "llm_judge"
