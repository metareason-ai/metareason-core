"""Tests for the new pipeline-based configuration format."""

from metareason.config import load_yaml_config
from metareason.config.models import EvaluationConfig


def test_single_step_pipeline(config_builder):
    """Test loading a single-step pipeline configuration."""
    config = (
        config_builder.spec_id("single_step_test")
        .add_pipeline_step(
            template="Analyze {{topic}} with {{style}}",
            adapter="openai",
            model="gpt-3.5-turbo",
            temperature=0.7,
            axes={
                "topic": {"type": "categorical", "values": ["AI", "ML", "DL"]},
                "style": {"type": "categorical", "values": ["formal", "casual"]},
            },
        )
        .with_variants(1000)
        .with_oracle(
            "accuracy",
            lambda o: o.embedding_similarity(
                "Expected comprehensive analysis", threshold=0.85
            ),
        )
        .build()
    )

    assert config.spec_id == "single_step_test"
    assert len(config.pipeline) == 1
    assert config.pipeline[0].template == "Analyze {{topic}} with {{style}}"
    assert config.pipeline[0].adapter == "openai"
    assert config.pipeline[0].model == "gpt-3.5-turbo"
    assert config.pipeline[0].temperature == 0.7
    assert "topic" in config.pipeline[0].axes
    assert "style" in config.pipeline[0].axes
    assert config.n_variants == 1000
    assert config.oracles.accuracy.threshold == 0.85


def test_multi_step_pipeline(config_builder):
    """Test loading a multi-step pipeline configuration."""
    config = (
        config_builder.spec_id("multi_step_test")
        .add_pipeline_step(
            template="Summarize {{document}}",
            adapter="openai",
            model="gpt-3.5-turbo",
            axes={"document": {"type": "categorical", "values": ["doc1", "doc2"]}},
        )
        .add_pipeline_step(
            template="Analyze: {{stage_1_output}} using {{method}}",
            adapter="anthropic",
            model="claude-3-sonnet",
            axes={
                "method": {"type": "categorical", "values": ["critical", "supportive"]}
            },
        )
        .with_oracle(
            "explainability",
            lambda o: o.llm_judge(
                "Rate response quality from 1-5 based on: 1. Clarity 2. Accuracy 3. Completeness"
            ),
        )
        .build()
    )

    assert config.spec_id == "multi_step_test"
    assert len(config.pipeline) == 2

    # Check first step
    step1 = config.pipeline[0]
    assert step1.template == "Summarize {{document}}"
    assert step1.adapter == "openai"
    assert "document" in step1.axes

    # Check second step
    step2 = config.pipeline[1]
    assert step2.template == "Analyze: {{stage_1_output}} using {{method}}"
    assert step2.adapter == "anthropic"
    assert "method" in step2.axes


def test_pipeline_with_factory(evaluation_factory):
    """Test using the evaluation factory with pipeline format."""
    config = evaluation_factory.with_single_axis("param", ["value1", "value2"])

    assert config.spec_id == "single_axis_test"
    assert len(config.pipeline) == 1
    assert config.pipeline[0].template == "Test {{param}}"
    assert "param" in config.pipeline[0].axes


def test_comprehensive_pipeline(config_builder):
    """Test comprehensive pipeline configuration."""
    config = (
        config_builder.spec_id("comprehensive_test")
        .add_pipeline_step(
            template="Analyze {{topic}} with {{approach}}",
            adapter="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            axes={
                "topic": {
                    "type": "categorical",
                    "values": ["AI", "ML", "DL"],
                    "weights": [0.4, 0.4, 0.2],
                },
                "approach": {
                    "type": "categorical",
                    "values": ["technical", "business"],
                },
            },
        )
        .with_oracle(
            "accuracy",
            lambda o: o.embedding_similarity("Comprehensive analysis", threshold=0.85),
        )
        .with_oracle(
            "explainability",
            lambda o: o.llm_judge(
                "Rate quality from 1-5 based on: 1. Technical accuracy "
                "2. Clarity of explanation 3. Depth of analysis"
            ),
        )
        .with_variants(500)
        .with_sampling(method="latin_hypercube", random_seed=42)
        .build()
    )

    assert isinstance(config, EvaluationConfig)
    assert config.spec_id == "comprehensive_test"
    assert len(config.pipeline) == 1
    assert config.n_variants == 500
    assert config.sampling.method == "latin_hypercube"
    assert (
        len([o for o in [config.oracles.accuracy, config.oracles.explainability] if o])
        >= 1
    )


def test_validation_with_new_format(tmp_path):
    """Test that validation works with new pipeline format."""
    yaml_content = """
spec_id: validation_test
pipeline:
  - template: "Test {{param}}"
    adapter: openai
    model: gpt-3.5-turbo
    axes:
      param:
        type: categorical
        values: ["a", "b"]
oracles:
  accuracy:
    type: embedding_similarity
    canonical_answer: "Test answer"
    threshold: 0.8
"""

    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)

    config = load_yaml_config(yaml_file)
    assert config.spec_id == "validation_test"
    assert len(config.pipeline) == 1
    assert config.pipeline[0].template == "Test {{param}}"
