from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from metareason.adapters.adapter_base import AdapterResponse
from metareason.config.models import AdapterConfig, PipelineConfig
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.loader import load_calibrate_spec, load_spec
from metareason.pipeline.renderer import TemplateRenderer
from metareason.pipeline.runner import SampleResult, _process_sample

# --- TemplateRenderer ---


class TestTemplateRenderer:
    def test_render_simple_template(self):
        renderer = TemplateRenderer()
        result = renderer.render_request("Hello {{ name }}", {"name": "World"})
        assert result == "Hello World"

    def test_render_multiple_variables(self):
        renderer = TemplateRenderer()
        template = "{{ greeting }}, {{ name }}! You are {{ age }} years old."
        variables = {"greeting": "Hi", "name": "Alice", "age": 30}
        result = renderer.render_request(template, variables)
        assert result == "Hi, Alice! You are 30 years old."

    def test_render_missing_variable(self):
        renderer = TemplateRenderer()
        result = renderer.render_request(
            "Hello {{ name }} and {{ missing }}", {"name": "World"}
        )
        assert result == "Hello World and "


# --- load_spec ---


class TestLoadSpec:
    def test_load_valid_spec(self, tmp_path):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text(
            """
spec_id: test-spec
pipeline:
  - template: "Hello {{ name }}"
    adapter:
      name: ollama
    model: llama2
    temperature: 0.7
    top_p: 0.9
    max_tokens: 100
sampling:
  method: latin_hypercube
  optimization: maximin
n_variants: 3
oracles:
  judge:
    type: llm_judge
    model: llama2
    adapter:
      name: ollama
    rubric: "Score 1-5"
"""
        )
        spec = load_spec(spec_yaml)
        assert spec.spec_id == "test-spec"
        assert spec.n_variants == 3
        assert len(spec.pipeline) == 1
        assert "judge" in spec.oracles

    def test_load_invalid_spec(self, tmp_path):
        spec_yaml = tmp_path / "bad.yaml"
        spec_yaml.write_text(
            """
spec_id: test
pipeline: []
sampling:
  method: latin_hypercube
  optimization: maximin
oracles: {}
"""
        )
        with pytest.raises(ValidationError):
            load_spec(spec_yaml)


# --- load_calibrate_spec ---


class TestLoadCalibrateSpec:
    def test_load_calibrate_spec_valid(self, tmp_path):
        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-test
type: calibrate
prompt: "Test prompt"
response: "Test response"
repeats: 10
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        cfg = load_calibrate_spec(spec_yaml)
        assert cfg.spec_id == "cal-test"
        assert cfg.prompt == "Test prompt"
        assert cfg.response == "Test response"
        assert cfg.repeats == 10

    def test_load_calibrate_spec_with_file_references(self, tmp_path):
        # Create fixture files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("My test prompt from file")
        response_file = tmp_path / "response.txt"
        response_file.write_text("My test response from file")

        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-files
type: calibrate
prompt: "file:prompt.txt"
response: "file:response.txt"
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        cfg = load_calibrate_spec(spec_yaml)
        assert cfg.prompt == "My test prompt from file"
        assert cfg.response == "My test response from file"

    def test_load_calibrate_spec_file_not_found(self, tmp_path):
        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-missing
type: calibrate
prompt: "file:nonexistent.txt"
response: "inline response"
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        with pytest.raises(FileNotFoundError):
            load_calibrate_spec(spec_yaml)

    def test_load_calibrate_spec_inline_strings(self, tmp_path):
        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-inline
type: calibrate
prompt: "Just a regular prompt"
response: "Just a regular response"
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        cfg = load_calibrate_spec(spec_yaml)
        assert cfg.prompt == "Just a regular prompt"
        assert cfg.response == "Just a regular response"

    def test_load_calibrate_spec_rejects_relative_traversal(self, tmp_path):
        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-traversal
type: calibrate
prompt: "file:../../etc/passwd"
response: "inline response"
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        with pytest.raises(ValueError, match="resolves outside"):
            load_calibrate_spec(spec_yaml)

    def test_load_calibrate_spec_rejects_absolute_path(self, tmp_path):
        spec_yaml = tmp_path / "cal.yaml"
        spec_yaml.write_text(
            """
spec_id: cal-absolute
type: calibrate
prompt: "file:/etc/passwd"
response: "inline response"
oracle:
  type: llm_judge
  model: test-model
  adapter:
    name: ollama
  rubric: "Score 1-5"
"""
        )
        with pytest.raises(ValueError, match="resolves outside"):
            load_calibrate_spec(spec_yaml)


# --- _process_sample ---


def make_pipeline_config(**overrides):
    defaults = dict(
        template="Hello {{ name }}",
        adapter=AdapterConfig(name="ollama"),
        model="test-model",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestProcessSample:
    @patch("metareason.pipeline.runner.get_adapter")
    @pytest.mark.asyncio
    async def test_single_stage(self, mock_get_adapter):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text="LLM response"
        )
        mock_get_adapter.return_value = mock_adapter

        mock_oracle = AsyncMock()
        mock_oracle.evaluate.return_value = EvaluationResult(
            score=4.0, explanation="good"
        )

        pipeline = [make_pipeline_config()]
        sample = {"name": "World"}
        oracles = {"test_oracle": mock_oracle}

        result = await _process_sample(pipeline, sample, oracles)

        assert isinstance(result, SampleResult)
        assert result.original_prompt == "Hello World"
        assert result.final_response == "LLM response"
        assert result.evaluations["test_oracle"].score == 4.0
        assert result.sample_params == {"name": "World"}

    @patch("metareason.pipeline.runner.get_adapter")
    @pytest.mark.asyncio
    async def test_oracle_failure_continues(self, mock_get_adapter):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text="LLM response"
        )
        mock_get_adapter.return_value = mock_adapter

        mock_oracle = AsyncMock()
        mock_oracle.evaluate.side_effect = RuntimeError("oracle broke")

        pipeline = [make_pipeline_config()]
        sample = {"name": "World"}
        oracles = {"failing_oracle": mock_oracle}

        result = await _process_sample(pipeline, sample, oracles)

        assert result.evaluations["failing_oracle"].score == 1.0
        assert "Evaluation failed" in result.evaluations["failing_oracle"].explanation

    @patch("metareason.pipeline.runner.get_adapter")
    @pytest.mark.asyncio
    async def test_multi_stage(self, mock_get_adapter):
        mock_adapter = AsyncMock()
        mock_adapter.send_request.side_effect = [
            AdapterResponse(response_text="stage 1 output"),
            AdapterResponse(response_text="stage 2 output"),
        ]
        mock_get_adapter.return_value = mock_adapter

        pipeline = [
            make_pipeline_config(template="First: {{ name }}"),
            make_pipeline_config(template="ignored"),
        ]
        sample = {"name": "Test"}
        oracles = {}

        result = await _process_sample(pipeline, sample, oracles)

        assert result.original_prompt == "First: Test"
        assert result.final_response == "stage 2 output"

        # Verify the second call used the first stage's output as user_prompt
        calls = mock_adapter.send_request.call_args_list
        assert len(calls) == 2
        second_call_request = calls[1].args[0]
        assert second_call_request.user_prompt == "stage 1 output"


class TestAdapterReuse:
    """Test that adapters are instantiated once per pipeline stage, not per sample."""

    @patch("metareason.pipeline.runner.get_adapter")
    @pytest.mark.asyncio
    async def test_get_adapter_called_once_per_stage_not_per_sample(
        self, mock_get_adapter
    ):
        """Verify get_adapter called once per stage, not per sample.

        With 2 pipeline stages and 5 samples, get_adapter should be called
        exactly 2 times (once per stage), not 10 times (once per sample per
        stage). This mimics how run() pre-creates adapters and passes them
        to _process_sample.
        """
        mock_adapter = AsyncMock()
        mock_adapter.send_request.return_value = AdapterResponse(
            response_text="response"
        )
        mock_get_adapter.return_value = mock_adapter

        mock_oracle = AsyncMock()
        mock_oracle.evaluate.return_value = EvaluationResult(
            score=4.0, explanation="good"
        )

        pipeline = [
            make_pipeline_config(template="Stage1: {{ name }}"),
            make_pipeline_config(template="Stage2: ignored"),
        ]
        samples = [{"name": f"sample_{i}"} for i in range(5)]
        oracles = {"test_oracle": mock_oracle}

        # Pre-create adapters once per stage (as run() now does)
        adapters = [
            mock_get_adapter(pipe.adapter.name, **pipe.adapter.params)
            for pipe in pipeline
        ]

        # Process all 5 samples through the 2-stage pipeline
        import asyncio

        tasks = [
            _process_sample(pipeline, sample, oracles, adapters=adapters)
            for sample in samples
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        # get_adapter should be called once per stage = 2, not 5*2 = 10
        assert mock_get_adapter.call_count == 2
