"""Integration tests for the evaluation pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metareason.adapters.base import CompletionResponse
from metareason.oracles.base import OracleResult
from metareason.pipeline import PipelineRunner
from metareason.pipeline.models import PipelineResult, StepResult
from metareason.sampling.base import SampleResult
from tests.fixtures.config_builders import ConfigBuilder


@pytest.fixture
def simple_config():
    """Create a simple pipeline configuration for testing."""
    return (
        ConfigBuilder()
        .spec_id("test_pipeline")
        .add_pipeline_step(
            template="Test prompt with {{variable}}",
            adapter="ollama",
            model="llama3",
            temperature=0.7,
            axes={"variable": {"type": "categorical", "values": ["A", "B", "C"]}},
        )
        .with_variants(100)
        .with_oracle(
            "accuracy",
            lambda o: o.embedding_similarity(
                "Expected correct response for testing", threshold=0.8
            ),
        )
        .build()
    )


@pytest.fixture
def mock_adapter():
    """Create a mock adapter for testing."""
    adapter = AsyncMock()
    adapter.initialize = AsyncMock()
    adapter.cleanup = AsyncMock()
    adapter.complete = AsyncMock(
        return_value=CompletionResponse(content="Mock response", model="llama3")
    )
    return adapter


@pytest.fixture
def mock_oracle():
    """Create a mock oracle for testing."""
    oracle = AsyncMock()
    oracle.initialize = AsyncMock()
    oracle.cleanup = AsyncMock()
    oracle.evaluate = AsyncMock(
        return_value=OracleResult(score=0.8, metadata={"mock": True})
    )
    oracle.get_name = MagicMock(return_value="mock_oracle")
    return oracle


class TestPipelineRunner:
    """Test the main pipeline runner."""

    def test_pipeline_runner_initialization(self, simple_config):
        """Test pipeline runner can be initialized."""
        runner = PipelineRunner(simple_config)
        assert runner.config == simple_config
        assert runner.max_concurrent == 10
        assert runner.step_executor is not None

    @pytest.mark.asyncio
    async def test_create_execution_plan(self, simple_config):
        """Test execution plan creation."""
        runner = PipelineRunner(simple_config)
        plan = await runner.create_execution_plan()

        assert plan.config == simple_config
        assert plan.estimated_samples == 100
        assert plan.estimated_prompts == 100  # 100 samples * 1 step
        assert plan.estimated_api_calls == 100
        assert len(plan.steps) == 1
        assert plan.steps[0]["adapter"] == "ollama"
        assert plan.steps[0]["model"] == "llama3"

    @pytest.mark.asyncio
    async def test_generate_samples(self, simple_config):
        """Test sample generation."""
        runner = PipelineRunner(simple_config)
        samples = await runner._generate_samples()

        assert isinstance(samples, SampleResult)
        assert samples.samples.shape[0] == 100  # n_variants
        assert samples.metadata["n_samples"] == 100
        assert "variable" in samples.metadata.get("axis_names", [])

    def test_samples_to_contexts(self, simple_config):
        """Test conversion of samples to template contexts."""
        runner = PipelineRunner(simple_config)

        # Create mock sample result
        import numpy as np

        samples = SampleResult(
            samples=np.array([["A"], ["B"], ["C"]]),
            metadata={"axis_names": ["variable"]},
        )

        contexts = runner._samples_to_contexts(samples)

        assert len(contexts) == 3
        assert contexts[0] == {"variable": "A"}
        assert contexts[1] == {"variable": "B"}
        assert contexts[2] == {"variable": "C"}

    @pytest.mark.asyncio
    @patch("metareason.pipeline.executor.AdapterFactory")
    async def test_pipeline_execution_flow(
        self, mock_adapter_factory, simple_config, mock_adapter, mock_oracle
    ):
        """Test the complete pipeline execution flow."""
        # Mock adapter factory
        mock_adapter_factory.create.return_value = mock_adapter

        # Mock oracle creation
        with patch.object(PipelineRunner, "_create_oracle", return_value=mock_oracle):
            runner = PipelineRunner(simple_config, max_concurrent=2)

            # Run pipeline
            result = await runner.run()

            # Verify result structure
            assert isinstance(result, PipelineResult)
            assert result.config == simple_config
            assert result.execution_id is not None
            assert result.start_time is not None
            assert result.end_time is not None

            # Verify step execution
            assert len(result.step_results) == 1
            step_result = result.step_results[0]
            assert isinstance(step_result, StepResult)
            assert step_result.step_index == 0
            assert step_result.step_name == "ollama/llama3"

            # Verify oracle evaluation
            assert len(result.oracle_results) >= 1
            oracle_results = list(result.oracle_results.values())[0]
            assert all(isinstance(r, OracleResult) for r in oracle_results)

            # Verify adapter and oracle were called
            mock_adapter.initialize.assert_called()
            mock_adapter.cleanup.assert_called()
            mock_oracle.initialize.assert_called()
            mock_oracle.cleanup.assert_called()


class TestStepExecutor:
    """Test the step executor component."""

    @pytest.mark.asyncio
    @patch("metareason.pipeline.executor.AdapterFactory")
    async def test_step_execution(self, mock_adapter_factory, mock_adapter):
        """Test individual step execution."""
        from metareason.config.models import PipelineStep
        from metareason.pipeline.executor import StepExecutor

        # Setup
        mock_adapter_factory.create.return_value = mock_adapter
        executor = StepExecutor(max_concurrent=2)

        step = PipelineStep(
            template="Test {{var}}",
            adapter="ollama",
            model="llama3",
            axes={"var": {"type": "categorical", "values": ["A", "B"]}},
        )

        contexts = [{"var": "A"}, {"var": "B"}]

        # Execute step
        result = await executor.execute_step(step, 0, contexts)

        # Verify result
        assert isinstance(result, StepResult)
        assert result.step_index == 0
        assert result.step_name == "ollama/llama3"
        assert len(result.prompts) == 2
        assert len(result.responses) == 2
        assert result.success_rate == 1.0

        # Verify adapter was called correctly
        mock_adapter.initialize.assert_called_once()
        mock_adapter.cleanup.assert_called_once()
        assert mock_adapter.complete.call_count == 2


class TestResultExporter:
    """Test the result exporter component."""

    def test_result_to_dict(self, simple_config):
        """Test conversion of pipeline result to dictionary."""
        from datetime import datetime

        import numpy as np

        from metareason.results.exporter import ResultExporter

        # Create mock result
        result = PipelineResult(
            config=simple_config,
            samples=SampleResult(
                samples=np.array([["A"], ["B"]]), metadata={"n_samples": 2}
            ),
            step_results=[
                StepResult(
                    step_index=0,
                    step_name="test_step",
                    prompts=["Test A", "Test B"],
                    responses=[
                        CompletionResponse(content="Response A", model="llama3"),
                        CompletionResponse(content="Response B", model="llama3"),
                    ],
                )
            ],
            oracle_results={
                "test_oracle": [
                    OracleResult(score=0.8, metadata={}),
                    OracleResult(score=0.9, metadata={}),
                ]
            },
            start_time=datetime.now(),
        )
        result.finalize()

        exporter = ResultExporter()
        data = exporter._result_to_dict(result)

        # Verify structure
        assert "execution_summary" in data
        assert "config" in data
        assert "samples" in data
        assert "step_results" in data
        assert "oracle_results" in data

        # Verify content
        assert len(data["step_results"]) == 1
        assert data["step_results"][0]["step_name"] == "test_step"
        assert "test_oracle" in data["oracle_results"]
        assert len(data["oracle_results"]["test_oracle"]) == 2


class TestResultFormatter:
    """Test the result formatter component."""

    def test_format_summary(self, simple_config):
        """Test summary formatting."""
        from datetime import datetime

        import numpy as np

        from metareason.results.formatter import ResultFormatter

        # Create mock result
        result = PipelineResult(
            config=simple_config,
            samples=SampleResult(samples=np.array([["A"]]), metadata={"n_samples": 1}),
            step_results=[
                StepResult(
                    step_index=0,
                    step_name="test_step",
                    prompts=["Test"],
                    responses=[CompletionResponse(content="Response", model="llama3")],
                )
            ],
            oracle_results={},
            start_time=datetime.now(),
        )
        result.finalize()

        formatter = ResultFormatter()
        summary = formatter.format_summary(result)

        # Verify summary contains key information
        assert simple_config.spec_id in summary
        assert "SUCCESS" in summary or "PARTIAL" in summary
        assert "100.0%" in summary  # success rate
        assert "1" in summary  # sample count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
