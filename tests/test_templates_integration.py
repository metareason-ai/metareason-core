"""Tests for template integration module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from metareason.sampling.base import SampleResult
from metareason.templates.integration import (
    PromptGenerationResult,
    PromptGenerator,
    generate_prompts_from_config,
)
from metareason.templates.renderer import RenderResult
from metareason.templates.validator import ValidationResult


class TestPromptGenerationResult:
    """Test PromptGenerationResult class."""

    def test_initialization(self):
        """Test initialization of PromptGenerationResult."""
        prompts = ["Hello", "World"]
        samples = np.array([[1, 2], [3, 4]])
        contexts = [{"a": 1}, {"a": 2}]
        render_result = RenderResult(
            rendered_prompts=prompts,
            success_count=2,
            error_count=0,
            errors=[],
        )
        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            variables={"a"},
        )

        result = PromptGenerationResult(
            prompts=prompts,
            samples=samples,
            contexts=contexts,
            render_result=render_result,
            validation_result=validation_result,
        )

        assert result.prompts == prompts
        assert np.array_equal(result.samples, samples)
        assert result.contexts == contexts
        assert result.render_result == render_result
        assert result.validation_result == validation_result
        assert result.metadata == {}

    def test_is_successful_true(self):
        """Test is_successful property returns True for successful generation."""
        render_result = RenderResult(
            rendered_prompts=["test"],
            success_count=1,
            error_count=0,
            errors=[],
        )
        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            variables=set(),
        )

        result = PromptGenerationResult(
            prompts=["test"],
            samples=np.array([[1]]),
            contexts=[{"a": 1}],
            render_result=render_result,
            validation_result=validation_result,
        )

        assert result.is_successful is True

    def test_is_successful_false_invalid_validation(self):
        """Test is_successful returns False for invalid validation."""
        render_result = RenderResult(
            rendered_prompts=["test"],
            success_count=1,
            error_count=0,
            errors=[],
        )
        validation_result = ValidationResult(
            is_valid=False,
            errors=["Template error"],
            warnings=[],
            variables=set(),
        )

        result = PromptGenerationResult(
            prompts=["test"],
            samples=np.array([[1]]),
            contexts=[{"a": 1}],
            render_result=render_result,
            validation_result=validation_result,
        )

        assert result.is_successful is False

    def test_is_successful_false_low_success_rate(self):
        """Test is_successful returns False for low success rate."""
        # Create render result with 90% success rate (below 95% threshold)
        render_result = Mock()
        render_result.success_rate = 90.0

        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            variables=set(),
        )

        result = PromptGenerationResult(
            prompts=["test"],
            samples=np.array([[1]]),
            contexts=[{"a": 1}],
            render_result=render_result,
            validation_result=validation_result,
        )

        assert result.is_successful is False


class TestPromptGenerator:
    """Test PromptGenerator class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock evaluation config."""
        config = Mock()
        config.axes = {"param1": Mock(), "param2": Mock()}
        config.prompt_template = "Template with {{param1}} and {{param2}}"
        return config

    @pytest.fixture
    def generator(self, mock_config):
        """Create PromptGenerator with mocked dependencies."""
        with (
            patch(
                "metareason.templates.integration.TemplateEngine"
            ) as mock_engine_class,
            patch(
                "metareason.templates.integration.TemplateValidator"
            ) as mock_validator_class,
            patch(
                "metareason.templates.integration.BatchRenderer"
            ) as mock_renderer_class,
        ):

            mock_engine = Mock()
            mock_validator = Mock()
            mock_renderer = Mock()

            mock_engine_class.return_value = mock_engine
            mock_validator_class.return_value = mock_validator
            mock_renderer_class.return_value = mock_renderer

            generator = PromptGenerator(mock_config)
            generator.engine = mock_engine
            generator.validator = mock_validator
            generator.renderer = mock_renderer

            return generator

    def test_initialization_with_defaults(self, mock_config):
        """Test initialization with default dependencies."""
        with (
            patch(
                "metareason.templates.integration.TemplateEngine"
            ) as mock_engine_class,
            patch(
                "metareason.templates.integration.TemplateValidator"
            ) as mock_validator_class,
            patch(
                "metareason.templates.integration.BatchRenderer"
            ) as mock_renderer_class,
        ):

            generator = PromptGenerator(mock_config)

            assert generator.config == mock_config
            mock_engine_class.assert_called_once()
            mock_validator_class.assert_called_once()
            mock_renderer_class.assert_called_once()

    def test_initialization_with_provided_dependencies(self, mock_config):
        """Test initialization with provided dependencies."""
        mock_engine = Mock()
        mock_validator = Mock()
        mock_renderer = Mock()

        generator = PromptGenerator(
            mock_config,
            engine=mock_engine,
            validator=mock_validator,
            renderer=mock_renderer,
        )

        assert generator.config == mock_config
        assert generator.engine == mock_engine
        assert generator.validator == mock_validator
        assert generator.renderer == mock_renderer

    def test_validate_template(self, generator, mock_config):
        """Test template validation."""
        expected_validation = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            variables={"param1", "param2"},
        )
        generator.validator.validate.return_value = expected_validation

        result = generator.validate_template()

        generator.validator.validate.assert_called_once_with(
            mock_config.prompt_template,
            expected_variables={"param1", "param2"},
            max_length=10000,
        )
        assert result == expected_validation

    def test_generate_from_samples_validation_failed(self, generator):
        """Test generate_from_samples when validation fails."""
        # Setup validation failure
        validation_result = ValidationResult(
            is_valid=False,
            errors=["Template error"],
            warnings=[],
            variables=set(),
        )
        generator.validator.validate.return_value = validation_result

        # Create sample result
        sample_result = SampleResult(
            samples=np.array([[1, 2]]),
            metadata={"test": "data"},
        )

        result = generator.generate_from_samples(sample_result)

        assert result.prompts == []
        assert np.array_equal(result.samples, sample_result.samples)
        assert result.contexts == []
        assert result.validation_result == validation_result
        assert result.metadata["validation_failed"] is True
        assert result.render_result.success_count == 0
        assert result.render_result.error_count == 1

    def test_generate_from_samples_success(self, generator, mock_config):
        """Test successful generate_from_samples."""
        # Setup successful validation
        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            variables={"param1", "param2"},
        )
        generator.validator.validate.return_value = validation_result

        # Setup successful rendering
        render_result = RenderResult(
            rendered_prompts=["Generated prompt 1", "Generated prompt 2"],
            success_count=2,
            error_count=0,
            errors=[],
        )
        generator.renderer.render.return_value = render_result

        # Create sample result
        sample_result = SampleResult(
            samples=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metadata={"sampling_info": "test"},
        )

        result = generator.generate_from_samples(sample_result)

        assert result.prompts == render_result.rendered_prompts
        assert np.array_equal(result.samples, sample_result.samples)
        assert len(result.contexts) == 2
        assert result.validation_result == validation_result
        assert result.render_result == render_result
        assert result.metadata["n_samples"] == 2
        assert result.metadata["n_axes"] == 2
        assert result.metadata["sampling_metadata"] == sample_result.metadata

    def test_generate_from_sampler(self, generator):
        """Test generate_from_sampler method."""
        # Create mock sampler
        mock_sampler = Mock()
        sample_result = SampleResult(
            samples=np.array([[1.0, 2.0]]),
            metadata={"test": "data"},
        )
        mock_sampler.sample.return_value = sample_result

        # Mock generate_from_samples to return a result
        expected_result = Mock()
        with patch.object(
            generator, "generate_from_samples", return_value=expected_result
        ) as mock_generate:
            result = generator.generate_from_sampler(
                mock_sampler, progress_callback=None
            )

            mock_sampler.sample.assert_called_once()
            mock_generate.assert_called_once_with(sample_result, None)
            assert result == expected_result

    def test_samples_to_contexts(self, generator, mock_config):
        """Test _samples_to_contexts method."""
        # Setup config with known axes
        mock_config.axes = {"param1": Mock(), "param2": Mock()}

        # Create sample result with different numpy types
        # Can't mix numpy arrays with scalars in same array, so test separately
        samples = np.array(
            [
                [np.int32(1), np.float64(2.5)],
                [np.int64(3), np.float32(4.7)],
                [5, 6.8],
            ]
        )
        sample_result = SampleResult(samples=samples, metadata={})

        contexts = generator._samples_to_contexts(sample_result)

        assert len(contexts) == 3

        # Check first context
        assert contexts[0]["param1"] == 1
        assert contexts[0]["param2"] == 2.5
        # Note: numpy types are converted to Python types
        assert isinstance(contexts[0]["param1"], (int, float))  # Could be int or float
        assert isinstance(contexts[0]["param2"], float)

        # Check second context
        assert contexts[1]["param1"] == 3
        assert abs(contexts[1]["param2"] - 4.7) < 1e-6  # Account for float32 precision
        assert isinstance(contexts[1]["param1"], (int, float))  # Could be int or float
        assert isinstance(contexts[1]["param2"], float)

        # Check third context
        assert contexts[2]["param1"] == 5
        assert contexts[2]["param2"] == 6.8

    def test_samples_to_contexts_with_numpy_array(self, generator, mock_config):
        """Test _samples_to_contexts method with numpy arrays."""
        # Setup config with one axis
        mock_config.axes = {"param1": Mock()}

        # Create sample with numpy array values
        sample_data = []
        for i in range(2):
            sample_data.append([np.array([i, i + 1])])

        samples = np.array(sample_data, dtype=object)
        sample_result = SampleResult(samples=samples, metadata={})

        contexts = generator._samples_to_contexts(sample_result)

        assert len(contexts) == 2
        assert contexts[0]["param1"] == [0, 1]
        assert contexts[1]["param1"] == [1, 2]
        assert isinstance(contexts[0]["param1"], list)
        assert isinstance(contexts[1]["param1"], list)

    def test_estimate_generation_cost(self, generator):
        """Test estimate_generation_cost method."""
        # Mock renderer memory estimation
        memory_estimates = {
            "total_estimated_mb": 128.5,
            "recommended_batch_size": 50,
        }
        generator.renderer.estimate_memory_usage.return_value = memory_estimates

        # Test with small batch (no parallel processing)
        cost_estimate = generator.estimate_generation_cost(50, avg_prompt_length=150)

        generator.renderer.estimate_memory_usage.assert_called_with(
            generator.config.prompt_template,
            50,
            avg_context_size=100,
        )

        assert cost_estimate["n_samples"] == 50
        assert cost_estimate["memory_mb"] == 128.5
        assert cost_estimate["estimated_time_seconds"] == 50 * 0.001  # 1ms per prompt
        assert cost_estimate["total_tokens"] == int(50 * 150 / 4)
        assert cost_estimate["recommended_batch_size"] == 50
        assert cost_estimate["parallel_processing"] is False

    def test_estimate_generation_cost_large_batch(self, generator):
        """Test estimate_generation_cost with large batch (parallel processing)."""
        # Mock renderer memory estimation
        memory_estimates = {
            "total_estimated_mb": 512.0,
            "recommended_batch_size": 100,
        }
        generator.renderer.estimate_memory_usage.return_value = memory_estimates

        # Test with large batch (parallel processing)
        cost_estimate = generator.estimate_generation_cost(1000, avg_prompt_length=200)

        assert cost_estimate["n_samples"] == 1000
        assert cost_estimate["memory_mb"] == 512.0
        assert (
            cost_estimate["estimated_time_seconds"] == 1000 * 0.0005
        )  # 0.5ms per prompt
        assert cost_estimate["total_tokens"] == int(1000 * 200 / 4)
        assert cost_estimate["recommended_batch_size"] == 100
        assert cost_estimate["parallel_processing"] is True


class TestGeneratePromptsFromConfig:
    """Test the convenience function generate_prompts_from_config."""

    def test_generate_prompts_from_config_basic(self):
        """Test basic usage of generate_prompts_from_config."""
        mock_config = Mock()
        mock_sampler = Mock()

        expected_result = Mock()
        expected_result.is_successful = True

        with patch(
            "metareason.templates.integration.PromptGenerator"
        ) as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate_from_sampler.return_value = expected_result
            mock_generator_class.return_value = mock_generator

            result = generate_prompts_from_config(mock_config, mock_sampler)

            mock_generator_class.assert_called_once_with(mock_config)
            mock_generator.generate_from_sampler.assert_called_once_with(
                mock_sampler, None
            )
            assert result == expected_result

    def test_generate_prompts_from_config_with_callback(self):
        """Test generate_prompts_from_config with progress callback."""
        mock_config = Mock()
        mock_sampler = Mock()
        mock_callback = Mock()

        expected_result = Mock()
        expected_result.is_successful = True

        with patch(
            "metareason.templates.integration.PromptGenerator"
        ) as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate_from_sampler.return_value = expected_result
            mock_generator_class.return_value = mock_generator

            result = generate_prompts_from_config(
                mock_config, mock_sampler, progress_callback=mock_callback
            )

            mock_generator.generate_from_sampler.assert_called_once_with(
                mock_sampler, mock_callback
            )
            assert result == expected_result

    def test_generate_prompts_with_output_validation_success(self):
        """Test generate_prompts_from_config with successful output validation."""
        mock_config = Mock()
        mock_sampler = Mock()

        # Create successful result
        expected_result = Mock()
        expected_result.is_successful = True
        expected_result.prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        expected_result.metadata = {}

        with (
            patch(
                "metareason.templates.integration.PromptGenerator"
            ) as mock_generator_class,
            patch(
                "metareason.templates.integration.TemplateValidator"
            ) as mock_validator_class,
        ):

            mock_generator = Mock()
            mock_generator.generate_from_sampler.return_value = expected_result
            mock_generator_class.return_value = mock_generator

            # Mock validator for output validation
            mock_validator = Mock()
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validator.validate_output.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            result = generate_prompts_from_config(
                mock_config, mock_sampler, validate_outputs=True
            )

            mock_validator_class.assert_called_once()
            # Should validate first 3 prompts (all of them in this case)
            assert mock_validator.validate_output.call_count == 3
            assert result == expected_result
            # No validation errors should be added
            assert "output_validation_errors" not in result.metadata

    def test_generate_prompts_with_output_validation_errors(self):
        """Test generate_prompts_from_config with output validation errors."""
        mock_config = Mock()
        mock_sampler = Mock()

        # Create successful result with many prompts
        expected_result = Mock()
        expected_result.is_successful = True
        expected_result.prompts = [f"Prompt {i}" for i in range(15)]
        expected_result.metadata = {}

        with (
            patch(
                "metareason.templates.integration.PromptGenerator"
            ) as mock_generator_class,
            patch(
                "metareason.templates.integration.TemplateValidator"
            ) as mock_validator_class,
        ):

            mock_generator = Mock()
            mock_generator.generate_from_sampler.return_value = expected_result
            mock_generator_class.return_value = mock_generator

            # Mock validator with some validation failures
            mock_validator = Mock()

            def mock_validate_output(prompt, **kwargs):
                validation_result = Mock()
                if "5" in prompt:  # Fail validation for prompt containing "5"
                    validation_result.is_valid = False
                    validation_result.errors = ["Contains forbidden content"]
                else:
                    validation_result.is_valid = True
                    validation_result.errors = []
                return validation_result

            mock_validator.validate_output.side_effect = mock_validate_output
            mock_validator_class.return_value = mock_validator

            result = generate_prompts_from_config(
                mock_config, mock_sampler, validate_outputs=True
            )

            # Should validate first 10 prompts only
            assert mock_validator.validate_output.call_count == 10
            assert result == expected_result
            # Should have validation errors for prompt 5
            assert "output_validation_errors" in result.metadata
            output_errors = result.metadata["output_validation_errors"]
            assert len(output_errors) == 1
            assert output_errors[0][0] == 5  # Index of the failing prompt

    def test_generate_prompts_no_validation_when_unsuccessful(self):
        """Test that output validation is skipped when generation is unsuccessful."""
        mock_config = Mock()
        mock_sampler = Mock()

        # Create unsuccessful result
        expected_result = Mock()
        expected_result.is_successful = False
        expected_result.prompts = []
        expected_result.metadata = {}

        with patch(
            "metareason.templates.integration.PromptGenerator"
        ) as mock_generator_class:
            mock_generator = Mock()
            mock_generator.generate_from_sampler.return_value = expected_result
            mock_generator_class.return_value = mock_generator

            result = generate_prompts_from_config(
                mock_config, mock_sampler, validate_outputs=True
            )

            # Since generation was unsuccessful, no validator should be created
            assert result == expected_result
            # Verify no additional metadata was added
            assert result.metadata == {}
