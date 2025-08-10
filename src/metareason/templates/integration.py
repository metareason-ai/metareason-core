"""Integration between template rendering and sampling systems."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..config import EvaluationConfig
from ..sampling import BaseSampler, SampleResult
from .engine import TemplateEngine
from .renderer import BatchRenderer, RenderResult
from .validator import TemplateValidator, ValidationLevel, ValidationResult


@dataclass
class PromptGenerationResult:
    """Result of prompt generation from sampling."""

    prompts: List[str]
    samples: np.ndarray
    contexts: List[Dict[str, Any]]
    render_result: RenderResult
    validation_result: ValidationResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if generation was successful."""
        return (
            self.validation_result.is_valid and self.render_result.success_rate > 95.0
        )


class PromptGenerator:
    """Generate prompts from sampled parameters using templates."""

    def __init__(
        self,
        config: EvaluationConfig,
        engine: Optional[TemplateEngine] = None,
        validator: Optional[TemplateValidator] = None,
        renderer: Optional[BatchRenderer] = None,
    ) -> None:
        """Initialize the prompt generator.

        Args:
            config: Evaluation configuration
            engine: Template engine (creates new if None)
            validator: Template validator (creates new if None)
            renderer: Batch renderer (creates new if None)
        """
        self.config = config
        self.engine = engine or TemplateEngine()
        self.validator = validator or TemplateValidator(
            engine=self.engine, level=ValidationLevel.STANDARD
        )
        self.renderer = renderer or BatchRenderer(
            engine=self.engine, batch_size=100, max_workers=4
        )

    def validate_template(self) -> ValidationResult:
        """Validate the template from configuration.

        Returns:
            ValidationResult with any errors or warnings
        """
        # Get expected variables from axes
        expected_variables = set(self.config.axes.keys())

        # Validate template
        return self.validator.validate(
            self.config.prompt_template,
            expected_variables=expected_variables,
            max_length=10000,
        )

    def generate_from_samples(
        self,
        sample_result: SampleResult,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> PromptGenerationResult:
        """Generate prompts from sampling results.

        Args:
            sample_result: Results from sampling
            progress_callback: Optional progress callback

        Returns:
            PromptGenerationResult with generated prompts
        """
        # First validate the template
        validation_result = self.validate_template()

        if not validation_result.is_valid:
            # Return early with validation errors
            return PromptGenerationResult(
                prompts=[],
                samples=sample_result.samples,
                contexts=[],
                render_result=RenderResult(
                    rendered_prompts=[],
                    success_count=0,
                    error_count=len(validation_result.errors),
                    errors=[(0, e) for e in validation_result.errors],
                ),
                validation_result=validation_result,
                metadata={"validation_failed": True},
            )

        # Convert samples to contexts
        contexts = self._samples_to_contexts(sample_result)

        # Render prompts
        render_result = self.renderer.render(
            self.config.prompt_template,
            contexts,
            progress_callback=progress_callback,
            strict=True,
            parallel=len(contexts) >= 100,
        )

        return PromptGenerationResult(
            prompts=render_result.rendered_prompts,
            samples=sample_result.samples,
            contexts=contexts,
            render_result=render_result,
            validation_result=validation_result,
            metadata={
                "n_samples": len(contexts),
                "n_axes": len(self.config.axes),
                "sampling_metadata": sample_result.metadata,
            },
        )

    def generate_from_sampler(
        self,
        sampler: BaseSampler,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> PromptGenerationResult:
        """Generate prompts directly from a sampler.

        Args:
            sampler: The sampler to use
            progress_callback: Optional progress callback

        Returns:
            PromptGenerationResult with generated prompts
        """
        # Generate samples
        sample_result = sampler.sample()

        # Generate prompts from samples
        return self.generate_from_samples(sample_result, progress_callback)

    def _samples_to_contexts(self, sample_result: SampleResult) -> List[Dict[str, Any]]:
        """Convert sample array to list of context dictionaries.

        Args:
            sample_result: Sampling results

        Returns:
            List of context dictionaries for template rendering
        """
        contexts = []
        axis_names = list(self.config.axes.keys())

        for sample_row in sample_result.samples:
            context = {}
            for i, axis_name in enumerate(axis_names):
                value = sample_row[i]

                # Convert numpy types to Python types for Jinja2
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    value = value.tolist()

                context[axis_name] = value

            contexts.append(context)

        return contexts

    def estimate_generation_cost(
        self, n_samples: int, avg_prompt_length: int = 200
    ) -> Dict[str, Any]:
        """Estimate cost and resources for prompt generation.

        Args:
            n_samples: Number of samples to generate
            avg_prompt_length: Estimated average prompt length

        Returns:
            Dictionary with cost estimates
        """
        # Memory estimates
        memory_estimates = self.renderer.estimate_memory_usage(
            self.config.prompt_template, n_samples, avg_context_size=100
        )

        # Time estimates (rough)
        time_per_prompt = 0.001  # 1ms per prompt (conservative)
        if n_samples >= 100:
            # Parallel processing is faster
            time_per_prompt = 0.0005

        estimated_time = n_samples * time_per_prompt

        # Token estimates (for LLM costs)
        avg_tokens_per_prompt = avg_prompt_length / 4  # Rough token estimate
        total_tokens = n_samples * avg_tokens_per_prompt

        return {
            "n_samples": n_samples,
            "memory_mb": memory_estimates["total_estimated_mb"],
            "estimated_time_seconds": estimated_time,
            "total_tokens": int(total_tokens),
            "recommended_batch_size": memory_estimates["recommended_batch_size"],
            "parallel_processing": n_samples >= 100,
        }


def generate_prompts_from_config(
    config: EvaluationConfig,
    sampler: BaseSampler,
    progress_callback: Optional[Callable[[int], None]] = None,
    validate_outputs: bool = False,
) -> PromptGenerationResult:
    """Convenience function to generate prompts from configuration.

    Args:
        config: Evaluation configuration
        sampler: Sampler to use for parameter generation
        progress_callback: Optional progress callback
        validate_outputs: Whether to validate generated outputs

    Returns:
        PromptGenerationResult with all generated prompts
    """
    generator = PromptGenerator(config)
    result = generator.generate_from_sampler(sampler, progress_callback)

    # Optionally validate outputs
    if validate_outputs and result.is_successful:
        validator = TemplateValidator()
        output_errors = []

        for i, prompt in enumerate(result.prompts[:10]):  # Check first 10
            output_validation = validator.validate_output(
                prompt, max_length=5000, forbidden_patterns=[r"ERROR:", r"undefined"]
            )
            if not output_validation.is_valid:
                output_errors.append((i, output_validation.errors))

        if output_errors:
            result.metadata["output_validation_errors"] = output_errors

    return result
