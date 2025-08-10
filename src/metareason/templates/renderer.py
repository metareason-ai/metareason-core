"""Batch rendering system for MetaReason templates."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .engine import TemplateEngine


@dataclass
class RenderResult:
    """Result of a batch rendering operation."""

    rendered_prompts: List[str]
    success_count: int
    error_count: int
    errors: List[Tuple[int, str]] = field(default_factory=list)
    render_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class BatchRenderer:
    """Efficient batch rendering with progress reporting and error handling."""

    def __init__(
        self,
        engine: Optional[TemplateEngine] = None,
        batch_size: int = 100,
        max_workers: int = 4,
        memory_limit_mb: int = 500,
    ) -> None:
        """Initialize the batch renderer.

        Args:
            engine: Template engine to use (creates new if None)
            batch_size: Number of templates to render in each batch
            max_workers: Maximum worker threads for parallel processing
            memory_limit_mb: Soft memory limit in MB for batch processing
        """
        self.engine = engine or TemplateEngine()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb

        # Error accumulator
        self._errors: List[Tuple[int, str]] = []

    def render(
        self,
        template_string: str,
        contexts: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]] = None,
        strict: bool = True,
        parallel: bool = True,
    ) -> RenderResult:
        """Render template with multiple contexts.

        Args:
            template_string: The template to render
            contexts: List of context dictionaries
            progress_callback: Callback for progress updates
            strict: Raise errors on undefined variables
            parallel: Use parallel processing for large batches

        Returns:
            RenderResult with all rendered prompts and statistics
        """
        start_time = time.time()
        n_contexts = len(contexts)

        # Reset error accumulator
        self._errors = []

        # Validate template once
        errors = self.engine.validate_template(template_string)
        if errors:
            return RenderResult(
                rendered_prompts=[],
                success_count=0,
                error_count=len(errors),
                errors=[(0, error) for error in errors],
                render_time=time.time() - start_time,
                metadata={"validation_failed": True},
            )

        # Compile template once for efficiency
        template = self.engine.compile_template(template_string)

        # Decide on processing strategy
        if n_contexts < 100 or not parallel:
            # Sequential processing for small batches
            results = self._render_sequential(
                template, contexts, progress_callback, strict
            )
        else:
            # Parallel processing for large batches
            results = self._render_parallel(
                template, contexts, progress_callback, strict
            )

        # Count successes and errors
        success_count = sum(1 for r in results if not r.startswith("ERROR:"))
        error_count = len(self._errors)

        render_time = time.time() - start_time

        return RenderResult(
            rendered_prompts=results,
            success_count=success_count,
            error_count=error_count,
            errors=self._errors,
            render_time=render_time,
            metadata={
                "total_contexts": n_contexts,
                "batch_size": self.batch_size,
                "parallel": parallel and n_contexts >= 100,
                "avg_time_per_prompt": (
                    render_time / n_contexts if n_contexts > 0 else 0
                ),
            },
        )

    def _render_sequential(
        self,
        template: Any,
        contexts: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]],
        strict: bool,
    ) -> List[str]:
        """Render templates sequentially."""
        results = []

        for i, context in enumerate(contexts):
            try:
                result = template.render(context)
                results.append(result)
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                results.append(error_msg)
                self._errors.append((i, str(e)))

            # Report progress
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(10)

        # Final progress update
        if progress_callback:
            remaining = len(contexts) % 10
            if remaining > 0:
                progress_callback(remaining)

        return results

    def _render_parallel(
        self,
        template: Any,
        contexts: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]],
        strict: bool,
    ) -> List[str]:
        """Render templates in parallel batches."""
        results = [None] * len(contexts)
        completed = 0

        # Split into batches
        batches = self._create_batches(contexts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch jobs
            future_to_batch = {
                executor.submit(
                    self._render_batch, template, batch, batch_idx, strict
                ): (batch_idx, len(batch))
                for batch_idx, batch in batches
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx, batch_size = future_to_batch[future]

                try:
                    batch_results, batch_errors = future.result()

                    # Store results in correct positions
                    start_idx = batch_idx * self.batch_size
                    for i, result in enumerate(batch_results):
                        results[start_idx + i] = result

                    # Accumulate errors with correct indices
                    for error_idx, error_msg in batch_errors:
                        self._errors.append((start_idx + error_idx, error_msg))

                    # Update progress
                    completed += batch_size
                    if progress_callback:
                        progress_callback(batch_size)

                except Exception as e:
                    # Handle batch failure
                    start_idx = batch_idx * self.batch_size
                    for i in range(batch_size):
                        results[start_idx + i] = f"ERROR: Batch processing failed: {e}"
                        self._errors.append((start_idx + i, str(e)))

        return results

    def _create_batches(
        self, contexts: List[Dict[str, Any]]
    ) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """Split contexts into batches."""
        batches = []
        n_contexts = len(contexts)

        for i in range(0, n_contexts, self.batch_size):
            batch = contexts[i : i + self.batch_size]
            batch_idx = i // self.batch_size
            batches.append((batch_idx, batch))

        return batches

    def _render_batch(
        self,
        template: Any,
        batch: List[Dict[str, Any]],
        batch_idx: int,
        strict: bool,
    ) -> Tuple[List[str], List[Tuple[int, str]]]:
        """Render a single batch of templates."""
        results = []
        errors = []

        for i, context in enumerate(batch):
            try:
                result = template.render(context)
                results.append(result)
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                results.append(error_msg)
                errors.append((i, str(e)))

        return results, errors

    def estimate_memory_usage(
        self, template_string: str, n_contexts: int, avg_context_size: int = 100
    ) -> Dict[str, float]:
        """Estimate memory usage for batch rendering.

        Args:
            template_string: The template string
            n_contexts: Number of contexts to render
            avg_context_size: Average size of context dict in bytes

        Returns:
            Dictionary with memory estimates in MB
        """
        # Rough estimates
        template_size = len(template_string) / 1024 / 1024  # MB
        contexts_size = (n_contexts * avg_context_size) / 1024 / 1024  # MB

        # Assume each rendered prompt is ~2x template size on average
        output_size = (n_contexts * len(template_string) * 2) / 1024 / 1024  # MB

        # Working memory for processing (rough estimate)
        working_memory = (template_size + contexts_size) * 0.5

        total_estimated = template_size + contexts_size + output_size + working_memory

        return {
            "template_size_mb": template_size,
            "contexts_size_mb": contexts_size,
            "output_size_mb": output_size,
            "working_memory_mb": working_memory,
            "total_estimated_mb": total_estimated,
            "recommended_batch_size": self._recommend_batch_size(total_estimated),
        }

    def _recommend_batch_size(self, total_memory_mb: float) -> int:
        """Recommend optimal batch size based on memory usage."""
        if total_memory_mb < 100:
            return 500
        elif total_memory_mb < 500:
            return 200
        elif total_memory_mb < 1000:
            return 100
        else:
            return 50
