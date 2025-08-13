"""Pipeline step executor for individual stage execution."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..adapters.base import CompletionRequest, LLMAdapter, Message, MessageRole
from ..adapters.registry import AdapterFactory
from ..config.models import PipelineStep
from .models import StepResult

logger = logging.getLogger(__name__)


class StepExecutor:
    """Executes individual pipeline steps with LLM calls."""

    def __init__(self, max_concurrent: int = 10, retry_attempts: int = 3):
        """Initialize step executor.

        Args:
            max_concurrent: Maximum concurrent requests
            retry_attempts: Number of retry attempts for failed requests
        """
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_step(
        self,
        step: PipelineStep,
        step_index: int,
        contexts: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[int, List[str]]] = None,
        progress_callback: Optional[callable] = None,
    ) -> StepResult:
        """Execute a single pipeline step.

        Args:
            step: Pipeline step configuration
            step_index: Index of current step (0-based)
            contexts: List of context dictionaries for template rendering
            previous_outputs: Outputs from previous steps (step_index -> outputs)
            progress_callback: Optional progress callback function

        Returns:
            StepResult with responses and metadata
        """
        import time

        start_time = time.time()

        logger.info(f"Executing step {step_index}: {step.adapter}/{step.model}")

        # Create adapter
        try:
            adapter_config = self._create_adapter_config(step)
            adapter = AdapterFactory.create(adapter_config)
            await adapter.initialize()
        except Exception as e:
            logger.error(f"Failed to create adapter for step {step_index}: {e}")
            return StepResult(
                step_index=step_index,
                step_name=f"{step.adapter}/{step.model}",
                prompts=[],
                responses=[],
                errors=[e],
                timing={"setup_time": time.time() - start_time},
            )

        try:
            # Render prompts with context and previous outputs
            prompts = self._render_prompts(step, contexts, previous_outputs, step_index)

            render_time = time.time()

            # Execute LLM requests
            responses, errors = await self._execute_requests(
                adapter, step, prompts, progress_callback
            )

            execution_time = time.time()

            # Create result
            result = StepResult(
                step_index=step_index,
                step_name=f"{step.adapter}/{step.model}",
                prompts=prompts,
                responses=responses,
                errors=errors,
                timing={
                    "setup_time": render_time - start_time,
                    "render_time": render_time - start_time,
                    "execution_time": execution_time - render_time,
                    "total_time": execution_time - start_time,
                },
                metadata={
                    "adapter": step.adapter,
                    "model": step.model,
                    "temperature": step.temperature,
                    "max_tokens": step.max_tokens,
                },
            )

            logger.info(
                f"Step {step_index} completed: {len(responses)}/{len(prompts)} successful "
                f"({result.success_rate:.1%}) in {result.timing['total_time']:.2f}s"
            )

            return result

        finally:
            await adapter.cleanup()

    def _create_adapter_config(self, step: PipelineStep) -> Dict[str, Any]:
        """Create adapter configuration from pipeline step.

        Args:
            step: Pipeline step configuration

        Returns:
            Adapter configuration dictionary
        """
        from ..config.adapters import (
            AnthropicConfig,
            GoogleConfig,
            OllamaConfig,
            OpenAIConfig,
            RateLimitConfig,
            RetryConfig,
        )

        # Create proper adapter configuration based on adapter type
        retry_config = RetryConfig(max_attempts=self.retry_attempts)
        rate_limit_config = RateLimitConfig(requests_per_minute=100)

        if step.adapter == "openai":
            return OpenAIConfig(
                type="openai",
                default_model=step.model,
                retry=retry_config,
                rate_limit=rate_limit_config,
                timeout=60,
            )
        elif step.adapter == "anthropic":
            return AnthropicConfig(
                type="anthropic",
                default_model=step.model,
                retry=retry_config,
                rate_limit=rate_limit_config,
                timeout=60,
            )
        elif step.adapter == "ollama":
            return OllamaConfig(
                type="ollama",
                default_model=step.model,
                retry=retry_config,
                rate_limit=rate_limit_config,
                timeout=60,
            )
        elif step.adapter == "google":
            return GoogleConfig(
                type="google",
                default_model=step.model,
                retry=retry_config,
                rate_limit=rate_limit_config,
                timeout=60,
            )
        else:
            # Fallback for unknown adapter types
            return {
                "type": step.adapter,
                "default_model": step.model,
                "timeout": 60,
                "retry": retry_config.model_dump(),
                "rate_limit": rate_limit_config.model_dump(),
            }

    def _render_prompts(
        self,
        step: PipelineStep,
        contexts: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[int, List[str]]],
        current_step_index: int,
    ) -> List[str]:
        """Render prompts for the step with context and previous outputs.

        Args:
            step: Pipeline step configuration
            contexts: Context dictionaries from sampling
            previous_outputs: Outputs from previous steps
            current_step_index: Current step index

        Returns:
            List of rendered prompts
        """
        from jinja2 import Template

        template = Template(step.template)
        prompts = []

        for i, context in enumerate(contexts):
            # Create enhanced context with previous step outputs
            enhanced_context = context.copy()

            # Add previous step outputs as stage_N_output variables
            if previous_outputs:
                for step_idx, outputs in previous_outputs.items():
                    if step_idx < current_step_index and i < len(outputs):
                        enhanced_context[f"stage_{step_idx + 1}_output"] = outputs[i]

            try:
                prompt = template.render(**enhanced_context)
                prompts.append(prompt)
            except Exception as e:
                logger.error(
                    f"Failed to render prompt {i} for step {current_step_index}: {e}"
                )
                # Use a fallback prompt
                prompts.append(f"Error rendering template: {e}")

        return prompts

    async def _execute_requests(
        self,
        adapter: LLMAdapter,
        step: PipelineStep,
        prompts: List[str],
        progress_callback: Optional[callable] = None,
    ) -> tuple[List, List]:
        """Execute LLM requests for all prompts.

        Args:
            adapter: LLM adapter instance
            step: Pipeline step configuration
            prompts: List of prompts to execute
            progress_callback: Optional progress callback

        Returns:
            Tuple of (responses, errors)
        """
        responses = []
        errors = []

        # Create completion requests
        requests = []
        for prompt in prompts:
            request = CompletionRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=step.model,
                temperature=step.temperature or 0.7,
                max_tokens=step.max_tokens,
                top_p=step.top_p,
                frequency_penalty=step.frequency_penalty,
                presence_penalty=step.presence_penalty,
                stop=step.stop,
            )
            requests.append(request)

        # Execute requests with concurrency control
        tasks = [self._execute_single_request(adapter, request) for request in requests]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                response = await coro
                responses.append(response)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                errors.append(e)

            completed += 1
            if progress_callback:
                progress_callback(1)

        return responses, errors

    async def _execute_single_request(
        self, adapter: LLMAdapter, request: CompletionRequest
    ):
        """Execute a single LLM request with retry logic.

        Args:
            adapter: LLM adapter instance
            request: Completion request

        Returns:
            Completion response
        """
        async with self._semaphore:  # Control concurrency
            for attempt in range(self.retry_attempts):
                try:
                    response = await adapter.complete(request)
                    return response
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    logger.warning(
                        f"Request attempt {attempt + 1} failed, retrying: {e}"
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
