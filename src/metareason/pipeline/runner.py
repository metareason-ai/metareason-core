"""Main pipeline runner for MetaReason evaluations."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..analysis.bayesian import MultiOracleAnalyzer
from ..config.models import EvaluationConfig
from ..config.statistical import StatisticalConfig
from ..oracles.base import BaseOracle, OracleResult
from ..oracles.embedding_similarity import EmbeddingSimilarityOracle
from ..oracles.llm_judge import LLMJudgeOracle
from ..sampling.base import SampleResult
from ..sampling.factory import create_sampler
from .executor import StepExecutor
from .models import ExecutionPlan, PipelineResult, StepResult

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Main pipeline runner for executing complete evaluations."""

    def __init__(
        self,
        config: EvaluationConfig,
        max_concurrent: int = 10,
        enable_progress: bool = True,
    ):
        """Initialize pipeline runner.

        Args:
            config: Evaluation configuration
            max_concurrent: Maximum concurrent requests
            enable_progress: Whether to show progress indicators
        """
        self.config = config
        self.max_concurrent = max_concurrent
        self.enable_progress = enable_progress
        self.step_executor = StepExecutor(max_concurrent=max_concurrent)

    async def run(self, progress_callback: Optional[callable] = None) -> PipelineResult:
        """Execute the complete evaluation pipeline.

        Args:
            progress_callback: Optional progress callback function

        Returns:
            Complete pipeline results
        """
        start_time = datetime.now()
        execution_id = start_time.strftime("%Y%m%d_%H%M%S")

        logger.info(
            f"Starting pipeline execution {execution_id} for spec {self.config.spec_id}"
        )

        try:
            # Step 1: Generate samples using Latin Hypercube Sampling
            logger.info("Generating samples using Latin Hypercube Sampling...")
            samples = await self._generate_samples()

            # Step 2: Execute pipeline steps
            logger.info(f"Executing {len(self.config.pipeline)} pipeline steps...")
            step_results = await self._execute_pipeline_steps(
                samples, progress_callback
            )

            # Step 3: Evaluate with oracles
            logger.info("Evaluating responses with oracles...")
            oracle_results = await self._evaluate_with_oracles(
                step_results, progress_callback
            )

            # Step 4: Perform Bayesian analysis
            logger.info("Performing Bayesian statistical analysis...")
            bayesian_results = await self._perform_bayesian_analysis(oracle_results)

            # Step 5: Create final result
            result = PipelineResult(
                config=self.config,
                samples=samples,
                step_results=step_results,
                oracle_results=oracle_results,
                bayesian_results=bayesian_results,
                execution_id=execution_id,
                start_time=start_time,
                metadata={
                    "max_concurrent": self.max_concurrent,
                    "total_pipeline_steps": len(self.config.pipeline),
                    "total_oracles": len(oracle_results),
                },
            )

            result.finalize()
            logger.info(f"Pipeline execution {execution_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Pipeline execution {execution_id} failed: {e}")
            # Return partial result with error information
            result = PipelineResult(
                config=self.config,
                samples=SampleResult(samples=[], metadata={}),
                step_results=[],
                oracle_results={},
                execution_id=execution_id,
                start_time=start_time,
                metadata={"error": str(e)},
            )
            result.finalize()
            raise

    async def create_execution_plan(self) -> ExecutionPlan:
        """Create an execution plan for dry-run mode.

        Returns:
            Execution plan with estimates
        """
        logger.info("Creating execution plan...")

        # Estimate samples
        estimated_samples = self.config.n_variants

        # Calculate total axes across all pipeline steps
        all_axes = {}
        for step in self.config.pipeline:
            all_axes.update(step.axes)

        # Estimate prompts (samples * pipeline steps)
        estimated_prompts = estimated_samples * len(self.config.pipeline)

        # Estimate API calls (same as prompts for now)
        estimated_api_calls = estimated_prompts

        plan = ExecutionPlan(
            config=self.config,
            estimated_samples=estimated_samples,
            estimated_prompts=estimated_prompts,
            estimated_api_calls=estimated_api_calls,
        )

        # Add step details
        for i, step in enumerate(self.config.pipeline):
            step_calls = estimated_samples
            plan.add_step_plan(
                step_index=i,
                step_name=f"Step {i+1}",
                adapter=step.adapter,
                model=step.model,
                estimated_calls=step_calls,
            )

        # Add warnings
        if estimated_api_calls > 10000:
            plan.add_warning(
                f"Large number of API calls ({estimated_api_calls:,}). Consider reducing n_variants."
            )

        if len(self.config.pipeline) > 5:
            plan.add_warning(
                f"Many pipeline steps ({len(self.config.pipeline)}). Execution may take significant time."
            )

        # Estimate duration (rough)
        avg_request_time = 2.0  # seconds per request
        plan.estimated_duration = (
            estimated_api_calls * avg_request_time
        ) / self.max_concurrent

        return plan

    async def _generate_samples(self) -> SampleResult:
        """Generate samples using configured sampling strategy."""
        # Collect all axes from all pipeline steps
        all_axes = {}
        for step in self.config.pipeline:
            all_axes.update(step.axes)

        # Create sampler
        sampler = create_sampler(
            axes=all_axes,
            sampling_config=self.config.sampling,
            n_samples=self.config.n_variants,
            random_seed=42,  # For reproducibility
        )

        # Generate samples
        return sampler.sample()

    async def _execute_pipeline_steps(
        self, samples: SampleResult, progress_callback: Optional[callable] = None
    ) -> List[StepResult]:
        """Execute all pipeline steps sequentially.

        Args:
            samples: Generated samples
            progress_callback: Optional progress callback

        Returns:
            List of step results
        """
        step_results = []
        previous_outputs = {}

        # Convert samples to contexts for template rendering
        contexts = self._samples_to_contexts(samples)

        for i, step in enumerate(self.config.pipeline):
            logger.info(f"Executing pipeline step {i+1}/{len(self.config.pipeline)}")

            # Execute step
            step_result = await self.step_executor.execute_step(
                step=step,
                step_index=i,
                contexts=contexts,
                previous_outputs=previous_outputs,
                progress_callback=progress_callback,
            )

            step_results.append(step_result)

            # Store outputs for next step
            if step_result.responses:
                previous_outputs[i] = [resp.content for resp in step_result.responses]

            # Log step completion
            logger.info(
                f"Step {i+1} completed with {step_result.success_rate:.1%} success rate"
            )

        return step_results

    def _samples_to_contexts(self, samples: SampleResult) -> List[Dict[str, Any]]:
        """Convert sample array to list of context dictionaries.

        Args:
            samples: Sample results from sampling

        Returns:
            List of context dictionaries for template rendering
        """
        contexts = []
        axis_names = samples.metadata.get("axis_names", [])

        for sample_row in samples.samples:
            context = {}
            for i, axis_name in enumerate(axis_names):
                if i < len(sample_row):
                    value = sample_row[i]

                    # Convert numpy types to Python types for Jinja2
                    if hasattr(value, "item"):  # numpy scalar
                        value = value.item()
                    elif hasattr(value, "tolist"):  # numpy array
                        value = value.tolist()

                    context[axis_name] = value

            contexts.append(context)

        return contexts

    async def _evaluate_with_oracles(
        self,
        step_results: List[StepResult],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[OracleResult]]:
        """Evaluate responses using configured oracles.

        Args:
            step_results: Results from pipeline steps
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping oracle names to result lists
        """
        oracle_results = {}

        # Get all responses from final step (or all steps if configured)
        final_step = step_results[-1] if step_results else None
        if not final_step or not final_step.responses:
            logger.warning("No responses to evaluate with oracles")
            return oracle_results

        responses = [resp.content for resp in final_step.responses]

        # Evaluate with each configured oracle
        oracles_to_evaluate = []

        if self.config.oracles.accuracy:
            oracles_to_evaluate.append(("accuracy", self.config.oracles.accuracy))

        if self.config.oracles.explainability:
            oracles_to_evaluate.append(
                ("explainability", self.config.oracles.explainability)
            )

        if self.config.oracles.confidence_calibration:
            oracles_to_evaluate.append(
                ("confidence_calibration", self.config.oracles.confidence_calibration)
            )

        if self.config.oracles.custom_oracles:
            for name, oracle_config in self.config.oracles.custom_oracles.items():
                oracles_to_evaluate.append((name, oracle_config))

        for oracle_name, oracle_config in oracles_to_evaluate:
            logger.info(f"Evaluating with oracle: {oracle_name}")

            try:
                # Create oracle instance
                oracle = self._create_oracle(oracle_config)
                if oracle:
                    await oracle.initialize()

                    # Evaluate all responses
                    results = []
                    for i, response in enumerate(responses):
                        context = {
                            "prompt": (
                                final_step.prompts[i]
                                if i < len(final_step.prompts)
                                else ""
                            ),
                            "response": response,
                            "oracle_name": oracle_name,
                        }

                        try:
                            result = await oracle.evaluate(response, context)
                            results.append(result)

                            if progress_callback:
                                progress_callback(1)

                        except Exception as e:
                            logger.error(
                                f"Oracle {oracle_name} evaluation failed for response {i}: {e}"
                            )

                    oracle_results[oracle_name] = results
                    await oracle.cleanup()

                    logger.info(
                        f"Oracle {oracle_name} evaluated {len(results)} responses"
                    )

            except Exception as e:
                logger.error(f"Failed to create or run oracle {oracle_name}: {e}")

        return oracle_results

    def _create_oracle(self, oracle_config) -> Optional[BaseOracle]:
        """Create oracle instance from configuration.

        Args:
            oracle_config: Oracle configuration

        Returns:
            Oracle instance or None if creation fails
        """
        try:
            if hasattr(oracle_config, "type"):
                if oracle_config.type == "llm_judge":
                    return LLMJudgeOracle(oracle_config)
                elif oracle_config.type == "embedding_similarity":
                    return EmbeddingSimilarityOracle(oracle_config)

            logger.warning(
                f"Unknown oracle type: {getattr(oracle_config, 'type', 'unknown')}"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to create oracle: {e}")
            return None

    async def _perform_bayesian_analysis(
        self, oracle_results: Dict[str, List[OracleResult]]
    ) -> Optional[Any]:
        """Perform Bayesian statistical analysis on oracle results.

        Args:
            oracle_results: Results from oracle evaluations

        Returns:
            Bayesian analysis results or None if analysis fails
        """
        if not oracle_results:
            logger.warning("No oracle results available for Bayesian analysis")
            return None

        try:
            # Use statistical config if available, otherwise use defaults
            statistical_config = self.config.statistical_config or StatisticalConfig()

            # Create multi-oracle analyzer
            analyzer = MultiOracleAnalyzer(statistical_config)

            # Perform analysis
            results = analyzer.analyze_multiple_oracles(
                oracle_results_dict=oracle_results,
                compute_joint=True,
                compute_correlations=True,
            )

            logger.info(
                f"Bayesian analysis completed for {len(oracle_results)} oracles"
            )
            return results

        except Exception as e:
            logger.error(f"Bayesian analysis failed: {e}")
            return None
