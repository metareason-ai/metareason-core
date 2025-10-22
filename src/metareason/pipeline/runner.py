import asyncio
import logging
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from ..adapters import AdapterRequest, OllamaAdapter
from ..config import SpecConfig
from ..oracles import EvaluationContext, EvaluationResult, LLMJudge
from ..sampling import LhsSampler
from . import TemplateRenderer, load_spec

logger = logging.getLogger(__name__)


class SampleResult(BaseModel):
    """Result from evaluating a single sample variant."""

    sample_params: Dict
    original_prompt: str
    final_response: str
    evaluations: Dict[str, EvaluationResult]


async def run(spec_path: Path) -> List[SampleResult]:
    """Run a complete evaluation pipeline from a spec file.

    Args:
        spec_path: Path to the YAML specification file.

    Returns:
        List of SampleResult objects containing responses and oracle evaluations.

    Raises:
        Exception: If spec loading or execution fails.
    """
    try:
        spec_config: SpecConfig = load_spec(spec_path)
        logger.info(f"Loaded spec {spec_config}")
    except Exception as e:
        logger.error(f"Failed to load spec: {e}")
        raise

    # Initialize adapter for LLM pipeline
    adapter = OllamaAdapter()

    # Initialize oracles from config
    oracles = {}
    for oracle_name, oracle_config in spec_config.oracles.items():
        if oracle_config.type == "llm_judge":
            oracles[oracle_name] = LLMJudge(oracle_config)
        else:
            logger.warning(f"Unknown oracle type: {oracle_config.type}")

    logger.info(f"Initialized {len(oracles)} oracle(s)")

    # Generate samples and process each one
    samples = LhsSampler(spec_config.axes).generate_samples(spec_config.n_variants)
    all_tasks = []

    for sample in samples:
        task = _process_sample(spec_config.pipeline, sample, adapter, oracles)
        all_tasks.append(task)

    return await asyncio.gather(*all_tasks)


async def _process_sample(pipeline, sample, adapter, oracles):
    """Process a single sample through the pipeline and evaluate with oracles.

    Args:
        pipeline: List of PipelineConfig stages to execute.
        sample: Parameter dictionary for this sample variant.
        adapter: LLM adapter for executing pipeline stages.
        oracles: Dictionary of oracle_name -> oracle instance for evaluation.

    Returns:
        SampleResult containing the prompt, response, and oracle evaluations.
    """
    renderer = TemplateRenderer()
    response = None
    original_prompt = None

    # Execute pipeline stages
    for i, pipe in enumerate(pipeline):
        if i == 0:
            # First stage: render template with sample parameters
            user_prompt = renderer.render_request(pipe.template, sample)
            original_prompt = user_prompt  # Save for oracle evaluation
        else:
            # Subsequent stages: use previous response as input
            user_prompt = response

        adapter_request = AdapterRequest(
            model=pipe.model,
            temperature=pipe.temperature,
            top_p=pipe.top_p,
            max_tokens=pipe.max_tokens,
            user_prompt=user_prompt,
        )
        adapter_response = await adapter.send_request(adapter_request)
        response = adapter_response.response_text

    # Evaluate final response with all configured oracles
    evaluations = {}
    eval_context = EvaluationContext(
        prompt=original_prompt,
        response=response,
        metadata=sample,  # Include sample parameters for context
    )

    for oracle_name, oracle in oracles.items():
        try:
            logger.info(f"Evaluating with oracle: {oracle_name}")
            eval_result = await oracle.evaluate(eval_context)
            evaluations[oracle_name] = eval_result
            logger.info(f"{oracle_name} score: {eval_result.score}")
        except Exception as e:
            logger.error(f"Oracle {oracle_name} failed: {e}")
            # Continue with other oracles even if one fails
            evaluations[oracle_name] = EvaluationResult(
                score=0.0, explanation=f"Evaluation failed: {str(e)}"
            )

    return SampleResult(
        sample_params=sample,
        original_prompt=original_prompt,
        final_response=response,
        evaluations=evaluations,
    )
