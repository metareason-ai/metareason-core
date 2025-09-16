import asyncio
import logging
from pathlib import Path
from typing import List

from ..adapters import AdapterRequest, AdapterResponse, OllamaAdapter
from ..config import SpecConfig
from ..sampling import LhsSampler
from . import TemplateRenderer, load_spec

logger = logging.getLogger(__name__)


async def run(spec_path: Path) -> List[AdapterResponse]:
    try:
        spec_config: SpecConfig = load_spec(spec_path)
        logger.info(f"Loaded spec {spec_config}")
    except Exception as e:
        logger.error(f"Failed to load spec: {e}")
        raise

    adapter = OllamaAdapter()
    all_tasks = []

    samples = LhsSampler(spec_config.axes).generate_samples(spec_config.n_variants)

    for sample in samples:
        task = _process_sample(spec_config.pipeline, sample, adapter)
        all_tasks.append(task)

    return await asyncio.gather(*all_tasks)


async def _process_sample(pipeline, sample, adapter):
    renderer = TemplateRenderer()
    response = None

    for i, pipe in enumerate(pipeline):
        if i == 0:
            user_prompt = renderer.render_request(pipe.template, sample)
        else:
            user_prompt = response

        adapter_request = AdapterRequest(
            model=pipe.model,
            temperature=pipe.temperature,
            top_p=pipe.top_p,
            max_tokens=pipe.max_tokens,
            user_prompt=user_prompt,
        )
        response = await adapter.send_request(adapter_request)

    return response
