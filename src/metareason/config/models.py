from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AxisConfig(BaseModel):
    name: str
    type: Literal["categorical", "continuous"]
    values: List[Any] = []
    weights: List[float] = []
    distribution: Optional[Literal["uniform", "normal", "truncnorm", "beta"]] = None
    params: Dict[str, float] = {}


class PipelineConfig(BaseModel):
    template: str
    adapter: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1)
    max_tokens: int


class SamplingConfig(BaseModel):
    method: Literal["latin_hypercube"]
    optimization: Literal["maximin"]
    random_seed: Optional[int] = None


class OracleConfig(BaseModel):
    type: Literal["llm_judge"]
    model: str
    adapter: str
    max_tokens: int = 2000
    temperature: Optional[int] = 1
    rubric: Optional[str] = None


class SpecConfig(BaseModel):
    spec_id: str
    pipeline: List[PipelineConfig] = Field(..., min_length=1)
    sampling: SamplingConfig
    n_variants: int = 1
    oracles: Dict[str, OracleConfig] = Field(..., min_length=1)
    axes: List[AxisConfig] = []
