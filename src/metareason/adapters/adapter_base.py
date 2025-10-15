from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class AdapterException(Exception):
    """Base exception for adapter-related errors."""

    pass


class AdapterRequest(BaseModel):
    model: str
    system_prompt: Optional[str] = None
    user_prompt: str
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_tokens: int = Field(gt=0)


class AdapterResponse(BaseModel):
    response_text: str


class AdapterBase(ABC):
    def __init__(self):
        self._init()

    @abstractmethod
    def _init(self): ...

    @abstractmethod
    def send_request(self, request: AdapterRequest) -> AdapterResponse: ...
