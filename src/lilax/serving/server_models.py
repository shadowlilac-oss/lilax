"""Request and response models for the OpenAI-compatible API server."""

import time
import uuid
from typing import List, Optional, Union, Dict, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator


class SamplingParams(BaseModel):
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  top_p: Optional[float] = None
  top_k: Optional[int] = None
  stream: Optional[bool] = False
  stop: Optional[Union[str, List[str]]] = None
  seed: Optional[int] = None


class CompletionRequest(SamplingParams):
  model: str
  prompt: Union[str, List[str], List[int], List[List[int]]]
  echo: Optional[bool] = False
  logprobs: Optional[int] = None

  @field_validator("logprobs")
  @classmethod
  def validate_logprobs(cls, v):
    if v is not None and v < 0:
      raise ValueError("logprobs must be a non-negative integer if provided.")
    return v


class LogProbsPayload(BaseModel):
  tokens: List[str]
  token_logprobs: List[Optional[float]]
  top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None
  text_offset: List[int]


class CompletionChoice(BaseModel):
  text: str
  index: int
  logprobs: Optional[LogProbsPayload] = None
  finish_reason: str = "stop"


class Usage(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


ChoiceType = TypeVar("ChoiceType")


class BaseCompletionResponse(BaseModel, Generic[ChoiceType]):
  id: str
  object: str
  created: int = Field(default_factory=lambda: int(time.time()))
  model: str
  choices: List[ChoiceType]
  usage: Usage


class CompletionResponse(BaseCompletionResponse[CompletionChoice]):
  id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
  object: str = "text_completion"


class ChatMessage(BaseModel):
  role: str
  content: str
  reasoning_content: Optional[str] = None


class ChatCompletionRequest(SamplingParams):
  model: str
  messages: List[ChatMessage]
  logprobs: Optional[bool] = False
  top_logprobs: Optional[int] = None
  thinking_mode: Optional[str] = "thinking"


class ChatCompletionChoice(BaseModel):
  index: int
  message: ChatMessage
  finish_reason: str = "stop"
  logprobs: Optional[LogProbsPayload] = None


class ChatCompletionResponse(BaseCompletionResponse[ChatCompletionChoice]):
  id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
  object: str = "chat.completion"
