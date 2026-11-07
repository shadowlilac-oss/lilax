"""Benchmark selection and output settings independent of serving placement."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from lilax.generation.configuration import GenerationConfig


class DatasetEvalConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  benchmark: str
  num_samples: int | None = None
  max_tokens: int = 1024
  temperature: float = 0.0
  concurrency: int = 64
  request_timeout: int = 600
  max_retries: int = 3
  skip_warmup: bool = False


class HarnessEvalConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  backend: Literal["evalchemy", "lm_eval"] = "evalchemy"
  tasks: tuple[str, ...] | list[str] = ("mmlu",)
  num_fewshot: int = 0
  num_samples: int | float | None = None
  apply_chat_template: bool = False
  fewshot_as_multiturn: bool = False
  generation_config: GenerationConfig | None = None
  skip_warmup: bool = False


class SimpleEvalsConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  tasks: tuple[Literal["mmlu", "gpqa", "gsm8k", "drop", "mgsm", "mgsm_en", "aime2024", "aime2025"], ...] = ("mmlu",)
  num_samples: int | None = None
  n_repeats: int | None = None
  max_tokens: int = 2048
  temperature: float = 0.5
  system_message: str | None = "You are a helpful assistant."
  reasoning_effort: str | None = None
  log_debug_info: bool = False
  continue_on_request_error: bool = False
  skip_warmup: bool = False


class EvalOutputConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  results_path: str
  gcs_results_path: str | None = None
