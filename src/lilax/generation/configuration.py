"""Generation preferences and the existing continuous batching controls."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class GenerationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  max_new_tokens: int | None = None
  do_sample: bool = False
  top_k: int = 0
  top_p: float = -1.0
  temperature: float = 1.0
  eos_token_id: int | list[int] | None = None

  def sampling_options(self):
    algorithm = "greedy"
    if self.do_sample:
      algorithm = ("composite" if self.top_k > 0 and self.top_p > 0 else "topk" if self.top_k > 0
                   else "nucleus" if self.top_p >= 0 else "weighted")
    return {"algorithm": algorithm, "topk": self.top_k, "nucleus_topp": self.top_p,
            "temperature": self.temperature}


class ContinuousBatchingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enable_batch_prefill: bool = False
  min_decode_steps: int = 10
  prefill_lengths: list[int] | Literal["auto"] = "auto"
  batch_prefill_max_batch_size: int = 16
  debug: bool = False
