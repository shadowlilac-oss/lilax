"""GRPO response-evaluation preferences independent of trainer construction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from lilax.generation.configuration import GenerationConfig


class GRPOEvaluationConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  eval_batch_size: int = -1
  sampling_strategy: str = "greedy"
  generation_configs: dict[str, GenerationConfig] | None = None
  num_eval_passes: int = 1
  response_selection: Literal["correct", "incorrect"] = "incorrect"
  return_responses: bool = False
  mode: Literal["pass", "maj", "pass_at_1"] = "pass"

  def _source_values(self):
    values = {"eval_batch_size": self.eval_batch_size, "eval_sampling_strategy": self.sampling_strategy,
              "num_eval_passes": self.num_eval_passes, "eval_corr_lst": self.response_selection == "correct",
              "eval_make_lst": self.return_responses, "eval_mode": self.mode}
    if self.generation_configs is not None:
      values["generation_configs"] = {
          name: {"eval_temperature": generation.temperature if generation.do_sample else 0.0,
                 "eval_top_k": generation.top_k, "eval_top_p": 1.0 if generation.top_p < 0 else generation.top_p}
          for name, generation in self.generation_configs.items()
      }
    return values
