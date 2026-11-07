"""Training objectives and batch-size rampup."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ObjectiveConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  training_objective: Literal["causal_lm", "block_diffusion"] = "causal_lm"
  block_diffusion_mask_id: int = -1
  block_diffusion_min_noise: float = 1e-3
  block_diffusion_logit_alignment: Literal["same_position", "shifted"] = "same_position"
  block_diffusion_canvas_policy: Literal["all_masked", "seed_and_mask"] = "all_masked"


class BatchRampupConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enable_rampup_batch_size: bool = False
  per_device_batch_size_start: float = 4.0
  per_device_batch_size_increment: float = 2.0
  global_rampup_samples: int = 500
