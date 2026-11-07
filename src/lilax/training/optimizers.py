"""Focused optimizer and schedule configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LRSchedulerConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  schedule_steps: int = -1
  final_fraction: float = 0.1
  wsd_decay_steps_fraction: float = 0.1
  wsd_decay_style: Literal["linear", "cosine"] = "linear"


class MuonConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  implementation: Literal["optax_muon", "maxtext_muon"] = "optax_muon"
  beta: float = 0.95
  weight_decay: float = 0.0
  consistent_rms: float | None = None
  include_routers: bool = True
  use_all_to_all: bool = False


class SpikeSkipConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enabled: bool = False
  interval: int = 128
  scaling_factor: float = 6.0


class OptimizerConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  adam_eps_root: float = 0.0
  adamw_mask: list[str] | None = None
  mu_dtype: str = ""
  muon: MuonConfig | None = None
  spike_skip: SpikeSkipConfig | None = None
  trainable_parameters: list[str] | None = None
