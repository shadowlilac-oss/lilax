"""Common recomputation and host-memory offload settings."""

from pydantic import BaseModel, ConfigDict


class RematConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  remat_policy: str = "full"
  remat_policy_for_vit: str = "minimal"
  optimizer_memory_host_offload: bool = False
  parameter_memory_host_offload: bool = False
