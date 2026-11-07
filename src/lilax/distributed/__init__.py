"""Common parallelism controls; physical placement belongs to jaxcoord profiles."""

from pydantic import BaseModel, ConfigDict

from lilax.configuration.domains.execution_mesh import ShardMode


class MeshConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  shard_mode: ShardMode = ShardMode.AUTO
  context_parallel_strategy: str = "all_gather"


class ShardingConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  shard_optimizer_over_data: bool = False
  save_fsdp_gathered_weights: bool = False
  prefetch_fsdp_weights: bool = False
  prefetch_fsdp_defer_grad_reduce_scatter: bool = False


class ParallelismConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  mesh: MeshConfig | None = None
  sharding: ShardingConfig | None = None


class DiLoCoConfig(BaseModel):
  model_config = ConfigDict(extra="forbid")
  enable_diloco: bool = False
  diloco_sync_period: int = 36
  diloco_outer_lr: float = 0.3
  diloco_outer_momentum: float = 0.9


__all__ = ["MeshConfig", "ShardingConfig", "ParallelismConfig", "DiLoCoConfig", "ShardMode"]
