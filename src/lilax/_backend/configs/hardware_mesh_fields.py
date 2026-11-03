"""Native hardware target, device mesh, and layer scan configuration fields."""

import copy
from typing import Literal

from lilax._backend.common.common_types import CustomRule, ReorderStrategy, ShardMode
from pydantic.fields import Field
from pydantic.main import BaseModel


DEFAULT_MESH_AXES: list[str] = [
    "diloco",
    "data",
    "stage",
    "fsdp",
    "fsdp_transpose",
    "context",
    "context_usp_ulysses",
    "context_autoregressive",
    "tensor",
    "tensor_sequence",
    "expert",
    "autoregressive",
]


class HardwareAndMesh(BaseModel):
  """Configuration for hardware and parallelism mesh."""

  hardware: Literal["tpu", "gpu", "gpu_multiprocess", "cpu"] = Field("tpu", description="The type of hardware to run on.")
  num_slices: int = Field(-1, description="Number of TPU slices. Automatically determined.")
  mesh_axes: list[str] = Field(
      default_factory=lambda: copy.deepcopy(DEFAULT_MESH_AXES),
      description="The names of the axes in the logical device mesh.",
  )
  shard_mode: ShardMode = Field("auto", description="can be either auto or explicit")
  inhomogeneous_layer_cycle_interval: int = Field(1, description="The interval of repeated inhomogeneous layer patterns.")
  scan_layers: bool = Field(
      True,
      description=(
          "Whether to use jax.lax.scan over layers (stacked/unstacked checkpoint). "
          "When resuming from a checkpoint, this flag is auto-determined from metadata."
      ),
  )
  param_scan_axis: int = Field(1, description="Axis to scan over for parameters.")
  context_parallel_load_balance: bool = Field(True, description="Whether to use load balancing for context parallelism.")
  context_parallel_strategy: str = Field(
      "all_gather",
      description="Strategy for context parallelism ('all_gather', 'ring', 'ulysses', or 'usp').",
  )
  context_parallel_reorder_strategy: ReorderStrategy = Field(
      ReorderStrategy.AUTO,
      description="Reorder strategy for load-balanced context parallelism.",
  )
  custom_mesh: str = Field("", description="Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']")
  custom_mesh_and_rule: CustomRule = Field(
      CustomRule.DEFAULT,
      description="Customized mesh and logical rules for granularity.",
  )
  custom_mesh_and_rule_for_eval: CustomRule = Field(
      CustomRule.DEFAULT,
      description="Customized mesh and logical rules for evaluation.",
  )
  allow_split_physical_axes: bool = Field(False, description="Allow splitting physical axes for device mesh creation.")
  optimize_mesh_for_tpu_v6e: bool = Field(False, description="Apply transformations to the mesh for TPU v6e.")
  shardy: bool = Field(True, description="Whether to use shardy XLA backend.")
  prng_impl: Literal["unsafe_rbg", "rbg", "threefry2x32"] = Field(
      "unsafe_rbg", description="JAX default PRNG implementation used for training."
  )
  remove_size_one_mesh_axis_from_type: bool = Field(
      True,
      description="Whether to remove size one mesh axis from type through jax.config.",
  )
