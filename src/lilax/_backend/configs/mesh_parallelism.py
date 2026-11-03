"""Native mesh parallelism fields and validation."""

from math import prod

from pydantic.fields import Field
from pydantic.main import BaseModel


class DcnParallelism(BaseModel):
  """Parallelism dimensions across the DCN (Data Center Network)."""

  dcn_diloco_parallelism: int = Field(1, description="DCN axis for Diloco parallelism.")
  dcn_data_parallelism: int = Field(-1, description="DCN axis for data parallelism.")
  dcn_fsdp_parallelism: int = Field(1, description="DCN axis for FSDP.")
  dcn_fsdp_transpose_parallelism: int = Field(1, description="DCN axis for FSDP transpose.")
  dcn_sequence_parallelism: int = Field(1, description="DCN axis for sequence parallelism (not recommended).")
  dcn_context_parallelism: int = Field(1, description="DCN axis for context parallelism.")
  dcn_context_usp_ulysses_parallelism: int = Field(
      1, description="DCN axis for the Ulysses dimension of USP context parallelism."
  )
  dcn_context_autoregressive_parallelism: int = Field(1, description="DCN axis for context autoregressive parallelism.")
  dcn_tensor_parallelism: int = Field(1, description="DCN axis for tensor parallelism (not recommended).")
  dcn_tensor_sequence_parallelism: int = Field(
      1, description="DCN axis for tensor sequence parallelism (not recommended)."
  )
  dcn_pipeline_parallelism: int = Field(1, description="DCN axis for pipeline parallelism.")
  dcn_expert_parallelism: int = Field(1, description="DCN axis for expert parallelism.")
  dcn_autoregressive_parallelism: int = Field(1, description="DCN axis for autoregressive parallelism (not recommended).")


class IciParallelism(BaseModel):
  """Parallelism dimensions within the ICI (Inter-Chip Interconnect)."""

  ici_diloco_parallelism: int = Field(1, description="ICI axis for Diloco parallelism.")
  ici_data_parallelism: int = Field(1, description="ICI axis for data parallelism.")
  ici_fsdp_parallelism: int = Field(-1, description="ICI axis for FSDP.")
  ici_fsdp_transpose_parallelism: int = Field(1, description="ICI axis for FSDP transpose.")
  ici_sequence_parallelism: int = Field(1, description="ICI axis for sequence parallelism.")
  ici_context_parallelism: int = Field(1, description="ICI axis for context parallelism.")
  ici_context_usp_ulysses_parallelism: int = Field(
      1, description="ICI axis for the Ulysses dimension of USP context parallelism."
  )
  ici_context_autoregressive_parallelism: int = Field(1, description="ICI axis for context autoregressive parallelism.")
  ici_tensor_parallelism: int = Field(1, description="ICI axis for tensor parallelism.")
  ici_tensor_sequence_parallelism: int = Field(1, description="ICI axis for tensor sequence parallelism.")
  ici_autoregressive_parallelism: int = Field(1, description="ICI axis for autoregressive parallelism.")
  ici_pipeline_parallelism: int = Field(1, description="ICI axis for pipeline parallelism.")
  ici_expert_parallelism: int = Field(1, description="ICI axis for expert parallelism.")


def _resolve_parallelism(configured: list[int] | None, axis_map: dict[str, int], mesh_axes: list[str]) -> list[int]:
  """Per-axis parallelism list for `mesh_axes` (preserves explicit list for custom physical mesh axes)."""
  if configured and len(configured) == len(mesh_axes) and any(axis not in axis_map for axis in mesh_axes):
    return list(configured)
  return [axis_map.get(axis, 1) for axis in mesh_axes]


def resolve_parallelism_lists(config):
  """Resolve `ici_parallelism` and `dcn_parallelism` from the per-axis parallelism fields."""
  ici_map = {
      "diloco": config.ici_diloco_parallelism,
      "data": config.ici_data_parallelism,
      "stage": config.ici_pipeline_parallelism,
      "fsdp": config.ici_fsdp_parallelism,
      "fsdp_transpose": config.ici_fsdp_transpose_parallelism,
      "sequence": config.ici_sequence_parallelism,
      "context": config.ici_context_parallelism,
      "context_usp_ulysses": config.ici_context_usp_ulysses_parallelism,
      "context_autoregressive": config.ici_context_autoregressive_parallelism,
      "tensor": config.ici_tensor_parallelism,
      "tensor_sequence": config.ici_tensor_sequence_parallelism,
      "model": config.ici_tensor_parallelism,
      "expert": config.ici_expert_parallelism,
      "autoregressive": config.ici_autoregressive_parallelism,
      "attn_dp": (1),  # initialized to 1, vLLM will auto calculate this value based on TP and num_kv_heads
      "attn_dp_expert": (1),  # initialized to 1, vLLM will auto calculate this value based on EP
      "dcp": (1),
      "pcp": (1),
  }
  config.ici_parallelism = _resolve_parallelism(config.ici_parallelism, ici_map, config.mesh_axes)

  dcn_map = {
      "diloco": config.dcn_diloco_parallelism,
      "data": config.dcn_data_parallelism,
      "stage": config.dcn_pipeline_parallelism,
      "fsdp": config.dcn_fsdp_parallelism,
      "fsdp_transpose": config.dcn_fsdp_transpose_parallelism,
      "sequence": config.dcn_sequence_parallelism,
      "context": config.dcn_context_parallelism,
      "context_usp_ulysses": config.dcn_context_usp_ulysses_parallelism,
      "context_autoregressive": config.dcn_context_autoregressive_parallelism,
      "tensor": config.dcn_tensor_parallelism,
      "tensor_sequence": config.dcn_tensor_sequence_parallelism,
      "model": config.dcn_tensor_parallelism,
      "expert": config.dcn_expert_parallelism,
      "autoregressive": config.dcn_autoregressive_parallelism,
      "attn_dp": (1),  # initialized to 1, vLLM will auto calculate this value based on TP and num_kv_heads
      "attn_dp_expert": (1),  # initialized to 1, vLLM will auto calculate this value based on EP
      "dcp": (1),
      "pcp": (1),
  }
  config.dcn_parallelism = _resolve_parallelism(config.dcn_parallelism, dcn_map, config.mesh_axes)


def _resolved_fsdp_size(mesh_axes: list[str], parallelism: list[int], num_devices: int) -> int:
  """Combined size of the fsdp mesh axes, resolving a `-1` the way mesh creation will.

  `maxtext_utils.fill_unspecified_mesh_axes` hands the single -1 entry whatever devices the
  other axes leave over, so config validation has to do the same to see the sizes a run will
  really get. A -1 that cannot be resolved here (an unknown device count, say) counts as 1;
  mesh creation raises on its own if it is still unresolvable there.
  """
  specified = prod(size for size in parallelism if size != -1)
  leftover = num_devices // specified if specified > 0 and num_devices > 0 and num_devices % specified == 0 else 1
  sizes = {axis: leftover if size == -1 else size for axis, size in zip(mesh_axes, parallelism)}
  return max(sizes.get("fsdp", 1), 1) * max(sizes.get("fsdp_transpose", 1), 1)


def validate_optimizer_sharding(config):
  """Reject optimizer sharding over data when the resolved mesh also uses FSDP."""
  if config.shard_optimizer_over_data:
    fsdp_size = _resolved_fsdp_size(
        config.mesh_axes, config.ici_parallelism, config.num_target_devices // max(config.num_slices, 1)
    ) * _resolved_fsdp_size(config.mesh_axes, config.dcn_parallelism, config.num_slices)
    if fsdp_size > 1:
      raise ValueError(
          "`shard_optimizer_over_data` (Zero-1) cannot be combined with FSDP: the resolved "
          f"fsdp/fsdp_transpose mesh axes have a combined size of {fsdp_size}. Set "
          "`ici_fsdp_parallelism` and `ici_fsdp_transpose_parallelism` (and their `dcn_` "
          "counterparts) to 1, or turn off `shard_optimizer_over_data`."
      )
