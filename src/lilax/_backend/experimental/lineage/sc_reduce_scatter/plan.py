"""Validated geometry and scaling mode of the SparseCore 0 reduce-scatter."""

import dataclasses
import math

import jax
import jax.numpy as jnp

NUM_SUBCORES = 16
AMAX_CHUNK = 16


@dataclasses.dataclass(frozen=True)
class ReduceScatterPlan:
  """Static row partitioning over the ring, dies, microbatches, subcores and VMEM tiles."""

  ring_axis: str
  core_axis: str
  ring_size: int
  num_devices: int
  num_microbatches: int
  num_tokens: int
  dim0: int
  dim1: int
  out_tokens: int
  chunk_rows: int
  mb_rows: int
  subcore_rows: int
  vmem_tile_rows: int
  tiles_per_dir: int
  num_tiles: int
  tokens_per_subcore: int
  tokens_per_tile: int
  block_tokens: int
  mb_tokens: int
  quantized_input: bool
  has_amax_input: bool
  scaled: bool
  per_hop_scaling: bool
  global_scaling: bool
  fused_per_hop: bool
  unfused_per_hop: bool

  @property
  def c2c_buf_rows(self) -> int:
    return (self.ring_size - 1) * self.num_microbatches * 2 * self.mb_rows

  @property
  def extra_names(self) -> tuple[str, ...]:
    """Names of the scaling outputs that follow the four data buffers."""
    if self.global_scaling:
      return ("amax_gather", "inv_rows")
    if self.unfused_per_hop:
      return ("amax_partner", "amax_send", "amax_recv")
    if self.fused_per_hop:
      return ("amax_send", "amax_recv")
    return ()

  def unpack_refs(self, refs):
    """Splits kernel refs into inputs, data buffers and named scaling buffers."""
    x_ref, *refs = refs
    amax_ref = refs.pop(0) if self.has_amax_input else None
    out_ref, running_sum_ref, d2d_recv_ref, c2c_recv_ref, *extra = refs
    return (
        x_ref, amax_ref, out_ref, running_sum_ref, d2d_recv_ref, c2c_recv_ref,
        dict(zip(self.extra_names, extra, strict=True)),
    )


def make_reduce_scatter_plan(
    x: jax.Array, *, has_amax_input: bool, per_hop_scaling: bool, ring_axis: str, core_axis: str,
    num_microbatches: int, vmem_tile_rows: int | None,
) -> ReduceScatterPlan:
  """Validates the reduce-scatter arguments and derives its partitioning."""
  if x.dtype not in (jnp.float8_e4m3fn, jnp.bfloat16):
    raise ValueError(f"x must have dtype float8_e4m3fn or bfloat16, got {x.dtype}.")
  quantized_input = x.dtype == jnp.float8_e4m3fn
  scaled = has_amax_input or per_hop_scaling
  if scaled and quantized_input:
    raise ValueError("token_amax or per_hop_scaling scaling requires a bfloat16 x.")
  global_scaling = has_amax_input and not per_hop_scaling
  fused_per_hop = per_hop_scaling and not has_amax_input
  unfused_per_hop = per_hop_scaling and has_amax_input
  num_tokens, dim0, dim1 = x.shape
  if dim1 != 128:
    raise ValueError(f"x trailing dimension must be 128, got {dim1}.")
  if dim0 % 8 != 0:
    raise ValueError(f"x second dimension must be divisible by 8, got {dim0}.")
  ring_size = jax.lax.axis_size(ring_axis)
  core_size = jax.lax.axis_size(core_axis)
  if ring_size < 2:
    raise ValueError(f"ring_axis '{ring_axis}' must have size >= 2, got {ring_size}.")
  if core_size != 2:
    raise ValueError(f"core_axis '{core_axis}' must have size 2, got {core_size}.")
  num_devices = ring_size * core_size
  if num_tokens % num_devices != 0:
    raise ValueError(f"num_tokens ({num_tokens}) must be divisible by ring_size * core_size" f" ({num_devices}).")
  out_tokens = num_tokens // num_devices
  chunk_rows = out_tokens * dim0
  if chunk_rows % 2 != 0:
    raise ValueError(f"chunk_rows ({chunk_rows}) must be divisible by 2.")
  half_chunk_rows = chunk_rows // 2
  if num_microbatches < 1 or half_chunk_rows % num_microbatches != 0:
    raise ValueError(
        f"half_chunk_rows ({half_chunk_rows}) must be divisible by" f" num_microbatches ({num_microbatches})."
    )
  mb_rows = half_chunk_rows // num_microbatches
  num_subcores = NUM_SUBCORES
  if mb_rows % num_subcores != 0:
    raise ValueError(f"mb_rows ({mb_rows}) must be divisible by {num_subcores} subcores.")
  subcore_rows = mb_rows // num_subcores
  tile_rows_unit = math.lcm(8, dim0) if scaled else 8
  if vmem_tile_rows is None:
    max_tile_rows = 448 if quantized_input else 224
    vmem_tile_rows = max(
        (r for r in range(tile_rows_unit, max_tile_rows + 1, tile_rows_unit) if subcore_rows % r == 0),
        default=tile_rows_unit,
    )
  if vmem_tile_rows <= 0 or vmem_tile_rows % 8 != 0:
    raise ValueError(f"vmem_tile_rows ({vmem_tile_rows}) must be a positive multiple of 8.")
  if subcore_rows % vmem_tile_rows != 0:
    raise ValueError(f"subcore_rows ({subcore_rows}) must be divisible by vmem_tile_rows" f" ({vmem_tile_rows}).")
  tiles_per_dir = subcore_rows // vmem_tile_rows
  num_tiles = 2 * tiles_per_dir
  amax_chunk = AMAX_CHUNK
  if scaled:
    if num_tokens % (num_subcores * amax_chunk) != 0:
      raise ValueError(f"num_tokens ({num_tokens}) must be divisible by" f" {num_subcores * amax_chunk} with token_amax.")
    if vmem_tile_rows % dim0 != 0:
      raise ValueError(f"vmem_tile_rows ({vmem_tile_rows}) must be divisible by d0 ({dim0})" " with token_amax.")
  tokens_per_subcore = num_tokens // num_subcores
  tokens_per_tile = vmem_tile_rows // dim0 if scaled else 0
  block_tokens = subcore_rows // dim0 if scaled else 0
  mb_tokens = mb_rows // dim0 if scaled else 0
  if per_hop_scaling and block_tokens % 8 != 0:
    raise ValueError(
        f"The tokens per subcore and microbatch block ({block_tokens}) must be" " divisible by 8 with per_hop_scaling."
    )
  return ReduceScatterPlan(
      ring_axis=ring_axis, core_axis=core_axis, ring_size=ring_size, num_devices=num_devices,
      num_microbatches=num_microbatches, num_tokens=num_tokens, dim0=dim0, dim1=dim1, out_tokens=out_tokens,
      chunk_rows=chunk_rows, mb_rows=mb_rows, subcore_rows=subcore_rows, vmem_tile_rows=vmem_tile_rows,
      tiles_per_dir=tiles_per_dir, num_tiles=num_tiles, tokens_per_subcore=tokens_per_subcore,
      tokens_per_tile=tokens_per_tile, block_tokens=block_tokens, mb_tokens=mb_tokens,
      quantized_input=quantized_input, has_amax_input=has_amax_input, scaled=scaled,
      per_hop_scaling=per_hop_scaling, global_scaling=global_scaling, fused_per_hop=fused_per_hop,
      unfused_per_hop=unfused_per_hop,
  )
