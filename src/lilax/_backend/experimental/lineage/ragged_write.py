"""Ragged tensor write operations."""

import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard


def _ragged_write_kernel(
    dst_off_ref: Any, src_off_ref: Any, slice_size_ref: Any, dst_in_ref: Any,
    src0_ref: Any, src1_ref: Any, dst_out_ref: Any, *, block_size: int, max_src_blocks: int,
) -> None:
  """Kernel body for 2D ragged_write along axis 0."""
  b_size = jnp.int32(block_size)
  two_b_size = jnp.int32(2 * block_size)
  zero = jnp.int32(0)
  max_src_idx = jnp.int32(max_src_blocks - 1)
  block_idx = jnp.int32(pl.program_id(0))
  d_off = jnp.int32(dst_off_ref[()])
  s_off = jnp.int32(src_off_ref[()])
  s_size = jnp.int32(slice_size_ref[()])
  dst_block = (d_off // b_size) + block_idx
  dst_block_start = dst_block * b_size
  first_valid_dst_row = jnp.maximum(dst_block_start, d_off)
  first_needed_src_row = s_off + first_valid_dst_row - d_off
  src_block_0 = jnp.clip(first_needed_src_row // b_size, zero, max_src_idx).astype(jnp.int32)
  src_block_0_start = src_block_0 * b_size
  src_concat = jnp.concatenate([src0_ref[...], src1_ref[...]], axis=0)
  shift = s_off + dst_block_start - d_off - src_block_0_start
  roll_shift = ((two_b_size - shift) % two_b_size).astype(jnp.int32)
  rolled_src = pltpu.roll(src_concat, roll_shift, axis=0)[:block_size, :]
  global_dst_rows = dst_block_start + jax.lax.broadcasted_iota(jnp.int32, dst_in_ref.shape, dimension=0)
  write_mask = (global_dst_rows >= d_off) & (global_dst_rows < d_off + s_size)
  dst_out_ref[...] = jnp.where(write_mask, rolled_src, dst_in_ref[...])


def _pallas_ragged_write(
    dst: jt.Num[jax.Array, "dst_tokens embedding_dim"],
    src: jt.Num[jax.Array, "src_tokens embedding_dim"],
    dst_offset: jt.Num[jax.Array, ""] | int,
    src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "dst_tokens embedding_dim"]:
  """Pallas call wrapper for 2D ragged_write along axis 0."""
  dst_tokens, embedding_dim = dst.shape
  src_tokens, _ = src.shape
  block_size = max(8, ((block_size_tokens + 7) // 8) * 8)
  max_dst_blocks = (dst_tokens + block_size - 1) // block_size
  max_src_blocks = (src_tokens + block_size - 1) // block_size
  d_off_i32 = jnp.int32(dst_offset)
  s_size_i32 = jnp.int32(slice_size)
  first_dst_block = d_off_i32 // jnp.int32(block_size)
  last_dst_block = (d_off_i32 + s_size_i32 - jnp.int32(1)) // jnp.int32(block_size)
  num_dst_blocks = jnp.where(
      s_size_i32 > jnp.int32(0), last_dst_block - first_dst_block + jnp.int32(1), jnp.int32(0),
  ).astype(jnp.int32)

  def dst_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    del src_off_ref, slice_size_ref
    b_size = jnp.int32(block_size)
    dst_block = jnp.clip(
        (jnp.int32(dst_off_ref[()]) // b_size) + jnp.int32(block_idx),
        jnp.int32(0), jnp.int32(max_dst_blocks - 1),
    )
    return (dst_block, jnp.int32(0))

  def src0_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    del slice_size_ref
    b_size = jnp.int32(block_size)
    d_off = jnp.int32(dst_off_ref[()])
    s_off = jnp.int32(src_off_ref[()])
    dst_block_start = ((d_off // b_size) + jnp.int32(block_idx)) * b_size
    first_valid_dst_row = jnp.maximum(dst_block_start, d_off)
    first_needed_src_row = s_off + first_valid_dst_row - d_off
    src_block_0 = jnp.clip(first_needed_src_row // b_size, jnp.int32(0), jnp.int32(max_src_blocks - 1))
    return (src_block_0, jnp.int32(0))

  def src1_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref):
    src_block_0, _ = src0_index_map(block_idx, dst_off_ref, src_off_ref, slice_size_ref)
    src_block_1 = jnp.clip(src_block_0 + jnp.int32(1), jnp.int32(0), jnp.int32(max_src_blocks - 1))
    return (src_block_1, jnp.int32(0))

  block_shape = (block_size, embedding_dim)
  manual_axis_type = getattr(jax.typeof(dst), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(_ragged_write_kernel, block_size=block_size, max_src_blocks=max_src_blocks),
      out_shape=jax.ShapeDtypeStruct(dst.shape, dst.dtype, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec(block_shape, dst_index_map), pl.BlockSpec(block_shape, src0_index_map),
              pl.BlockSpec(block_shape, src1_index_map),
          ],
          out_specs=pl.BlockSpec(block_shape, dst_index_map), grid=(num_dst_blocks,),
      ),
      input_output_aliases={3: 0},
  )(dst_offset, src_offset, slice_size, dst, src, src)


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("block_size_tokens",))
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_write(
    dst: jt.Num[jax.Array, "dst_tokens embedding_dim"], src: jt.Num[jax.Array, "src_tokens embedding_dim"],
    dst_offset: jt.Num[jax.Array, ""] | int, src_offset: jt.Num[jax.Array, ""] | int,
    slice_size: jt.Num[jax.Array, ""] | int, *, block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "dst_tokens embedding_dim"]:
  """Updates a dynamic slice along axis 0 of a 2D array from a 2D source array."""
  if dst.ndim != 2 or src.ndim != 2:
    raise ValueError(f"dst and src must be 2D arrays, got ranks {dst.ndim} and {src.ndim}")
  if dst.dtype != src.dtype:
    raise ValueError(f"dst and src must have the same dtype, got {dst.dtype} and {src.dtype}")
  if dst.shape[1] != src.shape[1]:
    raise ValueError(f"dst and src embedding dimensions must match, got {dst.shape[1]} and" f" {src.shape[1]}")
  if block_size_tokens <= 0:
    raise ValueError(f"block_size_tokens must be positive, got {block_size_tokens}")
  if dst.shape[0] == 0 or src.shape[0] == 0 or dst.shape[1] == 0:
    return dst
  dst_limit = dst.shape[0]
  src_limit = src.shape[0]
  dst_offset = jnp.asarray(dst_offset)
  src_offset = jnp.asarray(src_offset)
  slice_size = jnp.asarray(slice_size)
  is_oob = (
      (slice_size <= 0) | (slice_size > dst_limit) | (slice_size > src_limit) | (dst_offset < 0)
      | (dst_offset > dst_limit - jnp.clip(slice_size, 0, dst_limit)) | (src_offset < 0)
      | (src_offset > src_limit - jnp.clip(slice_size, 0, src_limit))
  )
  zero = jnp.int32(0)
  safe_slice_size = jnp.where(is_oob, zero, jnp.asarray(slice_size, dtype=jnp.int32))
  safe_dst_offset = jnp.where(is_oob, zero, jnp.asarray(dst_offset, dtype=jnp.int32))
  safe_src_offset = jnp.where(is_oob, zero, jnp.asarray(src_offset, dtype=jnp.int32))
  return _pallas_ragged_write(dst, src, safe_dst_offset, safe_src_offset, safe_slice_size, block_size_tokens)


