"""Routing metadata of the TensorCore ragged gather-reduce and its shared layout."""

import dataclasses

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

RAGGED_GATHER_REDUCE_TC_ALIGN = 128
_RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK = 256


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RaggedGatherReduceMetadata:
  """Routing of ragged_gather_reduce_tc."""

  src_rows: jax.Array
  tokens: jax.Array
  blocks: jax.Array
  block_size: int = jax.tree.static()


def ragged_gather_reduce_tc_layout(num_indices: int, top_k: int, block_size: int) -> tuple[int, int, int]:
  """Returns maximum blocks, window and slots length for the metadata layout."""
  align = RAGGED_GATHER_REDUCE_TC_ALIGN
  max_blocks = pl.cdiv(num_indices + num_indices // top_k, block_size - top_k)
  window = pl.cdiv(block_size, align) * align + align
  slots_len = pl.cdiv(max(num_indices, window), align) * align
  return max_blocks, window, slots_len


def check_ragged_gather_reduce_tc_args(top_k: int, block_size: int):
  if top_k <= 0:
    raise ValueError(f"top_k must be positive, got {top_k}.")
  if block_size <= top_k:
    raise ValueError(
        f"block_size must exceed top_k ({top_k}) so that a block fits a token" f" and its rows, got {block_size}."
    )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_gather_reduce_tc_metadata(
    indices: jt.Int[jax.Array, "num_indices"], start: jt.Int[jax.Array, ""] | int,
    num_tokens: jt.Int[jax.Array, ""] | int, *, top_k: int, block_size: int = 896,
    max_num_tokens: int | None = None,
) -> RaggedGatherReduceMetadata:
  """Derives routing consumed by ragged_gather_reduce_tc."""
  check_ragged_gather_reduce_tc_args(top_k, block_size)
  num_indices = indices.shape[0]
  if num_indices % top_k:
    raise ValueError(f"indices length ({num_indices}) must be divisible by top_k ({top_k}).")
  if isinstance(num_tokens, int) and num_tokens < 0:
    raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
  if isinstance(start, int) and start < 0:
    raise ValueError(f"start must be non-negative, got {start}.")
  if isinstance(start, int) and isinstance(num_tokens, int) and start + num_tokens > num_indices:
    raise ValueError(f"start + num_tokens ({start + num_tokens}) must not exceed indices" f" length ({num_indices}).")
  num_out_tokens = num_indices // top_k
  max_blocks, _, slots_len = ragged_gather_reduce_tc_layout(num_indices, top_k, block_size)
  if num_out_tokens == 0:
    empty_slots = jnp.zeros((slots_len,), jnp.int32)
    return RaggedGatherReduceMetadata(
        src_rows=empty_slots, tokens=empty_slots, blocks=jnp.zeros((1,), jnp.int32), block_size=block_size,
    )
  start = jnp.asarray(start, dtype=jnp.int32)
  num_tokens = jnp.asarray(num_tokens, dtype=jnp.int32)
  positions = jnp.arange(num_indices, dtype=jnp.int32)
  window = num_indices
  if max_num_tokens is not None:
    window = min(max(max_num_tokens, 1), num_indices)
  window_start = jnp.minimum(start, num_indices - window)
  window_positions = window_start + jnp.arange(window, dtype=jnp.int32)
  rel = window_positions - start
  valid = (rel >= 0) & (rel < num_tokens)
  window_indices = jax.lax.dynamic_slice_in_dim(indices, window_start, window)
  key = jnp.where(valid, window_indices, num_out_tokens).astype(jnp.int32)
  sorted_key, sorted_positions = jax.lax.sort((key, window_positions), num_keys=1, is_stable=True)
  if window < num_indices:
    invalid_positions = jnp.where(positions < start + num_tokens, positions - num_tokens, positions)
    sorted_positions = jnp.where(
        positions < num_tokens, jnp.pad(sorted_positions, (0, num_indices - window)), invalid_positions,
    )
    sorted_key = jnp.pad(sorted_key, (0, num_indices - window), constant_values=num_out_tokens)
  pad = slots_len - num_indices
  src_rows = jnp.pad(sorted_positions - start, (0, pad))
  tokens = jnp.pad(sorted_key, (0, pad))
  padded_key = jnp.pad(sorted_key, (top_k - 1, 0), constant_values=-1)
  run_offset = jnp.zeros_like(sorted_key)
  for k in range(1, top_k):
    run_offset = run_offset + (padded_key[top_k - 1 - k : top_k - 1 - k + num_indices] == sorted_key).astype(jnp.int32)
  offset_bits = (top_k - 1).bit_length()
  offset_mask = (1 << offset_bits) - 1
  int_max = jnp.iinfo(jnp.int32).max
  cost = (positions - run_offset) + sorted_key
  packed = jnp.where(sorted_key < num_out_tokens, (cost << offset_bits) + run_offset, int_max)
  chunk = _RAGGED_GATHER_REDUCE_TC_SEARCH_CHUNK
  num_chunks = pl.cdiv(num_indices, chunk)
  packed_chunks = jnp.pad(packed, (0, num_chunks * chunk - num_indices), constant_values=int_max).reshape(
      num_chunks, chunk
  )
  end_packed = packed_chunks[:, -1]
  threshold = jnp.arange(max_blocks + 1, dtype=jnp.int32) * (block_size - top_k)
  thr_packed = threshold[:, None] << offset_bits
  end_below = end_packed < thr_packed
  full_chunks = jnp.sum(end_below, axis=1, dtype=jnp.int32)
  row = jnp.minimum(full_chunks, num_chunks - 1)
  sel_packed = packed_chunks[row]
  sel_below = sel_packed < thr_packed
  in_chunk = jnp.sum(sel_below, axis=1, dtype=jnp.int32)
  slot_start = row * chunk + in_chunk
  prev_packed = jnp.maximum(
      jnp.max(jnp.where(end_below, end_packed, -1), axis=1), jnp.max(jnp.where(sel_below, sel_packed, -1), axis=1),
  )
  prev_key = jnp.where(
      slot_start > 0, (prev_packed >> offset_bits) + (prev_packed & offset_mask) - (slot_start - 1), -1,
  )
  next_packed = jnp.min(jnp.where(sel_below, int_max, sel_packed), axis=1)
  next_key = jnp.where(next_packed < int_max, (next_packed >> offset_bits) - slot_start, num_out_tokens)
  token_start = jnp.clip(threshold - slot_start, prev_key + 1, next_key)
  num_blocks = jnp.sum(token_start[:-1] < num_out_tokens, dtype=jnp.int32)
  blocks = jnp.concatenate([num_blocks[None], slot_start[:-1], jnp.diff(slot_start), token_start[:-1], jnp.diff(token_start)])
  return RaggedGatherReduceMetadata(src_rows=src_rows, tokens=tokens, blocks=blocks.astype(jnp.int32), block_size=block_size)
