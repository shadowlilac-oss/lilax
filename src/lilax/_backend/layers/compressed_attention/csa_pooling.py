"""Compressed Sparse Attention (CSA) overlapping window pooling (DeepSeek-V4)."""


import enum
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import Array


class CSAPoolingConfig(enum.IntEnum):
  """Configuration constants for Compressed Sparse Attention (CSA) overlap pooling.

  Attributes:
    OVERLAP_WINDOWS: Number of overlapping prior windows (W) carried forward from cache.
  """

  OVERLAP_WINDOWS = 1


def csa_overlap_pooling(
    chunk_kv: Array,
    chunk_gate: Array,
    kv_norm: Any,
    head_dim: int,
    prior_kv: Optional[Array] = None,
    prior_gate: Optional[Array] = None,
    is_same_doc: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
  """Computes overlapping window pooling for Compressed Sparse Attention (CSA).

  DeepSeek-V4 CSA uses a stride-4, window-8 pooling mechanism where each output block
  aggregates representations over a 2m (8-token) window formed by pairing the trailing
  m (4) tokens of the previous window (Ca) with the leading m (4) tokens of the current
  window (Cb).

  Pipeline:
    1. Split: `[batch, n_windows, compress_rate, 2 * head_dim]` -> 2x `[batch, n_windows, compress_rate, head_dim]`
    2. Shift: Ca shifted forward by one window (prepending cache prior if available).
    3. Concat (Ca + Cb): -> `[batch, n_windows, 2 * compress_rate, head_dim]`
    4. Gating & Sum: -> `[batch, n_windows, head_dim]`

  Args:
    chunk_kv: Input KV projection chunks. Shape: `[batch, n_windows, compress_rate, 2 * head_dim]`.
    chunk_gate: Input gate projection chunks. Shape: `[batch, n_windows, compress_rate, 2 * head_dim]`.
    kv_norm: RMSNorm instance applied to the aggregated representations.
    head_dim: Target head dimension.
    prior_kv: Previous window KV prior from cache (optional).
    prior_gate: Previous window gate prior from cache (optional).
    is_same_doc: Boolean tensor of shape [batch, n_windows] indicating if window i
      belongs to the same document as previous window i - 1. Without sequence packing,
      this is always True. With packing, this is False at document boundaries, causing
      the shifted prior window (Ca) to be masked out to prevent cross-document leakage.

  Returns:
    Tuple of (compressed, next_prior_kv, next_prior_gate):
      - compressed: The pooled overlapping states. Shape: `[batch, n_windows, head_dim]`.
      - next_prior_kv: Updated KV prior for the next window.
      - next_prior_gate: Updated gate prior for the next window.

  Shape Transformations:
    1. Split: `[batch, n_windows, compress_rate, 2 * head_dim]` -> 2x `[batch, n_windows, compress_rate, head_dim]`
    2. Shift: Ca shifted forward by one window (prepending cache prior if available).
    3. Concat (Ca + Cb): -> `[batch, n_windows, 2 * compress_rate, head_dim]`
    4. Gating & Sum: -> `[batch, n_windows, head_dim]`
  """
  # D2 is 2 * head_dim
  B, _, C, D2 = chunk_kv.shape
  # w is the number of overlapping windows carried forward across chunk boundaries (W=1)
  w = int(CSAPoolingConfig.OVERLAP_WINDOWS)
  expected_size = B * w * C * D2

  # 1. Split the projections into Ca and Cb components for overlapping
  # -> 2x [batch, n_windows, compress_rate, head_dim]
  a_kv, b_kv = jnp.split(chunk_kv, 2, axis=-1)
  a_gate, b_gate = jnp.split(chunk_gate, 2, axis=-1)

  # 2. Safely handle cache priors (using OVERLAP_WINDOWS for the prior window count w)
  if prior_kv is None or prior_kv.size != expected_size:
    prior_a_kv = jnp.zeros((B, w, C, head_dim), dtype=a_kv.dtype)
    # Note: Empty gate prior must be -inf so softmax(gate) = 0
    prior_a_gate = jnp.full((B, w, C, head_dim), -jnp.inf, dtype=a_gate.dtype)
  else:
    prior_kv = prior_kv.reshape((B, w, C, D2))
    prior_gate = prior_gate.reshape((B, w, C, D2))
    prior_a_kv, _ = jnp.split(prior_kv, 2, axis=-1)
    prior_a_gate, _ = jnp.split(prior_gate, 2, axis=-1)

    # KVCache initializes with zeros. If it's the very first step,
    # gate must be -inf for softmax to equal 0.
    is_empty = jnp.all(prior_a_kv == 0)
    prior_a_gate = jnp.where(is_empty, -jnp.inf, prior_a_gate)

  # 3. Shift Ca forward by one window
  # We prepend the prior window to the current chunks, and drop the last window of Ca
  a_kv_shifted = jnp.concatenate([prior_a_kv, a_kv[:, :-1]], axis=1)
  a_gate_shifted = jnp.concatenate([prior_a_gate, a_gate[:, :-1]], axis=1)

  if is_same_doc is not None:
    is_same_doc_exp = is_same_doc[:, :, None, None]
    a_kv_shifted = jnp.where(is_same_doc_exp, a_kv_shifted, 0.0)
    a_gate_shifted = jnp.where(is_same_doc_exp, a_gate_shifted, -jnp.inf)

  # 4. Concatenate shifted Ca and unshifted Cb to form the 2m overlapping window
  # -> [batch, n_windows, 2 * compress_rate, head_dim]
  new_kv = jnp.concatenate([a_kv_shifted, b_kv], axis=2)
  new_gate = jnp.concatenate([a_gate_shifted, b_gate], axis=2)

  # 5. Apply softmax gating and sum across the overlapping window dimension
  gate_weights = jax.nn.softmax(new_gate, axis=2).astype(new_kv.dtype)
  compressed = jnp.sum(new_kv * gate_weights, axis=2)

  # 6. Apply the projection norm
  if kv_norm is not None:
    compressed = kv_norm(compressed)

  # 7. Extract the next priors (Keep the full D2 so it fits the cache)
  # We grab the full last window of the current chunk to pass into the next chunk
  next_prior_kv = chunk_kv[:, -1:, :, :]
  next_prior_gate = chunk_gate[:, -1:, :, :]

  return compressed, next_prior_kv, next_prior_gate


def compute_csa_prefill_chunk_pooling(
    kv: Array,
    gate: Array,
    seq_len: int,
    batch_size: int,
    compress_rate: int,
    position_ids: Array,
    position_bias: Array,
    kv_norm: Any,
    rotary_emb: Any,
    head_dim: int,
    dtype: Any,
    cache: Optional[Any] = None,
    true_length: Optional[Array] = None,
) -> Tuple[Array, int, Optional[Array], Optional[Array], int]:
  """Helper for CSA prefill chunking, overlap pooling, and RoPE embedding.

  Args:
    kv: Projected KV representations.
    gate: Projected gate representations.
    seq_len: Total sequence length.
    batch_size: Batch size.
    compress_rate: Compression rate.
    position_ids: Absolute token positions.
    position_bias: Position bias tensor.
    kv_norm: Normalization layer.
    rotary_emb: Rotary embedding layer.
    head_dim: Target head dimension (`head_dim` or `index_head_dim`).
    dtype: Computation dtype.
    cache: Optional KVCache instance.
    true_length: Optional unpadded prompt length per batch row, shape `[batch]`. When set,
      the returned overlap prior is the last window made only of real tokens, so decode
      continues from the true prompt end rather than from the padded prefill end.

  Returns:
    Tuple of (compressed, compressed_len, next_prior_kv, next_prior_gate, usable).
  """
  usable = (seq_len // compress_rate) * compress_rate
  chunk_kv = kv[:, :usable]
  chunk_gate = gate[:, :usable]

  if cache is not None:
    assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
    assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
    prior_kv = cache.overlap_kv.get_value()
    prior_gate = cache.overlap_gate.get_value()
  else:
    prior_kv, prior_gate = None, None

  if chunk_kv.shape[1] > 0:
    n_windows = chunk_kv.shape[1] // compress_rate
    chunk_kv_reshaped = chunk_kv.reshape((batch_size, n_windows, compress_rate, -1))
    chunk_gate_reshaped = chunk_gate.reshape((batch_size, n_windows, compress_rate, -1)) + position_bias

    block_positions = position_ids[:, :usable:compress_rate]
    prior_block_positions = jnp.concatenate([block_positions[:, 0:1] - compress_rate, block_positions[:, :-1]], axis=1)
    is_same_doc = block_positions == (prior_block_positions + compress_rate)

    compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
        chunk_kv_reshaped, chunk_gate_reshaped, kv_norm, head_dim, prior_kv, prior_gate, is_same_doc=is_same_doc
    )
    compressed_len = compressed.shape[1]
    positions = position_ids[:, :usable:compress_rate]
    compressed = rotary_emb(compressed, positions, unsqueeze_dim=None)

    if true_length is not None:
      n_true = (true_length // compress_rate).astype(jnp.int32)  # [B]
      last = jnp.maximum(n_true - 1, 0)[:, None, None, None]
      true_prior_kv = jnp.take_along_axis(chunk_kv_reshaped, last, axis=1)
      true_prior_gate = jnp.take_along_axis(chunk_gate_reshaped, last, axis=1)
      if prior_kv is not None and prior_kv.size == true_prior_kv.size:
        fallback_kv = prior_kv.reshape(true_prior_kv.shape).astype(true_prior_kv.dtype)
        fallback_gate = prior_gate.reshape(true_prior_gate.shape).astype(true_prior_gate.dtype)
      else:
        fallback_kv = jnp.zeros_like(true_prior_kv)
        fallback_gate = jnp.zeros_like(true_prior_gate)
      has_window = (n_true > 0)[:, None, None, None]
      next_prior_kv = jnp.where(has_window, true_prior_kv, fallback_kv)
      next_prior_gate = jnp.where(has_window, true_prior_gate, fallback_gate)
  else:
    compressed = jnp.zeros((batch_size, 0, head_dim), dtype=dtype)
    compressed_len = 0
    next_prior_kv = prior_kv
    next_prior_gate = prior_gate

  return compressed, compressed_len, next_prior_kv, next_prior_gate, usable
