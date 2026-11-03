"""Compressor KV cache state: autoregressive updates and prefill priming (DeepSeek-V4)."""


from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import Array


def update_ar_cache_and_get_validity_mask(
    kv: Array,
    gate: Array,
    cache: Any,
    model_mode: str,
    compressor_fn: Any,
    comp_dim: int,
    batch_size: int,
    mask_ndims: Optional[int] = None,
) -> Tuple[Array, Optional[Array]]:
  """Helper for autoregressive decoding: updates KV cache and computes the AR validity mask.

  Delegates token-by-token compression and cache state updates to `cache(...)`, concatenates
  the cached prefill and autoregressive blocks, and optionally computes a binary validity mask
  to prevent queries from attending to statically allocated padding slots.

  Args:
    kv: Projected KV representations. Shape: `[batch, 1, emb_dim]`.
    gate: Projected gate representations. Shape: `[batch, 1, emb_dim]`.
    cache: KV cache instance for the compressor.
    model_mode: Execution mode (`MODEL_MODE_AUTOREGRESSIVE`).
    compressor_fn: Component-specific callback function that compresses a window of tokens.
    comp_dim: Target head dimension (`head_dim` or `index_head_dim`).
    batch_size: Original unpadded batch size.
    mask_ndims: Number of dimensions for the output validity mask (e.g. 4 for `[B, 1, 1, L]`,
      3 for `[B, 1, L]`, or None to skip mask calculation).

  Returns:
    Tuple of (compressed, is_valid_mask):
      - compressed: The unpadded concatenated compressed representations. Shape: `[batch, total_len, comp_dim]`.
      - is_valid_mask: Boolean array indicating valid (non-padded) cache positions, or None if `mask_ndims` is None.
  """
  kv_exp = jnp.expand_dims(kv, 2)
  gate_exp = jnp.expand_dims(gate, 2)

  cached_prefill, cached_ar = cache(
      key=kv_exp,
      value=kv_exp,
      gate=gate_exp,
      decoder_segment_ids=None,
      model_mode=model_mode,
      compressor_fn=compressor_fn,
  )

  compressed_full = jnp.concatenate([cached_prefill[0], cached_ar[0]], axis=1)
  compressed = compressed_full[:batch_size, :, 0, :comp_dim]

  if mask_ndims is None:
    return compressed, None

  # --- AUTOREGRESSIVE VALIDITY MASK ---
  # In autoregressive mode, the total compressed KV sequence is the concatenation of:
  #   1. The prefill cache region: indices [0, max_prefill_comp)
  #   2. The AR cache region:      indices [max_prefill_comp, max_prefill_comp + ar_max_len)
  # Because both regions are statically padded to maximum capacity, we construct a binary mask
  # so queries attend only to valid, completed compression blocks:
  #   - Prefill blocks are valid if index < prefill_blocks_count (total entry count - AR valid count).
  #   - AR blocks are valid if their offset (index - max_prefill_comp) < ar_valid.
  max_prefill_comp = cached_prefill[0].shape[1]
  ar_valid = jnp.expand_dims(cached_ar[3], axis=1)  # [B, 1]
  assert cache.entry_count is not None, "cache.entry_count cannot be None for compressed attention"
  prefill_blocks_count = cache.entry_count.get_value() - ar_valid  # [B, 1]

  total_len = compressed.shape[1]
  shape_prefix = (1,) * (mask_ndims - 1) + (total_len,)
  entry_indices = jnp.arange(total_len).reshape(shape_prefix)

  if mask_ndims == 4:
    # Expand to [B, 1, 1, 1] for [B, 1, 1, L] mask
    prefill_count_exp = jnp.expand_dims(prefill_blocks_count, axis=(1, 3))
    ar_valid_exp = jnp.expand_dims(ar_valid, axis=(1, 3))
  elif mask_ndims == 3:
    # Expand to [B, 1, 1] for [B, 1, L] mask
    prefill_count_exp = jnp.expand_dims(prefill_blocks_count, axis=1)
    ar_valid_exp = jnp.expand_dims(ar_valid, axis=1)
  else:
    raise ValueError(f"Unsupported mask_ndims={mask_ndims}; expected 3, 4, or None.")

  is_prefill = entry_indices < max_prefill_comp
  is_valid_prefill = is_prefill & (entry_indices < prefill_count_exp)

  is_ar = entry_indices >= max_prefill_comp
  is_valid_ar = is_ar & ((entry_indices - max_prefill_comp) < ar_valid_exp)

  is_valid_mask = is_valid_prefill | is_valid_ar

  return compressed, is_valid_mask


def prime_prefill_cache_state(
    kv: Array,
    gate: Array,
    compressed_kv: Array,
    seq_len: int,
    usable: int,
    compress_rate: int,
    cache: Any,
    next_prior_kv: Optional[Array] = None,
    next_prior_gate: Optional[Array] = None,
    true_length: Optional[Array] = None,
) -> None:
  """Shared utility to initialize and prime KVCache state during prefill.

  Handles:
    1. Leftover token buffering when `seq_len` is not divisible by `compress_rate`.
    2. Transposition, head-dim padding, sequence-len slicing, and batch-repeating of
       compressed blocks before inserting them into `cache.cached_prefill_key`.
    3. Updating `cache.entry_count` with the initial compressed length.
    4. Writing trailing Ca window priors to `cache.overlap_kv` / `cache.overlap_gate`
       for overlapping CSA compressors.

  Args:
    kv: Projected KV representations. Shape: `[batch, seq_len, head_dim]`.
    gate: Projected gate representations. Shape: `[batch, seq_len, head_dim]`.
    compressed_kv: Prefill compressed KV blocks. Shape: `[batch, compressed_len, 1, comp_dim]`.
    seq_len: Total uncompressed sequence length.
    usable: Number of tokens divisible by `compress_rate` that were compressed.
    compress_rate: Compression rate factor.
    cache: Target KVCache instance.
    next_prior_kv: Trailing window KV prior to write to overlap cache (optional).
    next_prior_gate: Trailing window gate prior to write to overlap cache (optional).
    true_length: Optional unpadded prompt length per batch row, shape `[batch]`. When set,
      the leftover buffer, accumulator index and entry count follow the true prompt end;
      compressed blocks past it stay in the cache but are masked out by `entry_count`.
  """
  if compressed_kv.ndim == 3:
    compressed_kv = jnp.expand_dims(compressed_kv, 2)
  remainder = seq_len % compress_rate
  if true_length is not None:
    true_length = true_length.astype(jnp.int32)
    n_true = true_length // compress_rate  # [B]
    true_remainder = true_length % compress_rate  # [B]
    # Tokens [n_true * rate, true_length) seed the leftover buffer; later slots stay zero.
    pad_spec = ((0, 0), (0, compress_rate), (0, 0))
    slice_rows = jax.vmap(lambda x, s: jax.lax.dynamic_slice_in_dim(x, s, compress_rate, axis=0))
    keep = (jnp.arange(compress_rate)[None, :] < true_remainder[:, None])[:, :, None]
    leftover_kv = jnp.where(keep, slice_rows(jnp.pad(kv, pad_spec), n_true * compress_rate), 0)
    leftover_gate = jnp.where(keep, slice_rows(jnp.pad(gate, pad_spec), n_true * compress_rate), 0)
    padded_kv = jnp.expand_dims(leftover_kv, 2)
    padded_gate = jnp.expand_dims(leftover_gate, 2)
    accumulator = true_remainder[:, None]
  elif remainder > 0:
    leftover_kv = kv[:, usable:]
    leftover_gate = gate[:, usable:]
    pad_len = compress_rate - remainder
    padded_kv = jnp.expand_dims(jnp.pad(leftover_kv, ((0, 0), (0, pad_len), (0, 0))), 2)
    padded_gate = jnp.expand_dims(jnp.pad(leftover_gate, ((0, 0), (0, pad_len), (0, 0))), 2)
    accumulator = jnp.full((kv.shape[0], 1), remainder, dtype=jnp.int32)

  if true_length is not None or remainder > 0:
    assert cache.leftover_buffer_kv is not None, "leftover_buffer_kv cannot be None"
    assert cache.leftover_buffer_gate is not None, "leftover_buffer_gate cannot be None"
    assert cache.accumulator_index is not None, "accumulator_index cannot be None"
    actual_batch = cache.leftover_buffer_kv.get_value().shape[0]
    # Note: We copy/repeat the same batch across the cache batch dimension so that
    # leftover buffers are initialized for all concurrent decode slots.
    if padded_kv.shape[0] != actual_batch:
      repeats = actual_batch // padded_kv.shape[0]
      padded_kv = jnp.repeat(padded_kv, repeats, axis=0)
      padded_gate = jnp.repeat(padded_gate, repeats, axis=0)
      accumulator = jnp.repeat(accumulator, repeats, axis=0)

    cache.leftover_buffer_kv.set_value(padded_kv)
    cache.leftover_buffer_gate.set_value(padded_gate)
    cache.accumulator_index.set_value(accumulator.astype(jnp.int32))

  compressed_len = compressed_kv.shape[1]
  if compressed_len > 0:
    cache_key_var = cache.cached_prefill_key
    assert cache_key_var is not None, "cached_prefill_key cannot be None"

    # Transpose from [B, L, H, D] -> [L, H, B, D] to match the cache's physical layout
    update_blocks = jnp.transpose(compressed_kv, cache.prefill_cache_axis_order)

    operand_shape = cache_key_var.get_value().shape
    # Pad the Head Dim (axis 3) if needed (e.g. 64 -> 128)
    if update_blocks.shape[3] < operand_shape[3]:
      pad_amt = operand_shape[3] - update_blocks.shape[3]
      update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, 0), (0, pad_amt)))

    update_blocks = update_blocks[:, : operand_shape[1], ...]

    operand = cache_key_var.get_value()
    batch_axis = cache.prefill_cache_axis_order.index(0)

    # Broadcast batch=1 prefill requests to fit the max_concurrent_decodes cache
    # Note: We repeat the same batch across the cache batch dimension (when prefill batch size
    # < max_concurrent_decodes) so that the prefill state is copied to all concurrent decode slots.
    if operand.shape[batch_axis] != update_blocks.shape[batch_axis]:
      repeats = operand.shape[batch_axis] // update_blocks.shape[batch_axis]
      update_blocks = jnp.repeat(update_blocks, repeats, axis=batch_axis)

    cache_key_var.set_value(jax.lax.dynamic_update_slice_in_dim(operand, update_blocks, 0, axis=0))
    # Ensure entry_count update matches the physical cache batch size
    assert cache.entry_count is not None, "entry_count cannot be None"
    actual_batch = cache.entry_count.get_value().shape[0]
    if true_length is not None:
      entry_count = jnp.minimum(n_true, compressed_len)[:, None]
      if entry_count.shape[0] != actual_batch:
        entry_count = jnp.repeat(entry_count, actual_batch // entry_count.shape[0], axis=0)
      cache.entry_count.set_value(entry_count.astype(jnp.int32))
    else:
      cache.entry_count.set_value(jnp.full((actual_batch, 1), compressed_len, dtype=jnp.int32))

    if next_prior_kv is not None and next_prior_gate is not None:
      overlap_kv_to_write = next_prior_kv
      overlap_gate_to_write = next_prior_gate

      # Note: We copy/repeat the same batch across the cache batch dimension so that
      # overlap registers are initialized for all concurrent decode slots.
      assert cache.overlap_kv is not None, "overlap_kv cannot be None"
      assert cache.overlap_gate is not None, "overlap_gate cannot be None"
      if overlap_kv_to_write.shape[0] != cache.overlap_kv.get_value().shape[0]:
        repeats = cache.overlap_kv.get_value().shape[0] // overlap_kv_to_write.shape[0]
        overlap_kv_to_write = jnp.repeat(overlap_kv_to_write, repeats, axis=0)
        overlap_gate_to_write = jnp.repeat(overlap_gate_to_write, repeats, axis=0)

      cache.overlap_kv.set_value(overlap_kv_to_write)
      cache.overlap_gate.set_value(overlap_gate_to_write)
