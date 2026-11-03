"""Dense attention mask construction."""

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import (
    Array,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DEFAULT_MASK_VALUE,
    MODEL_MODE_AUTOREGRESSIVE,
)


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true
  and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the
  mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this
  was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from
  https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the function
      documentation.

  Returns:
    Masked logits.
  """
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


def combine_compressed_attention_masks(uncompressed_mask, compressed_mask, output_mask, s_len, pad_kv_total):
  """Combines sliding, compressed-memory, and segment masks."""
  max_ndim = max(
      uncompressed_mask.ndim,
      compressed_mask.ndim if compressed_mask is not None else 0,
      output_mask.ndim if output_mask is not None else 0,
      4,
  )

  def _align_mask(mask, target_ndim):
    if mask is None or mask.ndim >= target_ndim:
      return mask
    return mask.reshape(mask.shape[:1] + (1,) * (target_ndim - mask.ndim) + mask.shape[1:])

  expanded_uncompressed_mask = _align_mask(uncompressed_mask, max_ndim)
  if compressed_mask is None:
    if output_mask is not None:
      output_mask = _align_mask(output_mask, max_ndim)
      output_mask = expanded_uncompressed_mask * output_mask
    else:
      output_mask = expanded_uncompressed_mask
    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE)

  compressed_mask = _align_mask(compressed_mask, max_ndim)
  target_shape = compressed_mask.shape[:-1] + (s_len,)
  expanded_uncompressed_mask = jnp.broadcast_to(expanded_uncompressed_mask, target_shape)
  if pad_kv_total > 0 and compressed_mask is not None:
    pad_width = [(0, 0)] * (compressed_mask.ndim - 1) + [(pad_kv_total, 0)]
    compressed_mask = jnp.pad(compressed_mask, pad_width, constant_values=DEFAULT_MASK_VALUE)

  if output_mask is not None:
    output_mask_aligned = _align_mask(output_mask, max_ndim)
    expanded_uncompressed_mask = expanded_uncompressed_mask & output_mask_aligned[..., :s_len]
    if output_mask_aligned.shape[-1] > s_len:
      comp_seg_mask = output_mask_aligned[..., s_len : s_len + compressed_mask.shape[-1]]
      compressed_mask = jnp.where(comp_seg_mask, compressed_mask, DEFAULT_MASK_VALUE)

  expanded_uncompressed_mask = jnp.where(expanded_uncompressed_mask, 0.0, DEFAULT_MASK_VALUE).astype(
      compressed_mask.dtype
  )
  return jnp.concatenate([expanded_uncompressed_mask, compressed_mask], axis=-1)


def _decode_cache_positions(segment_positions, decoder_segment_ids, s_len, kv_cache_part):
  """Absolute query and key positions for a single-token decode step over one KV cache.

  Prefill cache slot i holds position i. The AR cache is written in order from
  the first decode step, so its newest valid slot holds the query token.

  Returns:
    (abs_q [batch, 1, 1], abs_k [batch, 1, s_len]), or None when the cache part,
    the query position or the AR cache segment ids are unknown.
  """
  if kv_cache_part is None or segment_positions is None:
    return None
  abs_q = segment_positions[:, :, None]
  slots = jnp.arange(s_len)[None, None, :]
  if kv_cache_part == "prefill":
    return abs_q, jnp.broadcast_to(slots, abs_q.shape[:2] + (s_len,))
  if decoder_segment_ids is None:
    return None
  valid = decoder_segment_ids[:, :s_len] == DECODING_ACTIVE_SEQUENCE_INDICATOR
  newest = jnp.max(jnp.where(valid, jnp.arange(s_len)[None, :], -1), axis=-1)[:, None, None]
  return abs_q, abs_q - newest + slots


def local_sliding_attention_mask(
    q_seq_len, kv_seq_len, next_pos, sliding_window_size, use_segment_positions, position_row_ids,
    position_col_ids, segment_positions, model_mode, decoder_segment_ids, kv_cache_part,
):
  """Boolean sliding-window mask over the full KV sequence for AttentionType.LOCAL_SLIDING.

  Load-balanced context parallelism measures the window between `position_row_ids` and
  `position_col_ids`; single-token decode measures it between true cache positions when they are known.
  """
  if use_segment_positions:
    row_ids_sliding = position_row_ids
    col_ids_sliding = position_col_ids
  else:
    row_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, 1), 0) + next_pos
    col_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (1, kv_seq_len), 1)
  sliding_mask = (col_ids_sliding > (row_ids_sliding - sliding_window_size)) & (col_ids_sliding <= row_ids_sliding)
  if use_segment_positions:
    sliding_mask = sliding_mask[:, None, None, :, :]
  if model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
    decode_positions = _decode_cache_positions(segment_positions, decoder_segment_ids, kv_seq_len, kv_cache_part)
    if decode_positions is not None:
      distance = decode_positions[0] - decode_positions[1]
      sliding_mask = ((distance >= 0) & (distance < sliding_window_size))[:, None, None, :, :]
  return sliding_mask


def compressed_sliding_window_mask(
    q_seq_len, s_len, next_pos, sliding_window_size, use_segment_positions, position_row_ids,
    position_col_ids, segment_positions, model_mode, decoder_segment_ids, kv_cache_part,
):
  """Constructs the causal and sliding window boolean mask for uncompressed tokens.

  Handles three distinct operational modes:
    Case 1 (Autoregressive single-token decoding):
      Queries arrive one token at a time (q_seq_len == 1). In the KV cache,
      active cached tokens are located at indices 0..max_valid. We map cached
      indices to logical positions relative to abs_q, and enforce the causal
      sliding window: 0 <= (abs_q - abs_k) < sliding_window_size.
    Case 2 (Load-balanced Context Parallelism):
      Tokens are distributed non-contiguously across TPU devices.
      `segment_positions` contains the original global token indices for each
      local slice, so row/col coordinates are derived from `position_row_ids`
      and `position_col_ids`.
    Case 3 (Training & standard prefill on packed sequences):
      In packed datasets (e.g. C4), multiple documents are concatenated into
      a single contiguous sequence buffer [0..L-1].
      - We compute causal distance using contiguous global buffer coordinates:
          row_ids = [0..Q-1] + next_pos,  col_ids = [0..S-1]
          distance = row_ids - col_ids
          in_window = (0 <= distance < sliding_window_size)
      - IMPORTANT: We do NOT use per-document `segment_positions` (which reset
        to 0 at each document boundary) for row_ids or col_ids here. If per-doc
        positions were used with buffer indices, subtraction across packed
        document boundaries would produce negative values (e.g. pos_q - col_k < 0),
        erroneously masking out valid intra-document tokens and causing training
        loss explosions. Document boundary isolation is handled strictly and
        independently by `decoder_segment_ids`.
  """
  if model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
    # --- CASE 1: Single-token Autoregressive Decoding ---
    decode_positions = _decode_cache_positions(segment_positions, decoder_segment_ids, s_len, kv_cache_part)
    if decode_positions is not None:
      distance = decode_positions[0] - decode_positions[1]
      in_window = (distance < sliding_window_size) if sliding_window_size is not None else True
      return in_window & (distance >= 0)
    # Safely use segment_positions if provided, or fall back to next_pos
    if segment_positions is not None:
      abs_q = segment_positions[:, :, None]
    else:
      local_next = next_pos[:, None] if isinstance(next_pos, jax.Array) else next_pos
      abs_q = jnp.arange(q_seq_len)[None, :, None] + local_next

    if decoder_segment_ids is not None:
      # Find the highest valid slot index in the KV cache for each sequence in the batch
      is_valid = decoder_segment_ids[:, :s_len] == DECODING_ACTIVE_SEQUENCE_INDICATOR
      valid_indices = jnp.where(is_valid, jnp.arange(s_len)[None, :], -1)
      max_valid = jnp.max(valid_indices, axis=-1, keepdims=True)  # [batch, 1]

      # Determine whether each sequence in the batch is actively decoding (has valid cache slots)
      is_ar_cache = max_valid[:, :, None] >= 0  # [batch, 1, 1]

      # Active decode: align cache slot indices `i` so slot `max_valid` corresponds to `abs_q`
      i = jnp.arange(s_len)[None, None, :]
      abs_k_ar = abs_q - max_valid[:, :, None] + i
      abs_k_prefill = jnp.broadcast_to(i, abs_k_ar.shape)

      abs_k = jnp.where(is_ar_cache, abs_k_ar, abs_k_prefill)
    else:
      abs_k = jnp.arange(s_len)[None, None, :]

    distance = abs_q - abs_k
    in_window = (distance < sliding_window_size) if sliding_window_size is not None else True
    return in_window & (distance >= 0)

  # --- CASE 2 & 3: Prefill and Training Phases (q_seq_len > 1) ---
  if use_segment_positions:
    # Case 2: Context Parallelism — use global token positions from CP sharding
    row_ids = position_row_ids
    col_ids = position_col_ids[:, :, :s_len]
  else:
    # Case 3: Standard Training / Prefill on Packed Sequences.
    # Use contiguous global buffer coordinates so causal/sliding distance is valid
    # across all packed documents. Cross-document masking is handled by decoder_segment_ids.
    local_next = next_pos[:, None] if isinstance(next_pos, jax.Array) else next_pos
    row_ids = jnp.arange(q_seq_len)[None, :, None] + local_next
    col_ids = jnp.arange(s_len)[None, None, :]

  distance = row_ids - col_ids
  in_window = (distance < sliding_window_size) if sliding_window_size is not None else True
  return in_window & (distance >= 0)


def _generate_chunk_attention_mask(mask_shape: tuple[int, int], chunk_size: int, q_offset: int = 0) -> jax.Array:
  """Generates an explicit boolean mask for chunked causal attention.

  This function computes the full boolean mask array where True indicates
  attention is allowed based on chunked causal rules (tokens attend only
  within the same chunk, and causally within that chunk).

  Args:
    mask_shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """
  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_offset
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  same_chunk = (row_ids // chunk_size) == (col_ids // chunk_size)
  chunk_mask = same_chunk & (row_ids >= col_ids)
  return chunk_mask


def _generate_block_causal_attention_mask(
    mask_shape: tuple[int, int], causal_block_size: int, q_offset: int = 0
) -> jax.Array:
  """Generates a block-causal mask for Block Diffusion Language Models (BD3LMs).

  Implements the block-causal attention pattern (M_BC) described in Arriola et
  al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion
  Language Models" (https://arxiv.org/abs/2503.09573).

  For block size B = causal_block_size, query index q, and key index k:

  * Within block i (floor(q/B) == floor(k/B)), tokens attend bidirectionally.
  * Across blocks (floor(q/B) > floor(k/B)), block i attends causally to every
    preceding block, but not to future blocks.

  B = 1 reduces to standard causal attention, while B equal to the sequence
  length reduces to full bidirectional attention.
  """
  if causal_block_size <= 0:
    raise ValueError("causal_block_size must be positive")

  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_offset
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  return (row_ids // causal_block_size) >= (col_ids // causal_block_size)


def _make_block_mask_indices(bidirectional_mask):
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _make_bidirectional_block_mask(bidirectional_mask):
  """Creates bidirectional block mask from bidirectional_mask, where True corresponds to image tokens.

  bidirectional_mask shape: [B, L] bidirectional_block_mask shape: [B, L, L]
  Examples: bidirectional_mask = [[0, 1, 1, 1, 0, 0]] bidirectional_block_mask =
  [[

      [False, False, False, False, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
  ]]
  """
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  bidirectional_block_mask = (kv_block_indices[:, None, :] == q_block_indices[..., None]) & (
      q_block_indices[..., None] > 0
  )
  return bidirectional_block_mask
