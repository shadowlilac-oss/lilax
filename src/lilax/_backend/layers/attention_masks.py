"""Dense attention mask construction."""

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import Array, DECODING_ACTIVE_SEQUENCE_INDICATOR, DEFAULT_MASK_VALUE


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
