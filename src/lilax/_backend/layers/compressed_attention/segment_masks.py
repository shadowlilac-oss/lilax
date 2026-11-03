"""Packed-sequence segment ids and masks for compressed KV blocks (DeepSeek-V4)."""


from typing import Tuple

import jax.numpy as jnp

from lilax._backend.common.common_types import Array, DEFAULT_MASK_VALUE


def compress_segment_ids(decoder_segment_ids: Array, kv_len: int, compress_rate: int) -> Tuple[Array, Array, Array]:
  """Assigns a segment id to every compressed KV block and masks cross-segment block access.

  A block keeps its segment id only when all `compress_rate` source tokens share it; blocks
  spanning a document boundary get segment id -1 and are masked for every query.

  Args:
    decoder_segment_ids: Token segment ids. Shape: `[batch, seq_len]`.
    kv_len: Uncompressed KV sequence length.
    compress_rate: Compression rate factor.

  Returns:
    Tuple of (compressed_segment_ids, decoder_segment_ids_kv, compressed_segment_mask):
      - compressed_segment_ids: Segment id per compressed block. Shape: `[batch, num_blocks]`.
      - decoder_segment_ids_kv: Token segment ids followed by block segment ids.
      - compressed_segment_mask: Additive mask. Shape: `[batch, seq_len, num_blocks]`.
  """
  num_blocks = kv_len // compress_rate
  usable = num_blocks * compress_rate
  if decoder_segment_ids.shape[1] < usable:
    pad_seg = usable - decoder_segment_ids.shape[1]
    last_seg = decoder_segment_ids[:, -1:]
    pad_block = jnp.repeat(last_seg, pad_seg, axis=1)
    padded_seg_ids = jnp.concatenate([decoder_segment_ids, pad_block], axis=1)
  else:
    padded_seg_ids = decoder_segment_ids[:, :usable]

  chunked_segment_ids = padded_seg_ids.reshape((decoder_segment_ids.shape[0], num_blocks, compress_rate))
  min_seg = jnp.min(chunked_segment_ids, axis=-1)
  max_seg = jnp.max(chunked_segment_ids, axis=-1)
  is_valid_window = min_seg == max_seg

  compressed_segment_ids = jnp.where(is_valid_window, min_seg, -1)
  decoder_segment_ids_kv = jnp.concatenate([decoder_segment_ids, compressed_segment_ids], axis=1)

  valid_comp_seg = (decoder_segment_ids[:, :, None] == compressed_segment_ids[:, None, :]) & (
      compressed_segment_ids[:, None, :] >= 0
  )
  compressed_segment_mask = jnp.where(valid_comp_seg, 0.0, DEFAULT_MASK_VALUE)
  return compressed_segment_ids, decoder_segment_ids_kv, compressed_segment_mask
