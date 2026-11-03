"""Selects the static Splash Attention mask for an attention type."""

from lilax._backend.common.common_types import AttentionType
from lilax._backend.layers.attention_masks.splash_masks import (
    BlockCausalMask,
    ChunkedCausalMask,
    HCAStaticMask,
    LoadBalancedBlockCausalMask,
    LoadBalancedCausalMask,
    LoadBalancedChunkedCausalMask,
    LoadBalancedLocalMask,
)


def make_static_splash_mask(
    attention_type: AttentionType,
    mask_shape: tuple[int, int],
    mask_module,
    use_tokamax_splash: bool,
    cp_size: int,
    splash_q_seq_shards: int,
    use_load_balanced_cp: bool,
    sliding_window_size: int | None,
    chunk_attn_window_size: int | None,
    causal_block_size: int | None,
    compress_ratio: int,
    indexer_mask,
    orig_q_len: int,
    orig_kv_len: int,
    pad_kv_total: int,
):
  """Builds the single-head static Splash mask, or None when CSA supplies a runtime `indexer_mask`.

  `mask_module` is the Splash mask library (Tokamax or JAX) providing the built-in mask types.
  """
  if attention_type == AttentionType.FULL:
    mask = mask_module.FullMask(mask_shape)
  elif attention_type == AttentionType.COMPRESSED:
    if not use_tokamax_splash:
      raise ValueError("AttentionType.COMPRESSED flash attention requires use_tokamax_splash=True.")
    if compress_ratio is None or compress_ratio <= 0:
      raise ValueError("compress_ratio must be provided for AttentionType.COMPRESSED flash attention.")
    if compress_ratio > 4:
      local_kv_len = orig_q_len
      compressed_kv_len = max(0, orig_kv_len - orig_q_len - pad_kv_total)
      mask = HCAStaticMask(
          shape=mask_shape,
          local_kv_len=local_kv_len,
          compressed_kv_len=compressed_kv_len,
          pad_kv_total=pad_kv_total,
          compress_ratio=compress_ratio,
          local_window=sliding_window_size,
          shard_count=splash_q_seq_shards,
      )
    elif compress_ratio == 4:
      if indexer_mask is None:
        raise ValueError("indexer_mask must be provided for CSA (compress_ratio == 4) dynamic splash attention.")
      mask = None
    else:
      raise ValueError(
          f"Unsupported compress_ratio={compress_ratio} for AttentionType.COMPRESSED. "
          "Expected 4 for CSA (requires indexer_mask) or >4 for HCA (requires HCAStaticMask)."
      )
  elif attention_type == AttentionType.BLOCK_DIFFUSION:
    mask_type = LoadBalancedBlockCausalMask if use_load_balanced_cp else BlockCausalMask
    mask_kwargs = {"cp_size": cp_size} if use_load_balanced_cp else {}
    mask = mask_type(
        shape=mask_shape,
        causal_block_size=causal_block_size,
        **mask_kwargs,
    )
  else:
    mask = mask_module.CausalMask(shape=mask_shape)

  if use_load_balanced_cp and attention_type not in (
      AttentionType.FULL,
      AttentionType.BLOCK_DIFFUSION,
      AttentionType.COMPRESSED,
  ):
    mask = LoadBalancedCausalMask(shape=mask_shape, cp_size=cp_size)

  # Apply local masking if local sliding attention is enabled.
  if attention_type == AttentionType.LOCAL_SLIDING:
    if sliding_window_size is None:
      raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
    local_window_size = (sliding_window_size - 1, sliding_window_size)
    if use_load_balanced_cp:
      mask &= LoadBalancedLocalMask(
          shape=mask_shape,
          window_size=local_window_size,
          offset=0,
          cp_size=cp_size,
      )
    else:
      mask &= mask_module.LocalMask(
          shape=mask_shape,
          window_size=local_window_size,
          offset=0,
      )
  elif attention_type == AttentionType.CHUNK:
    if chunk_attn_window_size is None:
      raise ValueError("chunk_attn_window_size must be set for chunk attention type")

    if use_load_balanced_cp:
      mask &= LoadBalancedChunkedCausalMask(
          shape=mask_shape,
          chunk_size=chunk_attn_window_size,
          cp_size=cp_size,
      )
    else:
      mask &= ChunkedCausalMask(
          shape=mask_shape,
          chunk_size=chunk_attn_window_size,
      )
  return mask
