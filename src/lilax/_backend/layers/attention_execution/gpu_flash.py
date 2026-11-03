"""Runs causal GPU flash attention through the Tokamax Pallas Triton kernel."""

from lilax._backend.common.common_types import Array
from tokamax._src.ops.attention import base as tokamax_attention_base
from tokamax._src.ops.attention import pallas_triton as tokamax_pallas_triton


def gpu_flash_attention(query: Array, key: Array, value: Array, decoder_segment_ids: Array | None) -> Array:
  """Causal attention, additionally restricted to matching segments when segment ids are given."""
  mask = tokamax_attention_base.Mask(is_causal=True)
  if decoder_segment_ids is not None:
    seg_mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
    mask = mask & tokamax_attention_base.Mask(seg_mask[:, None, :, :])
  gpu_flash_attn = tokamax_pallas_triton.PallasTritonFlashAttention()
  return gpu_flash_attn(query, key, value, logits_scale=1.0, mask=mask)
