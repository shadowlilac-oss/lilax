"""Derivation of model dimension scales from the global parameter scale."""

import math


def get_individual_scales(scale: int) -> tuple[int, int, int, int]:
  """Choose appropriate scales for individual dimensions based on global scale."""
  if scale == 0:
    raise ValueError("Global parameter scale cannot be zero.")
  log_2_scale = math.floor(math.log2(scale))
  if 2**log_2_scale != scale:
    raise ValueError(
        "Global parameter scale should be a power of two. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads, "
        "base_mlp_dim, base_num_decoder_layers and/or head_dim."
    )
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale
