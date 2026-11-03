"""Decoder network for the Wan autoencoder."""

from typing import List, Optional

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from .autoencoder_cache_wan import CACHE_T, _update_cache
from .autoencoder_causal_conv_wan import WanCausalConv3d
from .autoencoder_norm_wan import WanRMS_norm
from .autoencoder_stage_wan import WanMidBlock, WanUpBlock
from ..utils_wan import get_activation


class WanDecoder3d(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int = 128, z_dim: int = 4, dim_mult: Optional[List[int]] = None,
               num_res_blocks: int = 2, attn_scales=List[float], temperal_upsample: Optional[List[bool]] = None,
               dropout=0.0, non_linearity: str = "silu", sharding: Sharding | None = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None):
    dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
    temperal_upsample = [False, True, True] if temperal_upsample is None else temperal_upsample
    self.dim, self.z_dim, self.dim_mult = dim, z_dim, dim_mult
    self.num_res_blocks, self.attn_scales = num_res_blocks, attn_scales
    self.temperal_upsample = temperal_upsample
    self.nonlinearity = get_activation(non_linearity)
    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)
    self.conv_in = WanCausalConv3d(rngs=rngs, in_channels=z_dim, out_channels=dims[0], kernel_size=3, padding=1,
                                   sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.mid_block = WanMidBlock(dim=dims[0], rngs=rngs, dropout=dropout, non_linearity=non_linearity,
                                 num_layers=1, sharding=sharding, dtype=dtype, weights_dtype=weights_dtype,
                                 precision=precision)
    self.up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      if i > 0:
        in_dim = in_dim // 2
      upsample_mode = None
      if i != len(dim_mult) - 1:
        upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
      up_block = WanUpBlock(in_dim=in_dim, out_dim=out_dim, num_res_blocks=num_res_blocks, dropout=dropout,
                            upsample_mode=upsample_mode, non_linearity=non_linearity, rngs=rngs, sharding=sharding,
                            dtype=dtype, weights_dtype=weights_dtype, precision=precision)
      self.up_blocks.append(up_block)
      if upsample_mode is not None:
        scale *= 2.0
    self.up_blocks = nnx.data(self.up_blocks)
    self.norm_out = WanRMS_norm(dim=out_dim, images=False, rngs=rngs, channel_first=False,
                                dtype=dtype, weights_dtype=weights_dtype)
    self.conv_out = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=3, kernel_size=3, padding=1,
                                    sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)
    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)
    for up_block in self.up_blocks:
      x, feat_cache, feat_idx = up_block(x, feat_cache, feat_idx)
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_out(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_out(x)
    return x, feat_cache, jnp.array(feat_idx, dtype=jnp.int32)


WanDecoder3d.__module__ = "lilax._backend.m3.models.wan.autoencoder.autoencoder_kl_wan"
