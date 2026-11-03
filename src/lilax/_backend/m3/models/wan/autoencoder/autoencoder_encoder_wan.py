"""Encoder network for the Wan autoencoder."""

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from .autoencoder_block_wan import WanAttentionBlock, WanResidualBlock
from .autoencoder_cache_wan import CACHE_T, _update_cache
from .autoencoder_causal_conv_wan import WanCausalConv3d
from .autoencoder_norm_wan import WanRMS_norm
from .autoencoder_resample_wan import WanResample
from .autoencoder_stage_wan import WanMidBlock
from ..utils_wan import get_activation


class WanEncoder3d(nnx.Module):
  def __init__(self, rngs: nnx.Rngs, dim: int = 128, z_dim: int = 4, dim_mult=None, num_res_blocks=2,
               attn_scales=None, temperal_downsample=None, dropout=0.0, non_linearity: str = "silu",
               sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None):
    dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
    attn_scales = [] if attn_scales is None else attn_scales
    temperal_downsample = [True, True, False] if temperal_downsample is None else temperal_downsample
    self.dim, self.z_dim, self.dim_mult = dim, z_dim, dim_mult
    self.num_res_blocks, self.attn_scales = num_res_blocks, attn_scales
    self.temperal_downsample = temperal_downsample
    self.nonlinearity = get_activation(non_linearity)
    dims = [dim * u for u in [1] + dim_mult]
    scale = 1.0
    self.conv_in = WanCausalConv3d(rngs=rngs, in_channels=3, out_channels=dims[0], kernel_size=3, padding=1,
                                   sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      for _ in range(num_res_blocks):
        self.down_blocks.append(WanResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout, rngs=rngs,
                                                  sharding=sharding, dtype=dtype, weights_dtype=weights_dtype,
                                                  precision=precision))
        if scale in attn_scales:
          self.down_blocks.append(WanAttentionBlock(dim=out_dim, rngs=rngs, sharding=sharding, dtype=dtype,
                                                    weights_dtype=weights_dtype, precision=precision))
        in_dim = out_dim
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.down_blocks.append(WanResample(out_dim, mode=mode, rngs=rngs, sharding=sharding, dtype=dtype,
                                            weights_dtype=weights_dtype, precision=precision))
        scale /= 2.0
    self.down_blocks = nnx.data(self.down_blocks)
    self.mid_block = WanMidBlock(dim=out_dim, rngs=rngs, dropout=dropout, non_linearity=non_linearity,
                                 num_layers=1, sharding=sharding, dtype=dtype, weights_dtype=weights_dtype,
                                 precision=precision)
    self.norm_out = WanRMS_norm(out_dim, channel_first=False, images=False, rngs=rngs,
                                dtype=dtype, weights_dtype=weights_dtype)
    self.conv_out = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=z_dim, kernel_size=3, padding=1,
                                    sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv_in(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv_in(x)
    for layer in self.down_blocks:
      x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
    x, feat_cache, feat_idx = self.mid_block(x, feat_cache, feat_idx)
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


WanEncoder3d.__module__ = "lilax._backend.m3.models.wan.autoencoder.autoencoder_kl_wan"
