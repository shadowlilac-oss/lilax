"""Middle and upsampling stages for the Wan autoencoder."""

from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from .autoencoder_block_wan import WanAttentionBlock, WanResidualBlock
from .autoencoder_resample_wan import WanResample


class WanMidBlock(nnx.Module):
  def __init__(self, dim: int, rngs: nnx.Rngs, dropout: float = 0.0, non_linearity: str = "silu",
               num_layers: int = 1, sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None):
    self.dim = dim
    resnets = [WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs, dropout=dropout, non_linearity=non_linearity,
                                sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)]
    attentions = []
    for _ in range(num_layers):
      attentions.append(WanAttentionBlock(dim=dim, rngs=rngs, sharding=sharding, dtype=dtype,
                                          weights_dtype=weights_dtype, precision=precision))
      resnets.append(WanResidualBlock(in_dim=dim, out_dim=dim, rngs=rngs, dropout=dropout,
                                      non_linearity=non_linearity, sharding=sharding, dtype=dtype,
                                      weights_dtype=weights_dtype, precision=precision))
    self.attentions = nnx.data(attentions)
    self.resnets = nnx.data(resnets)

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    x, feat_cache, feat_idx = self.resnets[0](x, feat_cache, feat_idx)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        x, feat_cache, feat_idx = attn(x, feat_cache, feat_idx)
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
    return x, feat_cache, feat_idx


class WanUpBlock(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, rngs: nnx.Rngs, dropout: float = 0.0,
               upsample_mode: Optional[str] = None, non_linearity: str = "silu", sharding: Sharding | None = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None):
    resnets = []
    current_dim = in_dim
    for _ in range(num_res_blocks + 1):
      resnets.append(WanResidualBlock(in_dim=current_dim, out_dim=out_dim, dropout=dropout,
                                      non_linearity=non_linearity, rngs=rngs, sharding=sharding, dtype=dtype,
                                      weights_dtype=weights_dtype, precision=precision))
      current_dim = out_dim
    self.resnets = nnx.data(resnets)
    self.upsamplers = nnx.data(None)
    if upsample_mode is not None:
      self.upsamplers = [WanResample(dim=out_dim, mode=upsample_mode, rngs=rngs, sharding=sharding,
                                     weights_dtype=weights_dtype, dtype=dtype, precision=precision)]

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    for resnet in self.resnets:
      x, feat_cache, feat_idx = resnet(x, feat_cache, feat_idx)
    if self.upsamplers is not None:
      x, feat_cache, feat_idx = self.upsamplers[0](x, feat_cache, feat_idx)
    return x, feat_cache, feat_idx


WanMidBlock.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"
WanUpBlock.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"
