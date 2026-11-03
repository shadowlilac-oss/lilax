"""Residual and attention blocks for the Wan autoencoder."""

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from .autoencoder_cache_wan import CACHE_T, _update_cache
from .autoencoder_causal_conv_wan import WanCausalConv3d
from .autoencoder_norm_wan import WanRMS_norm
from .autoencoder_resample_wan import Identity
from ..utils_wan import get_activation, param_metadata


class WanResidualBlock(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs, dropout: float = 0.0,
               non_linearity: str = "silu", sharding: Sharding | None = None, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32, precision: jax.lax.Precision = None):
    self.nonlinearity = get_activation(non_linearity)
    self.norm1 = WanRMS_norm(dim=in_dim, rngs=rngs, images=False, channel_first=False,
                             dtype=dtype, weights_dtype=weights_dtype)
    self.conv1 = WanCausalConv3d(rngs=rngs, in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1,
                                 sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.norm2 = WanRMS_norm(dim=out_dim, rngs=rngs, images=False, channel_first=False,
                             dtype=dtype, weights_dtype=weights_dtype)
    self.conv2 = WanCausalConv3d(rngs=rngs, in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1,
                                 sharding=sharding, dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    self.conv_shortcut = (WanCausalConv3d(rngs=rngs, in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                          sharding=sharding, dtype=dtype, weights_dtype=weights_dtype,
                                          precision=precision) if in_dim != out_dim else Identity())

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    h = self.conv_shortcut(x)
    x = self.norm1(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv1(x, feat_cache[idx], idx)
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    if feat_cache is not None:
      idx = feat_idx
      cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
      if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
      x = self.conv2(x, feat_cache[idx])
      feat_cache = _update_cache(feat_cache, idx, cache_x)
      feat_idx += 1
    else:
      x = self.conv2(x)
    x = x + h
    return x, feat_cache, feat_idx


class WanAttentionBlock(nnx.Module):
  def __init__(self, dim: int, rngs: nnx.Rngs, sharding: Sharding | None = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None):
    self.dim = dim
    self.norm = WanRMS_norm(rngs=rngs, dim=dim, channel_first=False, dtype=dtype, weights_dtype=weights_dtype)
    self.to_qkv = nnx.Conv(in_features=dim, out_features=dim * 3, kernel_size=(1, 1), rngs=rngs,
                           kernel_init=nnx.initializers.xavier_uniform(),
                           kernel_metadata=param_metadata(sharding, "attn_qkv_kernel", (None, None, None, "conv_out")),
                           dtype=dtype, param_dtype=weights_dtype, precision=precision)
    self.proj = nnx.Conv(in_features=dim, out_features=dim, kernel_size=(1, 1), rngs=rngs,
                         kernel_init=nnx.initializers.xavier_uniform(),
                         kernel_metadata=param_metadata(sharding, "attn_proj_kernel", (None, None, "conv_in", None)),
                         dtype=dtype, param_dtype=weights_dtype, precision=precision)

  @jax.named_scope("WanVAEAttentionBlock")
  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    identity = x
    batch_size, time, height, width, channels = x.shape
    x = x.reshape(batch_size * time, height, width, channels)
    x = self.norm(x)
    qkv = self.to_qkv(x)
    qkv = qkv.reshape(batch_size * time, height * width, 1, channels * 3)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    x = jax.nn.dot_product_attention(q, k, v)
    x = x.reshape(batch_size * time, height, width, channels)
    x = self.proj(x)
    x = x.reshape(batch_size, time, height, width, channels)
    return x + identity, feat_cache, feat_idx


WanResidualBlock.__module__ = "lilax._backend.m3.models.wan.autoencoder.autoencoder_kl_wan"
WanAttentionBlock.__module__ = "lilax._backend.m3.models.wan.autoencoder.autoencoder_kl_wan"
