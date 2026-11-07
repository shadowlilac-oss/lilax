"""Spatial and temporal resampling layers for the Wan autoencoder."""

from typing import Tuple, Union

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.m3.core.sharding import Sharding
from lilax._backend.observability import max_logging
from .autoencoder_cache_wan import CACHE_T, RepSentinel, _update_cache
from .autoencoder_causal_conv_wan import WanCausalConv3d
from ..utils_wan import param_metadata


class WanUpsample(nnx.Module):
  def __init__(self, scale_factor: Tuple[float, float], method: str = "nearest"):
    self.scale_factor = scale_factor
    self.method = method

  def __call__(self, x: jax.Array) -> jax.Array:
    input_dtype = x.dtype
    in_shape = x.shape
    assert len(in_shape) == 4, "This module only takes tensors with shape of 4."
    n, h, w, c = in_shape
    target_h = int(h * self.scale_factor[0])
    target_w = int(w * self.scale_factor[1])
    if (self.method == "nearest" and self.scale_factor[0] == int(self.scale_factor[0])
        and self.scale_factor[1] == int(self.scale_factor[1])):
      scale_h = int(self.scale_factor[0])
      scale_w = int(self.scale_factor[1])
      out = jnp.repeat(jnp.repeat(x, scale_h, axis=1), scale_w, axis=2)
    else:
      if self.method == "nearest":
        max_logging.log("Warning: WanUpsample2D nearest method requested but scale_factor"
                        f" {self.scale_factor} is not integer. Falling back to jax.image.resize.")
      out = jax.image.resize(x.astype(jnp.float32), (n, target_h, target_w, c), method=self.method)
      out = out.astype(input_dtype)
    return out


class Identity(nnx.Module):
  def __call__(self, x):
    return x


class ZeroPaddedConv2D(nnx.Module):
  def __init__(self, dim: int, rngs: nnx.Rngs, kernel_size: Union[int, Tuple[int, int, int]],
               stride: Union[int, Tuple[int, int, int]] = 1, sharding: Sharding | None = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None):
    rank = len(kernel_size) if isinstance(kernel_size, (tuple, list)) else 2
    kernel_sharding = (None,) * (rank + 2)
    self.conv = nnx.Conv(dim, dim, kernel_size=kernel_size, strides=stride, use_bias=True, rngs=rngs,
                         kernel_init=nnx.initializers.xavier_uniform(),
                         kernel_metadata=param_metadata(sharding, "downsample_conv_kernel", kernel_sharding),
                         dtype=dtype, param_dtype=weights_dtype, precision=precision)

  def __call__(self, x):
    return self.conv(x)


class WanResample(nnx.Module):
  def __init__(self, dim: int, mode: str, rngs: nnx.Rngs, sharding: Sharding | None = None,
               dtype: jnp.dtype = jnp.float32, weights_dtype: jnp.dtype = jnp.float32,
               precision: jax.lax.Precision = None):
    self.dim = dim
    self.mode = mode
    self.dtype = dtype
    self.time_conv = nnx.data(None)
    if mode in ("upsample2d", "upsample3d"):
      self.resample = nnx.Sequential(
          WanUpsample(scale_factor=(2.0, 2.0), method="nearest"),
          nnx.Conv(dim, dim // 2, kernel_size=(3, 3), padding="SAME", use_bias=True, rngs=rngs,
                   kernel_init=nnx.initializers.xavier_uniform(),
                   kernel_metadata=param_metadata(sharding, "upsample_conv_kernel", (None, None, None, "conv_out")),
                   dtype=dtype, param_dtype=weights_dtype, precision=precision),
      )
      if mode == "upsample3d":
        self.time_conv = WanCausalConv3d(rngs=rngs, in_channels=dim, out_channels=dim * 2,
                                         kernel_size=(3, 1, 1), padding=(1, 0, 0), sharding=sharding,
                                         dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    elif mode in ("downsample2d", "downsample3d"):
      self.resample = ZeroPaddedConv2D(dim=dim, rngs=rngs, kernel_size=(3, 3), stride=(2, 2), sharding=sharding,
                                      dtype=dtype, weights_dtype=weights_dtype, precision=precision)
      if mode == "downsample3d":
        self.time_conv = WanCausalConv3d(rngs=rngs, in_channels=dim, out_channels=dim, kernel_size=(3, 1, 1),
                                         stride=(2, 1, 1), padding=(0, 0, 0), sharding=sharding,
                                         dtype=dtype, weights_dtype=weights_dtype, precision=precision)
    else:
      self.resample = Identity()

  def __call__(self, x: jax.Array, feat_cache=None, feat_idx=0):
    b, t, h, w, c = x.shape
    assert c == self.dim
    x = x.astype(self.dtype)
    if self.mode == "upsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, RepSentinel())
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -CACHE_T:, :, :, :])
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and not isinstance(feat_cache[idx], RepSentinel):
            cache_x = jnp.concatenate([jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1), cache_x], axis=1)
          if cache_x.shape[1] < 2 and feat_cache[idx] is not None and isinstance(feat_cache[idx], RepSentinel):
            cache_x = jnp.concatenate([jnp.zeros(cache_x.shape, dtype=cache_x.dtype), cache_x], axis=1)
          if isinstance(feat_cache[idx], RepSentinel):
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1
          x = x.reshape(b, t, h, w, 2, c)
          x = x.transpose(0, 1, 4, 2, 3, 5)
          x = x.reshape(b, t * 2, h, w, c)
    t = x.shape[1]
    x = x.reshape(b * t, h, w, c)
    x = self.resample(x)
    h_new, w_new, c_new = x.shape[1:]
    x = x.reshape(b, t, h_new, w_new, c_new)
    if self.mode == "downsample3d":
      if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
          feat_cache = _update_cache(feat_cache, idx, jnp.copy(x))
          feat_idx += 1
        else:
          cache_x = jnp.copy(x[:, -1:, :, :, :])
          x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
          feat_cache = _update_cache(feat_cache, idx, cache_x)
          feat_idx += 1
    return x, feat_cache, feat_idx


