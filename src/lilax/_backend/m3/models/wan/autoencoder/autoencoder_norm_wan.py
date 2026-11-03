"""Normalization layer for the Wan autoencoder."""

from flax import nnx
import jax
import jax.numpy as jnp


class WanRMS_norm(nnx.Module):
  def __init__(self, dim: int, rngs: nnx.Rngs, channel_first: bool = True, images: bool = True,
               eps: float = 1e-6, use_bias: bool = False, dtype: jnp.dtype = jnp.float32,
               weights_dtype: jnp.dtype = jnp.float32):
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)
    self.eps = eps
    self.channel_first = channel_first
    self.scale = dim**0.5
    self.dtype = dtype
    self.gamma = nnx.Param(jnp.ones(shape, dtype=weights_dtype))
    self.bias = nnx.Param(jnp.zeros(shape, dtype=weights_dtype)) if use_bias else 0

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.astype(self.dtype)
    normalized = jnp.linalg.norm(x, ord=2, axis=(1 if self.channel_first else -1), keepdims=True)
    normalized = x / jnp.maximum(normalized, self.eps)
    normalized = normalized * self.scale * self.gamma
    if self.bias:
      return (normalized + self.bias.value).astype(self.dtype)
    return normalized.astype(self.dtype)


