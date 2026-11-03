"""RMS normalization for the DeepSeek BF16 batch-split schedule."""

import jax
import jax.numpy as jnp


def rms_norm(x, scale, *, epsilon, dtype, out_sharding=None):
  """RMS normalization."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
  return jnp.einsum("i...k,...k->i...k", y, scale, out_sharding=out_sharding)
