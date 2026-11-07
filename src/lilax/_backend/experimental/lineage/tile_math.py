"""Elementwise tile math shared by Pallas kernel bodies."""

import jax
import jax.numpy as jnp
import jaxtyping as jt


def silu_mul_tile(x: jax.Array, y: jax.Array, c: jax.Array, out_dtype: jt.DTypeLike) -> jax.Array:
  """Computes silu(x) * y * c in the output dtype."""
  return (jax.nn.silu(x) * y * c).astype(out_dtype)


def quantize_tile(x: jax.Array, inv_scale: jax.Array, qtype: jt.DTypeLike) -> jax.Array:
  """Quantizes a tile using its inverse scale."""
  q_max = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -q_max, q_max).astype(qtype)
