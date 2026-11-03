"""Half-split rotary positional embedding."""

import jax
import jax.numpy as jnp


def apply_rope(
    x: jax.Array,
    positions: jax.Array,
    *,
    max_timescale: float = 10_000.0,
    min_timescale: float = 1.0,
) -> jax.Array:
  if x.ndim != 4:
    raise ValueError(f"apply_rope expects x of shape (batch, length, heads, head_dim); got shape {x.shape}.")
  head_dim = x.shape[-1]
  if head_dim % 2:
    raise ValueError(f"apply_rope requires an even head_dim to form rotation pairs; got {head_dim}.")
  if positions.shape != x.shape[:2]:
    raise ValueError(
        f"apply_rope expects positions of shape (batch, length) = {x.shape[:2]}; got shape {positions.shape}."
    )
  half_dim = head_dim // 2
  fraction = 2 * jnp.arange(half_dim) / head_dim
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  angles = positions[:, :, jnp.newaxis, jnp.newaxis].astype(jnp.float32) / timescale
  sin = jnp.sin(angles).astype(x.dtype)
  cos = jnp.cos(angles).astype(x.dtype)
  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
