"""Dot product for the DeepSeek BF16 batch-split schedule."""

import jax.numpy as jnp


def dot(x, y, axes=1):
  return jnp.tensordot(x, y, axes=axes)
