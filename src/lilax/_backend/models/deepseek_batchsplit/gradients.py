"""Per-microbatch gradient accumulation for the DeepSeek BF16 batch-split schedule."""

import functools

import jax
import jax.numpy as jnp


def sum_grads(grads):
  return functools.reduce(lambda x, y: jax.tree_util.tree_map(jnp.add, x, y), grads)
