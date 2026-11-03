"""Microbatch layout for the DeepSeek FP8 batch-split schedule."""

import jax
import jax.numpy as jnp


@jax.named_scope("deepseek_batchsplit_split")
def split(x, split_factor=2):
  """Splits the input into `split_factor` parts along the batch dimension."""
  if split_factor == 1:
    return [x]
  if x is None:
    return [None] * split_factor
  else:
    x = jnp.reshape(x, (-1, split_factor) + x.shape[1:])
    return [x[:, i, ...] for i in range(split_factor)]


@jax.named_scope("deepseek_batchsplit_merge")
def merge(x, split_factor=2):
  """Merges the input microbatches back into a single tensor."""
  if split_factor == 1:
    return x[0]
  x = jnp.stack(x, axis=1)
  return jnp.reshape(x, (-1,) + x.shape[2:])
