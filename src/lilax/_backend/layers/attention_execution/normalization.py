"""Combines unnormalized attention computed separately over the prefill and autoregressive KV caches."""

import functools

import jax.numpy as jnp


def normalize_cudnn_attention(local_outs, local_stats):
  """Normalize across two cuDNN attentions."""
  stat0 = local_stats[0].reshape((*local_stats[0].shape, 1))
  stat1 = local_stats[1].reshape((*local_stats[1].shape, 1))
  global_stat = jnp.log(jnp.exp(stat0) + jnp.exp(stat1))
  attn_out = local_outs[0].astype(jnp.float32) * jnp.exp(stat0 - global_stat).transpose((0, 2, 1, 3)) + local_outs[
      1
  ].astype(jnp.float32) * jnp.exp(stat1 - global_stat).transpose((0, 2, 1, 3))
  return attn_out.astype(local_stats[0].dtype)


def normalize_attention(local_outs, local_maxes, local_sums):
  """Normalize across multiple localized attentions."""
  global_max = functools.reduce(jnp.maximum, local_maxes)
  global_sum = sum(
      (jnp.exp(local_max - global_max) * local_sum for (local_sum, local_max) in zip(local_sums, local_maxes))
  )
  attn_out = 0
  for local_max, local_out in zip(local_maxes, local_outs):
    local_normalizer = jnp.exp(local_max - global_max) / global_sum
    attn_out += local_normalizer * local_out
  return attn_out
