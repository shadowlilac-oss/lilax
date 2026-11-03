"""Initialization of the autoregressive decode state."""

import functools

from flax import nnx
import jax
import jax.numpy as jnp

def compile_init_cache(cache_dict_abs, kv_cache_shardings):
  @functools.partial(jax.jit, out_shardings=(kv_cache_shardings,))
  def _init_cache():
    return (jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), cache_dict_abs),)

  return _init_cache


def cache_annotations(model):
  _, cache_state, _ = nnx.split(model, nnx.Cache, ...)

  def _logical_axes_for(var):
    meta = var.get_metadata() if hasattr(var, "get_metadata") else {}
    out = meta.get("out_sharding") or meta.get("sharding") or meta.get("sharding_names")
    if out is None:
      return ()
    return (out,) if isinstance(out, str) else tuple(out)

  annotations_state = jax.tree.map(
      _logical_axes_for,
      cache_state,
      is_leaf=lambda v: isinstance(v, nnx.Variable),
  )
  return nnx.to_pure_dict(annotations_state)


def init_decode_state(batch, vocab, cache):
  return {
      "logits": jnp.zeros((batch, 1, vocab), dtype=jnp.float32),
      "cache": cache,
      "next_pos": jnp.zeros((batch, 1), dtype=jnp.int32),
      "generated_tokens": jnp.zeros((batch, 1), dtype=jnp.int32),
      "tokens": jnp.zeros((batch, 1), dtype=jnp.int32),
      "token_logp": jnp.zeros((batch, 1), dtype=jnp.float32),
  }
