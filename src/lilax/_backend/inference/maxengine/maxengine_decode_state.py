"""Initialization of the autoregressive decode state."""

import functools

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE


def init_decode_state(engine):
  """Initialize the decode cache, its logical axes, and token state."""
  batch = int(engine.config.per_device_batch_size * engine.mesh.size)
  vocab = engine.config.vocab_size
  cache_dict_abs = engine._nnx_init_cache_dict(mode=MODEL_MODE_AUTOREGRESSIVE)

  if engine._compiled_init_cache_fn is None:

    @functools.partial(jax.jit, out_shardings=(engine.kv_cache_shardings,))
    def _init_cache():
      return (jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), cache_dict_abs),)

    engine._compiled_init_cache_fn = _init_cache

  (cache,) = engine._compiled_init_cache_fn()
  _, cache_state, _ = nnx.split(engine.model_ar, nnx.Cache, ...)

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
  engine.kv_cache_annotations_named = nnx.to_pure_dict(annotations_state)

  return {
      "logits": jnp.zeros((batch, 1, vocab), dtype=jnp.float32),
      "cache": cache,
      "next_pos": jnp.zeros((batch, 1), dtype=jnp.int32),
      "generated_tokens": jnp.zeros((batch, 1), dtype=jnp.int32),
      "tokens": jnp.zeros((batch, 1), dtype=jnp.int32),
      "token_logp": jnp.zeros((batch, 1), dtype=jnp.float32),
  }
