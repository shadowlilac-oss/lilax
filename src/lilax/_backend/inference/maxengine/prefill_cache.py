"""Prefill cache layout transformations for MaxEngine."""

import jax
import jax.numpy as jnp


def stack_prefill_result_cache(config, cache):
  """Stack the caches across the layers."""
  if not config.stack_prefill_result_cache:
    return cache

  if config.scan_layers:
    return cache
  if "dense_layers_0" in cache["decoder"] or "moe_layers_0" in cache["decoder"]:
    first_dense = config.first_num_dense_layers
    num_moe = config.num_decoder_layers - first_dense
    layer_keys = [f"dense_layers_{i}" for i in range(first_dense)] + [f"moe_layers_{i}" for i in range(num_moe)]
  else:
    layer_keys = [f"layers_{i}" for i in range(config.num_decoder_layers)]

  layer_cache = [cache["decoder"][key] for key in layer_keys]
  stacked = jax.tree.map(lambda *c: jnp.stack(c), *layer_cache)
  return {"decoder": {"layers": stacked}}


def unstack_prefill_result_cache(config, model, cache):
  """Unstack the caches across the layers."""
  if not config.stack_prefill_result_cache:
    return cache

  if config.scan_layers:
    return cache
  stacked = cache["decoder"]["layers"]
  res_cache = {"decoder": {}}
  is_deepseek = (
      getattr(model, "is_deepseek", False)
      or (hasattr(model, "decoder") and getattr(model.decoder, "is_deepseek", False))
      or (hasattr(config, "decoder_block") and str(config.decoder_block).lower() == "deepseek")
  )
  if is_deepseek:
    first_dense = config.first_num_dense_layers
    num_moe = config.num_decoder_layers - first_dense
    layer_keys = [f"dense_layers_{i}" for i in range(first_dense)] + [f"moe_layers_{i}" for i in range(num_moe)]
  else:
    layer_keys = [f"layers_{i}" for i in range(config.num_decoder_layers)]

  for idx, key in enumerate(layer_keys):
    res_cache["decoder"][key] = jax.tree.map(lambda x, i=idx: x[i], stacked)
  return res_cache


def stacked_prefill_cache_shardings(shardings):
  decoder_dict = shardings["decoder"]
  first_key = next(k for k in decoder_dict.keys() if k.endswith("layers_0"))
  first_layer_sharding = decoder_dict[first_key]
  return {"decoder": {"layers": first_layer_sharding}}
