"""Cache leaf rules for inserting prefill results into decode slots."""

import jax
import jax.numpy as jnp


DEEPSEEK_V4_CACHE_KEYS = frozenset(
    (
        "entry_count",
        "accumulator_index",
        "leftover_buffer_kv",
        "leftover_buffer_gate",
        "overlap_kv",
        "overlap_gate",
    )
)


def cache_batch_index(path_key, annotations):
  """Find the batch axis used to update a cache leaf."""
  batch_idx = -1
  if "cache_batch" in annotations:
    batch_idx = annotations.index("cache_batch")
  elif "cache_scale_batch" in annotations:
    batch_idx = annotations.index("cache_scale_batch")
  if batch_idx < 0:
    raise ValueError(f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}")
  return batch_idx


def copy_cache_leaf(path, partial_cache, full_cache, annotations, slots):
  """Copy one non-packed cache leaf to one or more decode slots."""
  path_key = path[-1].key
  if path_key in [
      "cache_ar_index", "cached_ar_key", "cached_ar_value", "cached_ar_key_scale", "cached_ar_value_scale"
  ]:
    return full_cache
  batch_idx = cache_batch_index(path_key, annotations)
  if path_key in DEEPSEEK_V4_CACHE_KEYS:
    for slot in slots:
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
    return full_cache

  for slot in slots:
    if path_key == "cache_ar_segment_id":
      shape = list(full_cache.shape)
      shape[batch_idx] = 1
      zeros = jnp.zeros(tuple(shape), dtype=jnp.int32)
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
    elif path_key == "cache_prefill_segment_id":
      shape = list(full_cache.shape)
      shape[batch_idx] = 1
      zeros = jnp.zeros(tuple(shape), dtype=jnp.int32)
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
    elif path_key == "cached_ar_lengths":
      full_cache = full_cache.at[slot].set(0)
    elif path_key in [
        "cached_prefill_key", "cached_prefill_value", "cached_prefill_key_scale", "cached_prefill_value_scale"
    ]:
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
    elif path_key in ["recurrent_state", "conv_state"]:
      full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
    else:
      raise ValueError(f"We don't have a strategy for inserting {path_key}")
  return full_cache


def copy_packed_cache_leaf(path, partial_cache, full_cache, annotations, config, slot, start_idx, i, num_prompts, seq_len):
  """Copy a packed prompt's cache slice to a decode slot."""
  path_key = path[-1].key
  if path_key in [
      "cache_ar_index", "cached_ar_key", "cached_ar_value", "cached_ar_key_scale", "cached_ar_value_scale"
  ]:
    return full_cache
  batch_idx = cache_batch_index(path_key, annotations)
  if path_key in DEEPSEEK_V4_CACHE_KEYS:
    return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
  if path_key == "cache_ar_segment_id":
    zeros = jnp.zeros((1, config.max_target_length - config.max_prefill_predict_length), dtype=jnp.int32)
    return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
  elif path_key == "cache_prefill_segment_id":
    zeros = jnp.zeros((1, config.max_prefill_predict_length), dtype=jnp.int32)
    full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
    if i == num_prompts - 1:
      pad = jnp.zeros((1, seq_len), dtype=int)
      partial_cache = jnp.concatenate([partial_cache, pad], axis=1)
    partial_cache = jax.lax.dynamic_slice(partial_cache, (0, start_idx), (1, seq_len))
    partial_cache = (partial_cache == partial_cache[0, 0]).astype(int)
    full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
    return full_cache
  elif path_key == "cached_ar_lengths":
    return full_cache.at[slot].set(0)
  elif path_key in [
      "cached_prefill_key", "cached_prefill_value", "cached_prefill_key_scale", "cached_prefill_value_scale"
  ]:
    seqlen_index = config.prefill_cache_axis_order.split(",").index("1")
    start_indices = [0, 0, 0, 0]
    start_indices[seqlen_index] = start_idx
    slice_size = list(partial_cache.shape)
    slice_size[seqlen_index] = seq_len
    slice_size = tuple(slice_size)
    if i == num_prompts - 1:
      pad = jnp.zeros(slice_size, dtype=partial_cache.dtype)
      partial_cache = jnp.concatenate([partial_cache, pad], axis=seqlen_index)
    partial_cache = jax.lax.dynamic_slice(partial_cache, start_indices, slice_size)
    return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
  elif path_key in ["recurrent_state", "conv_state"]:
    if num_prompts > 1:
      raise NotImplementedError(
          "Packed prefill is currently incompatible with linear attention states (GDN). "
          "Prompt memory will bleed into adjacent prompts. Please disable packed prefill."
      )
    return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
  else:
    raise ValueError(f"We don't have a strategy for inserting {path_key}")
