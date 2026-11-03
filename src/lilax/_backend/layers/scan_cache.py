"""Group and update KV caches for scanned decoder layers."""

import jax
import jax.numpy as jnp


def prepare_kv_caches_for_scan(kv_caches, scan_length, block_len, stack=False):
  if kv_caches is None:
    return None
  if not isinstance(kv_caches, list):
    raise TypeError(f"kv_caches must be a list, got {type(kv_caches)}")
  grouped = [tuple(kv_caches[i * block_len : (i + 1) * block_len]) for i in range(scan_length)]
  if stack:
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args, axis=0), *grouped)
  return grouped


def update_kv_caches_after_scan(kv_caches, returned_kv_cache, scan_length, block_len, stacked=False):
  if kv_caches is None or returned_kv_cache is None:
    return
  if not isinstance(kv_caches, list):
    raise TypeError(f"kv_caches must be a list, got {type(kv_caches)}")
  if stacked:
    unstacked = [jax.tree_util.tree_map(lambda x, i=i: x[i], returned_kv_cache) for i in range(scan_length)]
    for i in range(scan_length):
      start_idx = i * block_len
      for offset, updated_item in enumerate(unstacked[i]):
        kv_caches[start_idx + offset] = updated_item
  else:
    for i in range(scan_length):
      start_idx = i * block_len
      for offset, updated_item in enumerate(returned_kv_cache[i]):
        kv_caches[start_idx + offset] = updated_item
