"""Insert prefill results into autoregressive decode-state slots."""

import jax
import jax.numpy as jnp

from lilax._backend.inference.maxengine import insertion as maxengine_insertion


def insert_slots_cache(prefix_cache, decode_cache, cache_annotations, slots):
  """Copy a prefill cache into the given decode slots of the decode cache."""

  def copy(path, partial_cache, full_cache, annotations):
    return maxengine_insertion.copy_cache_leaf(path, partial_cache, full_cache, annotations, slots)

  return jax.tree_util.tree_map_with_path(
      copy,
      prefix_cache,
      decode_cache,
      cache_annotations,
  )


def insert_slot_fields(unboxed_prefix, decode_state, slot):
  """Insert one prefill result's non-cache fields into one decode slot."""
  inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
  inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
      decode_state["next_pos"], jnp.asarray(unboxed_prefix["next_pos"], dtype=decode_state["next_pos"].dtype), slot, 0
  )
  inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
      decode_state["generated_tokens"],
      unboxed_prefix["generated_tokens"],
      slot,
      0,
  )
  inserted_tokens = jax.lax.dynamic_update_index_in_dim(decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)
  inserted_token_logp = jax.lax.dynamic_update_index_in_dim(
      decode_state["token_logp"], unboxed_prefix["token_logp"], slot, 0
  )
  return inserted_logits, inserted_next_pos, inserted_generated_tokens, inserted_tokens, inserted_token_logp


def bulk_insert_slot_fields(unboxed_prefix, decode_state, slots):
  """Insert one prefill result's non-cache fields into several decode slots of `decode_state` in place."""
  for i, slot in enumerate(slots):
    decode_state["logits"] = jax.lax.dynamic_update_index_in_dim(
        decode_state["logits"], unboxed_prefix["logits"], slot, 0
    )
    decode_state["next_pos"] = jax.lax.dynamic_update_index_in_dim(
        decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0
    )
    decode_state["generated_tokens"] = jax.lax.dynamic_update_index_in_dim(
        decode_state["generated_tokens"],
        jnp.expand_dims(unboxed_prefix["generated_tokens"][i], axis=0),
        slot,
        0,
    )
    decode_state["tokens"] = jax.lax.dynamic_update_index_in_dim(
        decode_state["tokens"],
        jnp.expand_dims(unboxed_prefix["tokens"][i], axis=0),
        slot,
        0,
    )
    decode_state["token_logp"] = jax.lax.dynamic_update_index_in_dim(
        decode_state["token_logp"],
        jnp.expand_dims(unboxed_prefix["token_logp"][i], axis=0),
        slot,
        0,
    )
