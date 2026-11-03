"""Rollback of a training step whose MoE ragged buffers overflowed."""

import jax
import jax.numpy as jnp


def _rollback_on_overflow(new_state, old_leaves, has_moe_overflow):
  if not isinstance(has_moe_overflow, jax.core.Tracer) and not bool(has_moe_overflow):
    return new_state

  def _select(new, old):
    dtype = getattr(new, "dtype", None)
    if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.extended):
      return new
    return jnp.where(has_moe_overflow, old, new)

  new_leaves, treedef = jax.tree_util.tree_flatten(new_state)
  if len(new_leaves) != len(old_leaves):
    raise ValueError(
        "moe_dropless_fallback='step': cannot roll back the step -- the updated state has "
        f"{len(new_leaves)} leaves but the input state had {len(old_leaves)}."
    )
  return jax.tree_util.tree_unflatten(treedef, [_select(new, old) for new, old in zip(new_leaves, old_leaves)])
