"""Per-step gradient and MoE routing-bias diagnostics."""

import jax
import jax.numpy as jnp


def step_diagnostics(raw_grads, has_moe_overflow, bias_values, bias_updates):
  leaves = [
      grad for grad in jax.tree_util.tree_leaves(raw_grads)
      if hasattr(grad, "dtype") and jnp.issubdtype(grad.dtype, jnp.inexact)
  ]
  max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(grad)).astype(jnp.float32) for grad in leaves])) if leaves else jnp.float32(0)
  checksum = jnp.float32(0)
  for bias in bias_values:
    checksum = checksum + jnp.sum(jnp.asarray(bias, jnp.float32))
  nonzero = jnp.float32(0)
  for update in bias_updates:
    nonzero = nonzero + jnp.sum(jnp.asarray(update) != 0).astype(jnp.float32)
  overflow = jnp.asarray(False if has_moe_overflow is None else has_moe_overflow).astype(jnp.float32)
  return {
      "diag/max_abs_grad": max_abs,
      "diag/moe_bias_checksum": checksum,
      "diag/moe_bias_update_nonzero": nonzero,
      "diag/moe_overflow": overflow,
  }
