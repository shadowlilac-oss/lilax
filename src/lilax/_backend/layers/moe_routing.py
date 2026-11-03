"""MoE routing operations."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def take_along_last_axis_dense_vjp(x: jax.Array, indices: jax.Array, num_classes: int) -> jax.Array:
  """`jnp.take_along_axis(x, indices, axis=-1)` whose backward is dense rather than a scatter-add.

  The transpose of `take_along_axis` is a scatter-add of the `[..., k]` cotangent into `[..., num_classes]`. On the
  TPU, XLA sorts the flattened indices and then runs an element-serial scatter; for DeepSeek routing
  ([tokens, 8] into [tokens, 256]) that is the largest VPU-lane op in the step. Here the backward is
  `d x[..., e] = sum_k g[..., k] * (indices[..., k] == e)`, one fused compare-select-reduce.

  Requires `0 <= indices < num_classes` (true for `lax.top_k` output). When the indices along the last axis are
  distinct (top-k), each output element receives at most one nonzero term, so the gradient equals the scatter-add
  result exactly; with repeated indices it is the same sum in a different order.
  """
  del num_classes
  return jnp.take_along_axis(x, indices, axis=-1)


def _take_along_last_axis_dense_vjp_fwd(x, indices, num_classes):
  del num_classes
  return jnp.take_along_axis(x, indices, axis=-1), indices


def _take_along_last_axis_dense_vjp_bwd(num_classes, indices, g):
  iota = jax.lax.broadcasted_iota(indices.dtype, (1,) * indices.ndim + (num_classes,), indices.ndim)
  hit = indices[..., None] == iota  # [..., k, num_classes]
  return jnp.sum(jnp.where(hit, g[..., None], jnp.zeros((), g.dtype)), axis=-2), None


take_along_last_axis_dense_vjp.defvjp(_take_along_last_axis_dense_vjp_fwd, _take_along_last_axis_dense_vjp_bwd)


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
      representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the indices of the selected experts for each
                       token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  selected_num = bs * seq_len * num_experts_per_tok
  # Directly generate random integers in the range [0, num_experts)
  top_k_indices = jax.random.randint(
      rng_key,
      shape=(selected_num,),
      minval=0,
      maxval=num_experts,
      dtype=jnp.int32,
  )
  top_k_indices = top_k_indices.reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def valid_expert_mask(indices, num_experts):
  """True where an expert id is real.

  Forced routing marks unused slots with -1, and any out-of-range id is treated
  the same way.
  """
  return (indices >= 0) & (indices < num_experts)
