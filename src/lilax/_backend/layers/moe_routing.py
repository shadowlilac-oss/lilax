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


def top_2_in_group_sum(scores_grouped: jax.Array) -> jax.Array:
  """Sum the two largest scores in each group, accumulating in float32."""
  if scores_grouped.shape[-1] < 2:
    raise ValueError("Each routing group needs at least 2 experts, got" f" {scores_grouped.shape[-1]}.")

  max_1 = jnp.max(scores_grouped, axis=-1)
  argmax_1 = jnp.argmax(scores_grouped, axis=-1)[..., None]
  lane = jax.lax.broadcasted_iota(argmax_1.dtype, scores_grouped.shape, scores_grouped.ndim - 1)
  max_2 = jnp.max(jnp.where(lane == argmax_1, -jnp.inf, scores_grouped), axis=-1)
  return max_1.astype(jnp.float32) + max_2.astype(jnp.float32)


class ExpertGroupRouting:
  @staticmethod
  def deepseek_scale_weights(weights, routed_score_func, routed_scaling_factor):
    """Scales weights according to DeepSeek's v3 reference implementation."""
    if routed_score_func in ("sigmoid", "sqrtsoftplus"):
      weights /= weights.sum(-1, keepdims=True) + 1e-20
    weights *= routed_scaling_factor
    return weights

  @staticmethod
  def expert_group_mask(gate_logits: jax.Array, n_routing_groups, topk_routing_group, num_experts) -> jax.Array:
    """Select experts from the groups with the largest top-two scores."""
    scores_grouped = jnp.reshape(
        gate_logits,
        gate_logits.shape[:-1] + (n_routing_groups, -1),
    )
    group_scores = top_2_in_group_sum(scores_grouped)
    _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)

    group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
    group_mask = jnp.sum(group_mask, axis=-2)

    score_mask_expanded = jnp.broadcast_to(
        group_mask[..., None],
        group_mask.shape + (num_experts // n_routing_groups,),
    )
    return jnp.reshape(
        score_mask_expanded,
        score_mask_expanded.shape[:-2] + (num_experts,),
    )
