"""Node-limited expert selection for the DeepSeek BF16 batch-split schedule."""

import jax
import jax.numpy as jnp

from lilax._backend.models.deepseek_batchsplit.dot import dot


# NOTE: Consider deduplicating this logic which is also in `moe.py`.
def expert_group_mask(gate_logits, *, n_routing_groups, topk_routing_group, top_k_in_group):
  """Computes expert group mask for node-limited routing."""
  num_experts = gate_logits.shape[-1]
  # Find top groups based on each group's top-2 expert scores, where
  # `scores_grouped.shape =
  # (batch * seq, n_routing_groups, experts_per_group)`.
  scores_grouped = jnp.reshape(
      gate_logits,
      gate_logits.shape[:-1] + (n_routing_groups, -1),
  )
  top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=top_k_in_group)
  group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)

  # Mask selected groups so that only those experts are considered.
  group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
  group_mask = jnp.sum(group_mask, axis=-2)

  # Apply masks and get top-k indices.
  score_mask_expanded = jnp.broadcast_to(
      group_mask[..., None],
      group_mask.shape + (num_experts // n_routing_groups,),
  )
  return jnp.reshape(
      score_mask_expanded,
      score_mask_expanded.shape[:-2] + (num_experts,),
  )


def expert_indices_and_weights(
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    top_k_in_group: int,
) -> tuple[jax.Array, jax.Array]:
  """Computes expert indices for each token and their corresponding weights."""
  expert_mask = expert_group_mask(
      gate_logits,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      top_k_in_group=top_k_in_group,
  )
  gate_logits = jnp.where(expert_mask > 0, gate_logits, -jnp.inf)

  _, indices = jax.lax.top_k(
      gate_logits,
      k=num_experts_per_tok,
  )
  weights = jnp.take_along_axis(pre_bias_logits, indices, axis=-1)
  weights = routed_scaling_factor * (weights / weights.sum(-1, keepdims=True))
  return indices, weights


def expert_selection(
    x,
    routing_kernel,
    routing_bias,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    top_k_in_group: int,
):
  """Selects experts for each token and calculates group sizes for each expert."""
  pre_bias_logits = jax.nn.sigmoid(dot(x, routing_kernel))
  logits = pre_bias_logits + routing_bias

  selected_experts, weights = expert_indices_and_weights(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      top_k_in_group=top_k_in_group,
  )
  group_sizes = jnp.bincount(jnp.ravel(selected_experts), length=num_experts)
  return selected_experts, weights, group_sizes
