"""Expert selection for the DeepSeek FP8 batch-split schedule."""

import jax
import jax.numpy as jnp
from lilax._backend.models.deepseek_batchsplit_fp8.dot import dot


def expert_indices_and_weights(
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes expert indices for each token and their corresponding weights."""
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
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    quant,
):
  """Selects experts for each token and calculates group sizes for each expert."""
  pre_bias_logits = jax.nn.sigmoid(dot(x, routing_kernel, quant=quant))
  logits = pre_bias_logits + routing_bias

  selected_experts, weights = expert_indices_and_weights(
      logits,
      pre_bias_logits,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
  )
  group_sizes = jnp.bincount(jnp.ravel(selected_experts), length=num_experts)
  return selected_experts, weights, group_sizes
