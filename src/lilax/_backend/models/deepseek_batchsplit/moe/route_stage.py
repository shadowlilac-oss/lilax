"""Shared expert and routing stage of the DeepSeek BF16 batch-split MoE block."""

import jax.numpy as jnp

from lilax._backend.models.deepseek_batchsplit.moe.dispatch import route
from lilax._backend.models.deepseek_batchsplit.moe.experts import shared_expert
from lilax._backend.models.deepseek_batchsplit.moe.gating import expert_selection
from lilax._backend.models.deepseek_batchsplit.norm import rms_norm


def shared_expert_and_route(
    inputs,
    post_attn_scale,
    shared_w0,
    shared_w1,
    shared_wo,
    gate_kernel,
    gate_bias,
    *,
    num_experts,
    num_experts_per_tok,
    num_expert_shards,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
):
  """Computes the shared expert and routes the activations."""
  inputs = rms_norm(
      inputs,
      post_attn_scale,
      epsilon=normalization_layer_epsilon,
      dtype=dtype,
  )
  y = shared_expert(inputs, shared_w0, shared_w1, shared_wo)

  inputs = jnp.reshape(inputs, (-1, inputs.shape[-1]))
  selected_experts, weights, group_sizes = expert_selection(
      inputs,
      gate_kernel,
      gate_bias,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      top_k_in_group=top_k_in_group,
  )
  x, selected_experts, weights, group_sizes = route(
      inputs,
      selected_experts,
      weights,
      group_sizes,
      expert_axis_name=expert_axis_name,
      num_experts=num_experts,
      num_expert_shards=num_expert_shards,
      use_gather_mosaic_kernel=use_gather_mosaic_kernel,
  )
  return x, y, selected_experts, weights, group_sizes
