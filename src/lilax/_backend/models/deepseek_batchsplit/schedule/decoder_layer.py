"""Single decoder layer forward and backward for the DeepSeek BF16 batch-split schedule."""

from lilax._backend.models.deepseek_batchsplit.attention.block import (
    mla_with_norms,
    mla_with_norms_bwd,
    mla_with_norms_remat,
)
from lilax._backend.models.deepseek_batchsplit.moe.layer import moe, moe_bwd


def batch_split_schedule(
    inputs,
    weights,
    positions,
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  norm_mla_ws, moe_ws = weights
  xs, attn_res = mla_with_norms(
      inputs,
      norm_mla_ws,
      positions,
      mesh=mesh,
      config=cfg,
      splash_kernel=splash_kernel,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  xs, moe_res = moe(
      xs,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      n_routing_groups=cfg.n_routing_groups,
      topk_routing_group=cfg.topk_routing_group,
      top_k_in_group=2,
      expert_axis_name="expert",
      num_expert_shards=mesh.shape.get("expert", 1),
      # Capacity factor indicates how many tokens will be checkpointed in the MoE
      # block. It does not actually drop any tokens, unlike the `capacity_factor`
      # option available in maxtext configs.
      capacity_factor=2,
      use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
      config=cfg,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  return xs, {"layer_inputs": inputs, **attn_res, **moe_res}


def batch_split_schedule_bwd(
    residuals,
    outputs_grad,
    weights,
    positions,
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Performs the backward pass for a single layer."""
  norm_mla_ws, moe_ws = weights
  _, mla_bwds = mla_with_norms_remat(
      residuals,
      norm_mla_ws,
      positions,
      mesh=mesh,
      config=cfg,
      splash_kernel=splash_kernel,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      num_query_heads=cfg.num_query_heads,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=cfg.rope_factor,
      mscale=cfg.mscale,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  attn_out_grad, moe_ws_grad = moe_bwd(
      residuals,
      outputs_grad,
      moe_ws,
      mesh=mesh,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      routed_scaling_factor=cfg.routed_scaling_factor,
      n_routing_groups=cfg.n_routing_groups,
      topk_routing_group=cfg.topk_routing_group,
      top_k_in_group=2,
      expert_axis_name="expert",
      num_expert_shards=mesh.shape.get("expert", 1),
      use_gather_mosaic_kernel=cfg.use_gather_mosaic_kernel,
      config=cfg,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      dtype=cfg.dtype,
      activation_pspec=activation_pspec,
  )
  inputs_grad, norm_mla_ws_grad = mla_with_norms_bwd(attn_out_grad, mla_bwds)
  return inputs_grad, (norm_mla_ws_grad, moe_ws_grad)
