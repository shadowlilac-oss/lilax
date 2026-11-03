"""FLOPs calculation per device for Qwen3.8-Flash-Next training."""

from typing import Any


def _ffn_matmul_flops(config: Any, mlp_dim: int) -> float:
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  ffn1_flops = 2 * B * S * mlp_dim * E * len(config.mlp_activations)
  ffn2_flops = 2 * B * S * mlp_dim * E
  return ffn1_flops + ffn2_flops


def _moe_ffn_flops_all_layers(config: Any) -> float:
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  gate_flops = 2 * B * S * E * config.num_experts
  shared_experts_flops = _ffn_matmul_flops(config, config.mlp_dim) * config.shared_experts
  routed_experts_flops = _ffn_matmul_flops(config, config.moe_mlp_dim) * config.num_experts_per_tok
  return (gate_flops + shared_experts_flops + routed_experts_flops) * config.num_decoder_layers


def _gated_delta_net_flops_per_layer(config: Any) -> tuple[float, float]:
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  H_k = config.gdn_num_key_heads
  H_v = config.gdn_num_value_heads
  D_k = config.gdn_key_head_dim
  D_v = config.gdn_value_head_dim
  K_conv = config.gdn_conv_kernel_dim
  K_dim = H_k * D_k
  V_dim = H_v * D_v
  flops_qkvz_ba = 2 * B * S * E * (2 * K_dim + 2 * V_dim + 2 * H_v)
  flops_out = 2 * B * S * E * V_dim
  flops_projections = flops_qkvz_ba + flops_out
  flops_conv = 2 * B * S * K_conv * (2 * K_dim + V_dim)
  flops_core_per_token = H_v * (D_k * D_v) * 8
  flops_core = B * S * flops_core_per_token
  return flops_projections + flops_conv, flops_core


def _indexer_mask_ratio(indexer_topk: int, max_target_length: int) -> float:
  T = float(max_target_length)
  K = float(indexer_topk)
  ratio = K / T
  return ratio - (0.5 * ratio**2)


def calculate_qwen3_8_flash_next_tflops_per_device(config: Any, log: bool = True) -> tuple[float, float, float]:
  """Calculate per-device training TFLOPs for the hybrid architecture."""
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  num_layers = config.num_decoder_layers
  cycle_interval = config.inhomogeneous_layer_cycle_interval
  num_qsa_layers = num_layers // cycle_interval
  num_gdn_layers = num_layers - num_qsa_layers
  total_ffn_flops_all_layers = _moe_ffn_flops_all_layers(config)
  embedding_flops = 2 * B * S * E * config.vocab_size
  gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = _gated_delta_net_flops_per_layer(config)

  H_q = config.num_query_heads
  H_kv = config.num_kv_heads
  D_head = config.head_dim
  index_n_heads = getattr(config, "index_n_heads", 0)
  index_kv_heads = getattr(config, "index_kv_heads", 1)
  index_head_dim = getattr(config, "index_head_dim", 0)
  compress_ratio = getattr(config, "index_select_block_size", 4) or 4
  token_budget = getattr(config, "index_select_budget", 0)
  qsa_q_flops = 2 * B * S * E * (2 * H_q * D_head)
  qsa_kv_flops = 2 * B * S * E * (2 * H_kv * D_head)
  qsa_o_flops = 2 * B * S * (H_q * D_head) * E
  idx_proj_flops = 2 * B * S * E * (index_n_heads + index_kv_heads) * index_head_dim
  qsa_weight_flops_per_layer = qsa_q_flops + qsa_kv_flops + qsa_o_flops + idx_proj_flops
  num_blocks = S / compress_ratio
  idx_score_flops = B * S * num_blocks * index_n_heads * index_head_dim
  idx_reduce_flops = B * S * num_blocks * index_n_heads
  effective_budget = min(token_budget, S) if token_budget > 0 else S
  qsa_mask_ratio = _indexer_mask_ratio(effective_budget, S)
  qsa_sparse_attn_flops = 4 * B * (S**2) * H_q * D_head * qsa_mask_ratio
  qsa_attn_flops_per_layer = qsa_sparse_attn_flops + idx_score_flops + idx_reduce_flops

  hc_count = getattr(config, "hc_count", 1)
  hc_lowrank = getattr(config, "hc_lowrank", 0)
  hc_dim = hc_count * E
  if hc_count > 1 and hc_lowrank > 0:
    hc_block_flops = 2 * B * S * hc_dim * (2 * hc_lowrank + hc_count)
    hc_layer_flops = 2 * hc_block_flops
    hc_mixer_flops = 2 * B * S * hc_dim * (2 * hc_lowrank)
    total_hc_flops = num_layers * hc_layer_flops + hc_mixer_flops
  else:
    hc_layer_flops = 0
    hc_mixer_flops = 0
    total_hc_flops = 0
  shared_expert_gate_flops_per_layer = 2 * B * S * E * getattr(config, "shared_experts", 0)
  total_shared_gate_flops = num_layers * shared_expert_gate_flops_per_layer
  if getattr(config, "use_ple_layer", False):
    ple_dim = getattr(config, "ple_embed_dim", 0) or E
    ple_conv_k = getattr(config, "ple_conv_kernel_size", 4)
    ple_weight_flops = 2 * B * S * (ple_dim * hc_dim + ple_dim * E + ple_conv_k * hc_dim)
    ple_attn_flops = 2 * B * S * hc_dim
  else:
    ple_weight_flops = 0
    ple_attn_flops = 0

  mtp_num_layers = getattr(config, "mtp_num_layers", 0)
  if mtp_num_layers > 0:
    mtp_fc_flops = 2 * B * S * (1 + hc_count) * E * E
    moe_layer_flops = (total_ffn_flops_all_layers / num_layers) + shared_expert_gate_flops_per_layer
    mtp_weight_flops_per_layer = (
        mtp_fc_flops + qsa_weight_flops_per_layer + hc_layer_flops + moe_layer_flops + hc_mixer_flops + embedding_flops
    )
    mtp_weight_flops = mtp_num_layers * mtp_weight_flops_per_layer
    mtp_attn_flops = mtp_num_layers * qsa_attn_flops_per_layer
  else:
    mtp_weight_flops = 0
    mtp_attn_flops = 0

  total_weight_flops = (
      total_ffn_flops_all_layers
      + total_shared_gate_flops
      + embedding_flops
      + num_qsa_layers * qsa_weight_flops_per_layer
      + num_gdn_layers * gdn_weight_flops_per_layer
      + total_hc_flops
      + ple_weight_flops
      + mtp_weight_flops
  )
  total_attn_flops = (
      num_qsa_layers * qsa_attn_flops_per_layer
      + num_gdn_layers * gdn_attn_flops_per_layer
      + ple_attn_flops
      + mtp_attn_flops
  )
  learnable_weight_tflops = total_weight_flops * 3 / 10**12
  attention_tflops = total_attn_flops * 3 / 10**12
  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps
  if config.use_dpo:
    reference_model_tflops = learnable_weight_tflops / 3
    reference_model_attention_tflops = attention_tflops / 3
    attention_tflops = attention_tflops + reference_model_attention_tflops
  else:
    reference_model_tflops = 0
  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops
  if log:
    print(
        f"{config.model_name} per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops / total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops / total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
