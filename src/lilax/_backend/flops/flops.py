"""FLOPs accounting for attention and inference."""

import jax


def calculate_tokens_training_per_device(config):
  """Calculate training Tokens per device"""
  return config.max_target_length * config.per_device_batch_size * config.gradient_accumulation_steps


def calculate_indexer_mask_ratio(indexer_topk, max_target_length):
  T = float(max_target_length)
  K = float(indexer_topk)
  ratio = K / T
  mask_multiplier = ratio - (0.5 * ratio**2)
  return mask_multiplier


def calculate_indexer_tflops_per_device(config):
  batch_len = config.per_device_batch_size * config.max_target_length
  q_flops = 2 * batch_len * config.q_lora_rank * config.indexer_n_heads * config.indexer_head_dim
  k_flops = 2 * batch_len * config.emb_dim * config.indexer_head_dim
  head_weight_flops = 2 * batch_len * config.emb_dim * config.indexer_n_heads
  proj_flops = q_flops + k_flops + head_weight_flops
  qk_product_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads * config.indexer_head_dim
  head_reduction_flops = 2 * batch_len * config.max_target_length * config.indexer_n_heads
  scoring_flops = (qk_product_flops + head_reduction_flops) / 2
  return proj_flops, scoring_flops


def calculate_mla_tflops_per_device(config):
  batch_len = config.per_device_batch_size * config.max_target_length
  qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
  if config.q_lora_rank == 0:
    q_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * qk_head_dim_sum
  else:
    q_flops = (
        2 * batch_len
        * (config.emb_dim * config.q_lora_rank + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum)
    )
  kv_flops = (
      2 * batch_len
      * (config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
         + config.kv_lora_rank * config.num_query_heads * (config.qk_nope_head_dim + config.v_head_dim))
  )
  qkv_flops = q_flops + kv_flops
  if config.use_indexer and config.max_target_length > config.indexer_topk:
    indexer_proj_flops, indexer_scoring_flops = calculate_indexer_tflops_per_device(config)
    qkv_flops += indexer_proj_flops
    multiplier = calculate_indexer_mask_ratio(config.indexer_topk, config.max_target_length)
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads
        * (qk_head_dim_sum + config.v_head_dim) * multiplier
    )
    attention_flops += indexer_scoring_flops
  else:
    attention_flops = (
        2 * batch_len * config.max_target_length * config.num_query_heads * (qk_head_dim_sum + config.v_head_dim)
    )
    attention_flops = attention_flops / 2
  projection_flops = 2 * batch_len * config.emb_dim * config.num_query_heads * config.v_head_dim
  return qkv_flops, attention_flops, projection_flops


def calculate_gated_delta_net_flops_per_device(config):
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
  gdn_weight_flops = flops_projections + flops_conv
  gdn_attn_flops = flops_core
  return gdn_weight_flops, gdn_attn_flops


def calculate_prefill_tflops_per_device(num_model_parameters, prefill_length, config, log=True):
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12
  noncausal_attention_flops = (
      4 * config.num_query_heads * config.num_decoder_layers * config.head_dim * prefill_length**2
      / jax.device_count() / 1e12
  )
  causal_attention_tflops = noncausal_attention_flops / 2
  total_tflops = learnable_weight_tflops + causal_attention_tflops
  if log:
    print(
        "Per prefill step per device: \n",
        f"\tTotal TFLOPs: {total_tflops:.2f} \n",
        f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
        f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
        f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
        f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
    )
  return total_tflops, learnable_weight_tflops, causal_attention_tflops
