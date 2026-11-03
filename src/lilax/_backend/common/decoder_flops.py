"""Architecture-specific decoder attention and weight TFLOPs accounting."""

from lilax._backend.common import flops
from lilax._backend.common.common_types import AttentionType
from lilax._backend.models import gemma4_small


def calculate_gemma2_tflops_training_per_device(config, total_ffn_flops, qkv_flops, projection_flops, embedding_flops):
  window = min(config.sliding_window_size, config.max_target_length)
  global_causal_flops = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )
  local_causal_flops = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  attention_tflops = (global_causal_flops + local_causal_flops) * config.num_decoder_layers * 3 / 10**12
  learnable_weight_tflops = (
      ((total_ffn_flops + qkv_flops + projection_flops) * config.num_decoder_layers * 2 + embedding_flops) * 3 / 10**12
  )
  return attention_tflops, learnable_weight_tflops


def calculate_mixed_attention_model_tflops_training_per_device(
    config, total_ffn_flops_all_layers, qkv_flops, projection_flops, embedding_flops, attention_pattern_length
):
  num_layers = config.num_decoder_layers
  num_global_layers = num_layers // attention_pattern_length
  num_local_layers = num_layers - num_global_layers
  global_causal_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * config.head_dim
  )
  window = min(config.sliding_window_size, config.max_target_length)
  local_causal_flops_per_layer = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  causal_attention_flops = (
      num_global_layers * global_causal_flops_per_layer + num_local_layers * local_causal_flops_per_layer
  )
  attention_tflops = causal_attention_flops * 3 / 10**12
  total_learnable_flops = total_ffn_flops_all_layers
  total_learnable_flops += (qkv_flops + projection_flops) * num_layers + embedding_flops
  learnable_weight_tflops = total_learnable_flops * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def calculate_gemma4_tflops_training_per_device(
    config, total_ffn_flops_all_layers, embedding_flops, attention_pattern_length
):
  num_layers = config.num_decoder_layers
  num_global_layers = num_layers // attention_pattern_length
  num_local_layers = num_layers - num_global_layers
  kv_multiplier = 1 if config.share_kv_projections else 2
  global_head_dim = config.global_head_dim or config.head_dim
  global_num_kv_heads = config.global_num_kv_heads or config.num_kv_heads
  global_causal_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length**2 * config.num_query_heads * global_head_dim
  )
  window = min(config.sliding_window_size, config.max_target_length)
  local_causal_flops_per_layer = (
      4 * config.per_device_batch_size * (config.max_target_length * window - 0.5 * window**2)
      * config.num_query_heads * config.head_dim
  )
  causal_attention_flops = (
      num_global_layers * global_causal_flops_per_layer + num_local_layers * local_causal_flops_per_layer
  )
  attention_tflops = causal_attention_flops * 3 / 10**12
  global_qkv_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim
      * (config.num_query_heads + kv_multiplier * global_num_kv_heads) * global_head_dim
  )
  global_projection_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_query_heads * global_head_dim
  )
  local_qkv_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim
      * (config.num_query_heads + 2 * config.num_kv_heads) * config.head_dim
  )
  local_projection_flops_per_layer = (
      2 * config.per_device_batch_size * config.max_target_length * config.emb_dim * config.num_query_heads * config.head_dim
  )
  total_learnable_flops = total_ffn_flops_all_layers
  total_learnable_flops += (
      (local_qkv_flops_per_layer + local_projection_flops_per_layer) * num_local_layers
      + (global_qkv_flops_per_layer + global_projection_flops_per_layer) * num_global_layers + embedding_flops
  )
  learnable_weight_tflops = total_learnable_flops * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size):
  num_chunks = seq_len // chunk_size
  rem_chunk_size = seq_len % chunk_size
  chunked_complexity = (num_chunks * chunk_size**2) + (rem_chunk_size**2)
  return 4 * config.per_device_batch_size * chunked_complexity * config.num_query_heads * config.head_dim


def calculate_llama4_attention_tflops(config):
  num_layers = config.num_decoder_layers
  seq_len = config.max_target_length
  chunk_size = config.chunk_attn_window_size
  num_global_layers = num_layers // config.nope_layer_interval
  num_chunked_layers = num_layers - num_global_layers
  global_attention_flops_per_layer = (
      4 * config.per_device_batch_size * seq_len**2 * config.num_query_heads * config.head_dim
  )
  chunked_attention_flops_per_layer = _calculate_chunked_attention_flops_per_layer(config, seq_len, chunk_size)
  noncausal_attention_flops = (num_global_layers * global_attention_flops_per_layer) + (
      num_chunked_layers * chunked_attention_flops_per_layer
  )
  causal_attention_flops = noncausal_attention_flops / 2
  attention_tflops = causal_attention_flops * 3 / 10**12
  return attention_tflops


def calculate_gemma4_small_tflops_training_per_device(config, embedding_flops):
  b = config.per_device_batch_size
  s = config.max_target_length
  num_layers = config.num_decoder_layers
  num_kv_shared = config.num_kv_shared_layers
  layer_types = gemma4_small.build_layer_types(num_layers, config.model_name)
  first_shared = gemma4_small.first_kv_shared_layer_idx(num_layers, num_kv_shared)
  global_head_dim = config.global_head_dim or config.head_dim
  global_num_kv_heads = config.global_num_kv_heads or config.num_kv_heads
  num_activations = len(config.mlp_activations)
  window = min(config.sliding_window_size, s)

  def attention_flops_for(attn_type):
    if attn_type == AttentionType.GLOBAL:
      return 2 * b * s * s * config.num_query_heads * global_head_dim
    return 4 * b * (s * window - 0.5 * window**2) * config.num_query_heads * config.head_dim

  def qo_flops_for(attn_type):
    head_dim = global_head_dim if attn_type == AttentionType.GLOBAL else config.head_dim
    q = 2 * b * s * config.emb_dim * config.num_query_heads * head_dim
    o = 2 * b * s * config.emb_dim * config.num_query_heads * head_dim
    return q + o

  def kv_flops_for(attn_type):
    if attn_type == AttentionType.GLOBAL:
      kv_heads, head_dim = global_num_kv_heads, global_head_dim
    else:
      kv_heads, head_dim = config.num_kv_heads, config.head_dim
    return 2 * b * s * config.emb_dim * (2 * kv_heads) * head_dim

  attention_flops = 0
  projection_flops = 0
  ffn_per_layer = 2 * b * s * config.mlp_dim * config.emb_dim * num_activations + 2 * b * s * config.mlp_dim * config.emb_dim
  ffn_flops = 0
  for lyr_idx, attn_type in enumerate(layer_types):
    attention_flops += attention_flops_for(attn_type)
    projection_flops += qo_flops_for(attn_type)
    is_shared = num_kv_shared > 0 and lyr_idx >= first_shared
    if not is_shared:
      projection_flops += kv_flops_for(attn_type)
    if is_shared and config.use_double_wide_mlp:
      ffn_flops += 2 * ffn_per_layer
    else:
      ffn_flops += ffn_per_layer
  ple_flops = 0
  ple_dim = config.hidden_size_per_layer_input
  if ple_dim > 0:
    ple_flops += 2 * b * s * config.emb_dim * (num_layers * ple_dim)
    ple_flops += num_layers * (2 * b * s * config.emb_dim * ple_dim + 2 * b * s * ple_dim * config.emb_dim)
  attention_tflops = attention_flops * 3 / 10**12
  learnable_weight_tflops = (projection_flops + ffn_flops + ple_flops + embedding_flops) * 3 / 10**12
  return attention_tflops, learnable_weight_tflops


def calculate_mhc_flops_per_layer(config):
  batch_size = config.per_device_batch_size
  seq_len = config.max_target_length
  k_mhc = getattr(config, "mhc_expansion_rate", 1)
  emb_dim = config.emb_dim
  pre_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * k_mhc
  pre_contract_flops = 2 * batch_size * seq_len * k_mhc * emb_dim
  post_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * k_mhc
  post_outer_flops = batch_size * seq_len * k_mhc * emb_dim
  res_alpha_flops = 2 * batch_size * seq_len * (k_mhc * emb_dim) * (k_mhc**2)
  res_contract_flops = 2 * batch_size * seq_len * (k_mhc**2) * emb_dim
  return pre_alpha_flops + pre_contract_flops + post_alpha_flops + post_outer_flops + res_alpha_flops + res_contract_flops


def calculate_deepseek4_tflops_training_per_device(config, total_ffn_flops_all_layers, embedding_flops):
  batch_size = config.per_device_batch_size
  seq_len = config.max_target_length
  sliding_window_size = config.sliding_window_size
  num_layers = config.num_decoder_layers
  HCA_RATIO = 128
  CSA_RATIO = 4
  c_ratios = getattr(config, "compress_ratios", [])
  if not c_ratios:
    num_sliding_layers, num_hca_layers, num_csa_layers = 2, 20, 21
  else:
    num_sliding_layers = c_ratios.count(0)
    num_hca_layers = c_ratios.count(HCA_RATIO)
    num_csa_layers = c_ratios.count(CSA_RATIO)
  wq_flops = (
      2 * batch_size * seq_len
      * ((config.emb_dim * config.q_lora_rank) + (config.q_lora_rank * config.num_query_heads * config.head_dim))
  )
  wkv_flops = 2 * batch_size * seq_len * config.emb_dim * (config.num_kv_heads * config.head_dim)
  grp_feat = (config.num_query_heads * config.head_dim) // config.o_groups
  o_flops = (2 * batch_size * seq_len * config.o_groups * grp_feat * config.o_lora_rank) + (
      2 * batch_size * seq_len * (config.o_groups * config.o_lora_rank) * config.emb_dim
  )
  base_attention_weights = wq_flops + wkv_flops + o_flops
  if getattr(config, "mhc_expansion_rate", 1) > 1:
    mhc_flops_per_layer = calculate_mhc_flops_per_layer(config)
  else:
    mhc_flops_per_layer = 0
  prefix_attn = (
      4 * batch_size * (seq_len * sliding_window_size - 0.5 * sliding_window_size**2)
      * config.num_query_heads * config.head_dim
  )
  hca_pooler = 4 * batch_size * seq_len * config.emb_dim * config.head_dim
  hca_attn = prefix_attn + 2 * batch_size * seq_len * (seq_len / HCA_RATIO) * config.num_query_heads * config.head_dim
  idx_proj = (
      2 * batch_size * seq_len
      * ((config.q_lora_rank * config.indexer_n_heads * config.indexer_head_dim) + (config.emb_dim * config.indexer_n_heads))
  )
  idx_score = batch_size * seq_len * (seq_len / CSA_RATIO) * config.indexer_n_heads * config.indexer_head_dim
  idx_reduce = batch_size * seq_len * (seq_len / CSA_RATIO) * config.indexer_n_heads
  idx_kv_gate_pool = 4 * batch_size * seq_len * config.emb_dim * (2 * config.indexer_head_dim)
  csa_indexer = idx_proj + idx_score + idx_reduce + idx_kv_gate_pool
  csa_pooler = 4 * batch_size * seq_len * config.emb_dim * (2 * config.head_dim)
  csa_k = min(config.indexer_topk, seq_len // CSA_RATIO)
  csa_mask_ratio = flops.calculate_indexer_mask_ratio(csa_k, seq_len // CSA_RATIO)
  csa_sparse_attn = (
      4 * batch_size * (seq_len * (seq_len / CSA_RATIO)) * config.num_query_heads * config.head_dim * csa_mask_ratio
  )
  csa_attn = prefix_attn + csa_sparse_attn + csa_indexer
  total_attn_flops = (num_sliding_layers * prefix_attn) + (num_hca_layers * hca_attn) + (num_csa_layers * csa_attn)
  total_weight_flops = (
      (num_layers * base_attention_weights) + (2 * num_layers * mhc_flops_per_layer)
      + (num_hca_layers * hca_pooler) + (num_csa_layers * csa_pooler) + total_ffn_flops_all_layers + embedding_flops
  )
  return total_attn_flops * 3 / 10**12, total_weight_flops * 3 / 10**12
