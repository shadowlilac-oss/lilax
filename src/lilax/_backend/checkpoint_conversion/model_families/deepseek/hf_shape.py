"""Hugging Face DeepSeek V2 / V3 / V3.2 weight shapes for checkpoint conversion."""


def DEEPSEEK_HF_WEIGHTS_TO_SHAPE(config):
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  attention_bias = config.get("attention_bias", False)
  q_lora_rank = config["q_lora_rank"]
  kv_lora_rank = config["kv_lora_rank"]
  num_attention_heads = config["num_attention_heads"]
  qk_head_dim = config.get("qk_head_dim", config["qk_nope_head_dim"] + config["qk_rope_head_dim"])
  q_dim = num_attention_heads * qk_head_dim
  kv_b_dim = num_attention_heads * (config["qk_nope_head_dim"] + config["v_head_dim"])
  o_proj_in_dim = num_attention_heads * config["v_head_dim"]
  kv_a_proj_out_dim = config["kv_lora_rank"] + config["qk_rope_head_dim"]
  intermediate_size = config["intermediate_size"]
  moe_intermediate_size = config["moe_intermediate_size"]
  n_routed_experts = config["n_routed_experts"]
  n_shared_experts = config.get("n_shared_experts", 0)
  first_k_dense = config.get("first_k_dense_replace", 0)
  index_head_dim = config.get("index_head_dim")
  index_n_heads = config.get("index_n_heads")
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }
  mtp_num_layers = config.get("num_nextn_predict_layers", config.get("num_mtp_layers", config.get("mtp_num_layers", 0)))
  for layer_idx in range(num_hidden_layers + mtp_num_layers):
    layer_prefix = f"model.layers.{layer_idx}"
    layer_mapping = {
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
        f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.weight": [kv_a_proj_out_dim, hidden_size],
        f"{layer_prefix}.self_attn.kv_a_layernorm.weight": [kv_lora_rank],
        f"{layer_prefix}.self_attn.kv_b_proj.weight": [kv_b_dim, kv_lora_rank],
        f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, o_proj_in_dim],
    }
    if layer_idx >= num_hidden_layers:
      layer_mapping.update({
          f"{layer_prefix}.enorm.weight": [hidden_size],
          f"{layer_prefix}.hnorm.weight": [hidden_size],
          f"{layer_prefix}.eh_proj.weight": [hidden_size, 2 * hidden_size],
          f"{layer_prefix}.shared_head.norm.weight": [hidden_size],
      })
    if q_lora_rank is None:
      layer_mapping[f"{layer_prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    else:
      layer_mapping.update({
          f"{layer_prefix}.self_attn.q_a_proj.weight": [q_lora_rank, hidden_size],
          f"{layer_prefix}.self_attn.q_a_layernorm.weight": [q_lora_rank],
          f"{layer_prefix}.self_attn.q_b_proj.weight": [q_dim, q_lora_rank],
      })
    if attention_bias:
      if q_lora_rank is not None:
        layer_mapping[f"{layer_prefix}.self_attn.q_a_proj.bias"] = [q_lora_rank]
      layer_mapping.update({
          f"{layer_prefix}.self_attn.kv_a_proj_with_mqa.bias": [kv_a_proj_out_dim],
          f"{layer_prefix}.self_attn.o_proj.bias": [hidden_size],
      })
    if index_head_dim is not None and index_n_heads is not None and q_lora_rank is not None:
      indexer_prefix = f"{layer_prefix}.self_attn.indexer"
      layer_mapping.update({
          f"{indexer_prefix}.k_norm.bias": [index_head_dim],
          f"{indexer_prefix}.k_norm.weight": [index_head_dim],
          f"{indexer_prefix}.weights_proj.weight": [index_n_heads, hidden_size],
          f"{indexer_prefix}.wk.weight": [index_head_dim, hidden_size],
          f"{indexer_prefix}.wq_b.weight": [index_n_heads * index_head_dim, q_lora_rank],
      })
    if layer_idx < first_k_dense:
      layer_mapping.update({
          f"{layer_prefix}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.up_proj.weight": [intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.down_proj.weight": [hidden_size, intermediate_size],
      })
    else:
      layer_mapping.update({
          f"{layer_prefix}.mlp.gate.weight": [n_routed_experts, hidden_size],
          f"{layer_prefix}.mlp.gate.e_score_correction_bias": [n_routed_experts],
      })
      for expert_idx in range(n_routed_experts):
        expert_prefix = f"{layer_prefix}.mlp.experts.{expert_idx}"
        layer_mapping.update({
            f"{expert_prefix}.gate_proj.weight": [moe_intermediate_size, hidden_size],
            f"{expert_prefix}.up_proj.weight": [moe_intermediate_size, hidden_size],
            f"{expert_prefix}.down_proj.weight": [hidden_size, moe_intermediate_size],
        })
      if n_shared_experts > 0:
        shared_intermediate_size = moe_intermediate_size * n_shared_experts
        layer_mapping.update({
            f"{layer_prefix}.mlp.shared_experts.gate_proj.weight": [shared_intermediate_size, hidden_size],
            f"{layer_prefix}.mlp.shared_experts.up_proj.weight": [shared_intermediate_size, hidden_size],
            f"{layer_prefix}.mlp.shared_experts.down_proj.weight": [hidden_size, shared_intermediate_size],
        })
    mapping.update(layer_mapping)
  return mapping
