"""Hugging Face Hy4 weight shapes for checkpoint conversion."""


def HY4_HF_WEIGHTS_TO_SHAPE(config):
  """Returns the HuggingFace parameter shapes for Hy4-preview."""
  hidden_size = config["hidden_size"]
  vocab_size = config["vocab_size"]
  num_hidden_layers = config["num_hidden_layers"]
  intermediate_size = config["intermediate_size"]
  moe_intermediate_size = config["moe_intermediate_size"]
  n_routed_experts = config["n_routed_experts"]
  num_attention_heads = config["num_attention_heads"]
  q_lora_rank = config["q_lora_rank"]
  kv_lora_rank = config["kv_lora_rank"]
  qk_nope_head_dim = config["qk_nope_head_dim"]
  qk_rope_head_dim = config["qk_rope_head_dim"]
  v_head_dim = config["v_head_dim"]
  index_n_heads = config["index_n_heads"]
  index_head_dim = config["index_head_dim"]
  hc_mult = config.get("hc_mult", 4)
  first_num_dense_layers = config.get("first_num_dense_layers", 1)
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.hc_head.hc_head_fn": [hc_mult, hc_mult * hidden_size],
      "model.hc_head.hc_head_base": [hc_mult],
      "model.hc_head.hc_head_scale": [1],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }
  for layer_idx in range(num_hidden_layers):
    pfx = f"model.layers.{layer_idx}"
    is_full_idx = (layer_idx < first_num_dense_layers) or ((layer_idx - first_num_dense_layers) % 4 == 0)
    is_moe = layer_idx >= first_num_dense_layers
    mapping.update({
        f"{pfx}.input_layernorm.weight": [hidden_size],
        f"{pfx}.post_attention_layernorm.weight": [hidden_size],
        f"{pfx}.hc_attn_layer.hc_pre.hc_fn": [2 * hc_mult, hc_mult * hidden_size],
        f"{pfx}.hc_attn_layer.hc_pre.hc_base": [2 * hc_mult],
        f"{pfx}.hc_attn_layer.hc_pre.hc_scale": [2],
        f"{pfx}.hc_mlp_layer.hc_pre.hc_fn": [2 * hc_mult, hc_mult * hidden_size],
        f"{pfx}.hc_mlp_layer.hc_pre.hc_base": [2 * hc_mult],
        f"{pfx}.hc_mlp_layer.hc_pre.hc_scale": [2],
        f"{pfx}.self_attn.q_a_proj.weight": [q_lora_rank, hidden_size],
        f"{pfx}.self_attn.q_a_layernorm.weight": [q_lora_rank],
        f"{pfx}.self_attn.q_b_proj.weight": [num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank],
        f"{pfx}.self_attn.kv_a_proj_with_mqa.weight": [kv_lora_rank + qk_rope_head_dim, hidden_size],
        f"{pfx}.self_attn.kv_a_layernorm.weight": [kv_lora_rank],
        f"{pfx}.self_attn.kv_b_proj.weight": [num_attention_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank],
        f"{pfx}.self_attn.linear_gate.weight": [num_attention_heads * v_head_dim, hidden_size],
        f"{pfx}.self_attn.o_proj.weight": [hidden_size, num_attention_heads * v_head_dim],
        f"{pfx}.self_attn.learnable_sink_param": [num_attention_heads],
    })
    if is_full_idx:
      mapping.update({
          f"{pfx}.self_attn.indexer.wq_b.weight": [index_n_heads * index_head_dim, q_lora_rank],
          f"{pfx}.self_attn.indexer.wk.weight": [index_head_dim, hidden_size],
          f"{pfx}.self_attn.indexer.k_norm.weight": [index_head_dim],
          f"{pfx}.self_attn.indexer.k_norm.bias": [index_head_dim],
          f"{pfx}.self_attn.indexer.weights_proj.weight": [index_n_heads, hidden_size],
      })
    if not is_moe:
      mapping.update({
          f"{pfx}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
          f"{pfx}.mlp.up_proj.weight": [intermediate_size, hidden_size],
          f"{pfx}.mlp.down_proj.weight": [hidden_size, intermediate_size],
      })
    else:
      mapping.update({
          f"{pfx}.mlp.gate.weight": [n_routed_experts, hidden_size],
          f"{pfx}.mlp.gate.e_score_correction_bias": [n_routed_experts],
          f"{pfx}.mlp.experts.gate_up_proj": [n_routed_experts, 2 * moe_intermediate_size, hidden_size],
          f"{pfx}.mlp.experts.down_proj": [n_routed_experts, hidden_size, moe_intermediate_size],
          f"{pfx}.mlp.shared_experts.gate_proj.weight": [moe_intermediate_size, hidden_size],
          f"{pfx}.mlp.shared_experts.up_proj.weight": [moe_intermediate_size, hidden_size],
          f"{pfx}.mlp.shared_experts.down_proj.weight": [hidden_size, moe_intermediate_size],
      })
  return mapping
