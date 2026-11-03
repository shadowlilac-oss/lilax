"""Hugging Face Qwen3.5 weight shapes for checkpoint conversion."""


def QWEN3_5_HF_WEIGHTS_TO_SHAPE(config, *, hf_prefix: str = "model.language_model"):
  """Returns mapping between HuggingFace Qwen3.5 weights path and their shape."""
  if "text_config" in config:
    config = config["text_config"]
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  num_experts = config.get("num_experts", 0) or 0
  head_dim = config["head_dim"]
  linear_conv_kernel_dim = config["linear_conv_kernel_dim"]
  linear_key_head_dim = config["linear_key_head_dim"]
  linear_value_head_dim = config["linear_value_head_dim"]
  linear_num_key_heads = config["linear_num_key_heads"]
  linear_num_value_heads = config["linear_num_value_heads"]
  moe_intermediate_size = config.get("moe_intermediate_size", 0)
  shared_expert_intermediate_size = config.get("shared_expert_intermediate_size", 0)
  intermediate_size = config.get("intermediate_size", 0)
  cycle_interval = config["full_attention_interval"]
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim
  linear_k_dim = linear_num_key_heads * linear_key_head_dim
  linear_v_dim = linear_num_value_heads * linear_value_head_dim
  conv_dim = 2 * linear_k_dim + linear_v_dim
  mapping = {
      f"{hf_prefix}.embed_tokens.weight": [vocab_size, hidden_size],
      f"{hf_prefix}.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }
  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"{hf_prefix}.layers.{layer_idx}"
    mapping[f"{layer_prefix}.input_layernorm.weight"] = [hidden_size]
    mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = [hidden_size]
    is_full_attention_layer = (layer_idx + 1) % cycle_interval == 0
    if is_full_attention_layer:
      mapping.update({
          f"{layer_prefix}.self_attn.q_proj.weight": [2 * q_dim, hidden_size],
          f"{layer_prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
          f"{layer_prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
          f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
          f"{layer_prefix}.self_attn.q_norm.weight": [head_dim],
          f"{layer_prefix}.self_attn.k_norm.weight": [head_dim],
      })
    else:
      mapping.update({
          f"{layer_prefix}.linear_attn.in_proj_qkv.weight": [conv_dim, hidden_size],
          f"{layer_prefix}.linear_attn.in_proj_z.weight": [linear_v_dim, hidden_size],
          f"{layer_prefix}.linear_attn.in_proj_b.weight": [linear_num_value_heads, hidden_size],
          f"{layer_prefix}.linear_attn.in_proj_a.weight": [linear_num_value_heads, hidden_size],
          f"{layer_prefix}.linear_attn.conv1d.weight": [conv_dim, 1, linear_conv_kernel_dim],
          f"{layer_prefix}.linear_attn.A_log": [linear_num_value_heads],
          f"{layer_prefix}.linear_attn.dt_bias": [linear_num_value_heads],
          f"{layer_prefix}.linear_attn.norm.weight": [linear_value_head_dim],
          f"{layer_prefix}.linear_attn.out_proj.weight": [hidden_size, linear_v_dim],
          (f"{layer_prefix}.linear_attn.in_proj_qkv.weight", f"{layer_prefix}.linear_attn.in_proj_z.weight"): (
              [conv_dim, hidden_size], [linear_v_dim, hidden_size],
          ),
          (f"{layer_prefix}.linear_attn.in_proj_b.weight", f"{layer_prefix}.linear_attn.in_proj_a.weight"): (
              [linear_num_value_heads, hidden_size], [linear_num_value_heads, hidden_size],
          ),
      })
    if num_experts > 1:
      mapping.update({
          f"{layer_prefix}.mlp.gate.weight": [num_experts, hidden_size],
          f"{layer_prefix}.mlp.shared_expert.gate_proj.weight": [shared_expert_intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.shared_expert.up_proj.weight": [shared_expert_intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.shared_expert.down_proj.weight": [hidden_size, shared_expert_intermediate_size],
          f"{layer_prefix}.mlp.shared_expert_gate.weight": [1, hidden_size],
          f"{layer_prefix}.mlp.experts.gate_up_proj": [num_experts, 2 * moe_intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.experts.down_proj": [num_experts, hidden_size, moe_intermediate_size],
      })
    else:
      mapping.update({
          f"{layer_prefix}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.up_proj.weight": [intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.down_proj.weight": [hidden_size, intermediate_size],
      })
  return mapping
