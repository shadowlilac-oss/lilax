"""Hugging Face Qwen 2.5 / Qwen 3 weight shapes for checkpoint conversion."""


def QWEN_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen weights path and the HuggingFace weights shape."""
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
  attention_bias = config.get("attention_bias", False)

  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [config["vocab_size"], hidden_size],
  }
  num_experts = config.get("num_experts", config.get("num_local_experts", 0))

  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"
    layer_mapping = {
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
        f"{layer_prefix}.self_attn.q_proj.weight": [num_attention_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.k_proj.weight": [num_key_value_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.v_proj.weight": [num_key_value_heads * head_dim, hidden_size],
        f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, num_attention_heads * head_dim],
        f"{layer_prefix}.self_attn.q_norm.weight": [head_dim],
        f"{layer_prefix}.self_attn.k_norm.weight": [head_dim],
    }
    if attention_bias:
      layer_mapping.update(
          {
              f"{layer_prefix}.self_attn.q_proj.bias": [num_attention_heads * head_dim],
              f"{layer_prefix}.self_attn.k_proj.bias": [num_key_value_heads * head_dim],
              f"{layer_prefix}.self_attn.v_proj.bias": [num_key_value_heads * head_dim],
          }
      )
    if num_experts > 1:
      moe_ffn_intermediate_size = config.get("moe_intermediate_size")
      if moe_ffn_intermediate_size is None:
        moe_ffn_intermediate_size = config.get("intermediate_size")
        if moe_ffn_intermediate_size is None:
          raise ValueError(
              "MoE model detected (num_experts > 1) but 'moe_intermediate_size' or 'intermediate_size' not found in config."
          )
      layer_mapping.update({f"{layer_prefix}.mlp.gate.weight": [num_experts, hidden_size]})
      for expert_j in range(num_experts):
        expert_prefix = f"{layer_prefix}.mlp.experts.{expert_j}"
        layer_mapping.update(
            {
                f"{expert_prefix}.gate_proj.weight": [moe_ffn_intermediate_size, hidden_size],
                f"{expert_prefix}.up_proj.weight": [moe_ffn_intermediate_size, hidden_size],
                f"{expert_prefix}.down_proj.weight": [hidden_size, moe_ffn_intermediate_size],
            }
        )
    else:
      dense_ffn_intermediate_size = config.get("intermediate_size")
      if dense_ffn_intermediate_size is None:
        raise ValueError("'intermediate_size' not found in config for a dense MLP.")
      layer_mapping.update(
          {
              f"{layer_prefix}.mlp.gate_proj.weight": [dense_ffn_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.up_proj.weight": [dense_ffn_intermediate_size, hidden_size],
              f"{layer_prefix}.mlp.down_proj.weight": [hidden_size, dense_ffn_intermediate_size],
          }
      )
    mapping.update(layer_mapping)
  return mapping
