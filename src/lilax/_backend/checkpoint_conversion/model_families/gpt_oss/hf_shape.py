"""Hugging Face GPT-OSS weight shapes for checkpoint conversion."""


def GPT_OSS_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace GptOss weights path and their shape."""
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
  intermediate_size = config["intermediate_size"]
  num_local_experts = config["num_local_experts"]
  attention_bias = config.get("attention_bias", False)
  head_dim = config.get("head_dim", hidden_size // num_attention_heads)
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }
  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"
    layer_mapping = {
        f"{layer_prefix}.input_layernorm.weight": [hidden_size],
        f"{layer_prefix}.post_attention_layernorm.weight": [hidden_size],
        f"{layer_prefix}.self_attn.q_proj.weight": [q_dim, hidden_size],
        f"{layer_prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
        f"{layer_prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
        f"{layer_prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
        f"{layer_prefix}.self_attn.sinks": [num_attention_heads],
    }
    if attention_bias:
      layer_mapping.update({
          f"{layer_prefix}.self_attn.q_proj.bias": [q_dim],
          f"{layer_prefix}.self_attn.k_proj.bias": [kv_dim],
          f"{layer_prefix}.self_attn.v_proj.bias": [kv_dim],
          f"{layer_prefix}.self_attn.o_proj.bias": [hidden_size],
      })
    layer_mapping.update({
        f"{layer_prefix}.mlp.router.weight": [num_local_experts, hidden_size],
        f"{layer_prefix}.mlp.router.bias": [num_local_experts],
        f"{layer_prefix}.mlp.experts.gate_up_proj": [num_local_experts, hidden_size, 2 * intermediate_size],
        f"{layer_prefix}.mlp.experts.gate_up_proj_bias": [num_local_experts, 2 * intermediate_size],
        f"{layer_prefix}.mlp.experts.down_proj": [num_local_experts, intermediate_size, hidden_size],
        f"{layer_prefix}.mlp.experts.down_proj_bias": [num_local_experts, hidden_size],
    })
    mapping.update(layer_mapping)
  return mapping
