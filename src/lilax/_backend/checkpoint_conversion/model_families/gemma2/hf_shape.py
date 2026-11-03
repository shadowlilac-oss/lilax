"""Hugging Face Gemma 2 weight shapes for checkpoint conversion."""


def GEMMA2_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and weights shape."""
  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
      "model.norm.weight": [config["hidden_size"]],
  }
  for layer_idx in range(config["num_hidden_layers"]):
    layer_mapping = {
        f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.down_proj.weight": [config["hidden_size"], config["intermediate_size"]],
        f"model.layers.{layer_idx}.mlp.up_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.post_feedforward_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": [
            config["num_key_value_heads"] * config["head_dim"], config["hidden_size"]
        ],
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": [
            config["hidden_size"], config["num_attention_heads"] * config["head_dim"]
        ],
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": [
            config["num_attention_heads"] * config["head_dim"], config["hidden_size"]
        ],
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": [
            config["num_key_value_heads"] * config["head_dim"], config["hidden_size"]
        ],
    }
    mapping = {**mapping, **layer_mapping}
  return mapping
