"""Hugging Face OLMo 3 weight shapes for checkpoint conversion."""

from lilax._backend.checkpoint_conversion.model_families.qwen.hf_shape import QWEN_HF_WEIGHTS_TO_SHAPE


def OLMO3_HF_WEIGHTS_TO_SHAPE(config):
  mapping = QWEN_HF_WEIGHTS_TO_SHAPE(config)
  head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
  for layer in range(config["num_hidden_layers"]):
    mapping.pop(f"model.layers.{layer}.input_layernorm.weight")
    mapping[f"model.layers.{layer}.self_attn.q_norm.weight"] = [config["num_attention_heads"] * head_dim]
    mapping[f"model.layers.{layer}.self_attn.k_norm.weight"] = [config["num_key_value_heads"] * head_dim]
    mapping[f"model.layers.{layer}.post_feedforward_layernorm.weight"] = [config["hidden_size"]]
  return mapping
