"""Hugging Face Mixtral weight shapes for checkpoint conversion."""


def MIXTRAL_HF_WEIGHTS_TO_SHAPE(config):
  """Returns a mapping of Hugging Face parameter names to their tensor shapes."""
  shapes = {}
  shapes["model.embed_tokens.weight"] = [config["vocab_size"], config["hidden_size"]]
  shapes["lm_head.weight"] = [config["vocab_size"], config["hidden_size"]]
  shapes["model.norm.weight"] = [config["hidden_size"]]
  head_dim = config["hidden_size"] // config["num_attention_heads"]
  kv_dim = config["num_key_value_heads"] * head_dim
  for i in range(config["num_hidden_layers"]):
    shapes[f"model.layers.{i}.self_attn.q_proj.weight"] = [config["hidden_size"], config["hidden_size"]]
    shapes[f"model.layers.{i}.self_attn.k_proj.weight"] = [kv_dim, config["hidden_size"]]
    shapes[f"model.layers.{i}.self_attn.v_proj.weight"] = [kv_dim, config["hidden_size"]]
    shapes[f"model.layers.{i}.self_attn.o_proj.weight"] = [config["hidden_size"], config["hidden_size"]]
    shapes[f"model.layers.{i}.input_layernorm.weight"] = [config["hidden_size"]]
    shapes[f"model.layers.{i}.post_attention_layernorm.weight"] = [config["hidden_size"]]
    shapes[f"model.layers.{i}.block_sparse_moe.gate.weight"] = [config["num_local_experts"], config["hidden_size"]]
    for j in range(config["num_local_experts"]):
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = [
          config["intermediate_size"], config["hidden_size"]
      ]
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = [
          config["hidden_size"], config["intermediate_size"]
      ]
      shapes[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = [
          config["intermediate_size"], config["hidden_size"]
      ]
  return shapes
