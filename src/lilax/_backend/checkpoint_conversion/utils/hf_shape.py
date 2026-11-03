"""Hugging Face shape checkpoint conversion utils."""

import math


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


def LLAMA31_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and weights shape."""
  mapping = {
      "model.embed_tokens.weight": [config["vocab_size"], config["hidden_size"]],
      "model.norm.weight": [config["hidden_size"]],
      "lm_head.weight": [config["vocab_size"], config["hidden_size"]],
  }
  for layer_idx in range(config["num_hidden_layers"]):
    layer_mapping = {
        f"model.layers.{layer_idx}.input_layernorm.weight": [config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.down_proj.weight": [config["hidden_size"], config["intermediate_size"]],
        f"model.layers.{layer_idx}.mlp.up_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": [config["intermediate_size"], config["hidden_size"]],
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": [config["hidden_size"]],
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


def QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen3-Next weights path and their shape."""
  hidden_size = config["hidden_size"]
  num_hidden_layers = config["num_hidden_layers"]
  vocab_size = config["vocab_size"]
  num_attention_heads = config["num_attention_heads"]
  num_key_value_heads = config["num_key_value_heads"]
  num_experts = config["num_experts"]
  head_dim = config["head_dim"]
  linear_conv_kernel_dim = config["linear_conv_kernel_dim"]
  linear_key_head_dim = config["linear_key_head_dim"]
  linear_num_key_heads = config["linear_num_key_heads"]
  linear_num_value_heads = config["linear_num_value_heads"]
  moe_intermediate_size = config["moe_intermediate_size"]
  shared_expert_intermediate_size = config["shared_expert_intermediate_size"]
  cycle_interval = config["full_attention_interval"]
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim
  linear_k_dim = linear_num_key_heads * linear_key_head_dim
  linear_v_dim = linear_num_value_heads * head_dim
  conv_dim = 2 * linear_k_dim + linear_v_dim
  qkvz_dim = 2 * linear_k_dim + 2 * linear_v_dim
  ba_dim = 2 * linear_num_value_heads
  mapping = {
      "model.embed_tokens.weight": [vocab_size, hidden_size],
      "model.norm.weight": [hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
  }
  for layer_idx in range(num_hidden_layers):
    layer_prefix = f"model.layers.{layer_idx}"
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
          f"{layer_prefix}.linear_attn.in_proj_qkvz.weight": [qkvz_dim, hidden_size],
          f"{layer_prefix}.linear_attn.in_proj_ba.weight": [ba_dim, hidden_size],
          f"{layer_prefix}.linear_attn.conv1d.weight": [conv_dim, 1, linear_conv_kernel_dim],
          f"{layer_prefix}.linear_attn.A_log": [linear_num_value_heads],
          f"{layer_prefix}.linear_attn.dt_bias": [linear_num_value_heads],
          f"{layer_prefix}.linear_attn.norm.weight": [head_dim],
          f"{layer_prefix}.linear_attn.out_proj.weight": [hidden_size, linear_v_dim],
      })
    mapping.update({
        f"{layer_prefix}.mlp.gate.weight": [num_experts, hidden_size],
        f"{layer_prefix}.mlp.shared_expert.gate_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{layer_prefix}.mlp.shared_expert.up_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{layer_prefix}.mlp.shared_expert.down_proj.weight": [hidden_size, shared_expert_intermediate_size],
        f"{layer_prefix}.mlp.shared_expert_gate.weight": [1, hidden_size],
    })
    for expert_idx in range(num_experts):
      mapping.update({
          f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj.weight": [moe_intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj.weight": [moe_intermediate_size, hidden_size],
          f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj.weight": [hidden_size, moe_intermediate_size],
      })


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


def QWEN3_8_HF_WEIGHTS_TO_SHAPE(config):
  """Returns HF weight shapes for the text-only Qwen3.8 checkpoint."""
  return QWEN3_5_HF_WEIGHTS_TO_SHAPE(config, hf_prefix="model")


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


def QWEN3_VL_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen3-VL weights path and weights shape."""
  text_shapes = QWEN_HF_WEIGHTS_TO_SHAPE(config["text_config"])
  vl_shapes = {}
  for key, value in text_shapes.items():
    if key.startswith("model."):
      new_key = key.replace("model.", "model.language_model.")
      vl_shapes[new_key] = value
    else:
      vl_shapes[key] = value
  vision_config = config["vision_config"]
  v_depth = vision_config["depth"]
  v_hidden_size = vision_config["hidden_size"]
  v_intermediate_size = vision_config["intermediate_size"]
  v_out_hidden_size = vision_config["out_hidden_size"]
  vl_shapes["model.visual.patch_embed.proj.weight"] = [v_hidden_size, 3, 2, 16, 16]
  vl_shapes["model.visual.patch_embed.proj.bias"] = [v_hidden_size]
  vl_shapes["model.visual.pos_embed.weight"] = [vision_config["num_position_embeddings"], v_hidden_size]
  for layer_idx in range(v_depth):
    prefix = f"model.visual.blocks.{layer_idx}"
    vl_shapes[f"{prefix}.norm1.weight"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm1.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm2.weight"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm2.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.attn.qkv.weight"] = [3 * v_hidden_size, v_hidden_size]
    vl_shapes[f"{prefix}.attn.qkv.bias"] = [3 * v_hidden_size]
    vl_shapes[f"{prefix}.attn.proj.weight"] = [v_hidden_size, v_hidden_size]
    vl_shapes[f"{prefix}.attn.proj.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.mlp.linear_fc1.weight"] = [v_intermediate_size, v_hidden_size]
    vl_shapes[f"{prefix}.mlp.linear_fc1.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.mlp.linear_fc2.weight"] = [v_hidden_size, v_intermediate_size]
    vl_shapes[f"{prefix}.mlp.linear_fc2.bias"] = [v_hidden_size]
  deepstack_indexes = vision_config.get("deepstack_visual_indexes", [5, 11, 17])
  for merger_idx, _ in enumerate(deepstack_indexes):
    prefix = f"model.visual.deepstack_merger_list.{merger_idx}"
    vl_shapes[f"{prefix}.norm.weight"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.norm.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc1.weight"] = [v_intermediate_size, v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc1.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc2.weight"] = [v_out_hidden_size, v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc2.bias"] = [v_out_hidden_size]
  vl_shapes["model.visual.merger.norm.weight"] = [v_hidden_size]
  vl_shapes["model.visual.merger.norm.bias"] = [v_hidden_size]
  vl_shapes["model.visual.merger.linear_fc1.weight"] = [v_intermediate_size, v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc1.bias"] = [v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc2.weight"] = [v_out_hidden_size, v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc2.bias"] = [v_out_hidden_size]
  return vl_shapes


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


def _qwen3_8_flash_next_is_prime(number: int) -> bool:
  if number < 2:
    return False
  if number < 4:
    return True
  if number % 2 == 0 or number % 3 == 0:
    return False
  divisor = 5
  while divisor * divisor <= number:
    if number % divisor == 0 or number % (divisor + 2) == 0:
      return False
    divisor += 6
  return True


def _qwen3_8_flash_next_nth_prime_after(start: int, nth: int) -> int:
  candidate = start
  found = 0
  while found < nth:
    candidate += 1
    if _qwen3_8_flash_next_is_prime(candidate):
      found += 1
  return candidate


def QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE(config):
  if hasattr(config, "to_dict"):
    config = config.to_dict()
  tcfg = (
      config.get("text_config", config)
      if isinstance(config, dict) and isinstance(config.get("text_config"), dict) else config
  )
  if (isinstance(config, dict) and not config.get("language_model_only", False)
      and ("text_config" in config or "vision_config" in config)):
    text_base = "model.language_model"
  else:
    text_base = "model"
  hidden_size = tcfg["hidden_size"]
  vocab_size = tcfg["vocab_size"]
  num_hidden_layers = tcfg["num_hidden_layers"]
  num_attention_heads = tcfg["num_attention_heads"]
  num_key_value_heads = tcfg["num_key_value_heads"]
  head_dim = tcfg["head_dim"]
  hc_count = tcfg.get("hc_count", 4)
  hc_lowrank = tcfg.get("hc_lowrank", 320)
  hc_dim = hc_count * hidden_size
  moe_intermediate_size = tcfg["moe_intermediate_size"]
  shared_expert_intermediate_size = tcfg["shared_expert_intermediate_size"]
  num_experts = tcfg["num_experts"]
  linear_num_key_heads = tcfg["linear_num_key_heads"]
  linear_num_value_heads = tcfg["linear_num_value_heads"]
  linear_key_head_dim = tcfg["linear_key_head_dim"]
  linear_value_head_dim = tcfg["linear_value_head_dim"]
  linear_conv_kernel_dim = tcfg.get("linear_conv_kernel_dim", 4)
  indexer_n_heads = tcfg.get("indexer_n_heads", 4)
  indexer_kv_heads = tcfg.get("indexer_kv_heads", 1)
  indexer_head_dim = tcfg.get("indexer_head_dim", 128)
  layer_types = tcfg.get("layer_types") or [
      "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention" for layer_idx in range(num_hidden_layers)
  ]
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim
  gdn_k_dim = linear_num_key_heads * linear_key_head_dim
  gdn_v_dim = linear_num_value_heads * linear_value_head_dim
  conv_dim = 2 * gdn_k_dim + gdn_v_dim
  mapping = {
      f"{text_base}.embed_tokens.weight": [vocab_size, hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
      f"{text_base}.hyper_connection_mixer.hc_norm.weight": [hc_dim],
      f"{text_base}.hyper_connection_mixer.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
      f"{text_base}.hyper_connection_mixer.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
  }

  def _layer_shapes(prefix: str, is_qsa: bool) -> dict:
    layer_map = {
        f"{prefix}.mlp.gate.weight": [num_experts, hidden_size],
        f"{prefix}.mlp.experts.gate_up_proj": [num_experts, 2 * moe_intermediate_size, hidden_size],
        f"{prefix}.mlp.experts.down_proj": [num_experts, hidden_size, moe_intermediate_size],
        f"{prefix}.mlp.shared_expert.gate_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{prefix}.mlp.shared_expert.up_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{prefix}.mlp.shared_expert.down_proj.weight": [hidden_size, shared_expert_intermediate_size],
        f"{prefix}.mlp.shared_expert_gate.weight": [1, hidden_size],
    }
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
      layer_map.update({
          f"{prefix}.{hc}.hc_norm.weight": [hc_dim],
          f"{prefix}.{hc}.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
          f"{prefix}.{hc}.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
          f"{prefix}.{hc}.block_inject_weight.weight": [hc_count, hc_dim],
      })
    if is_qsa:
      layer_map.update({
          f"{prefix}.self_attn.q_proj.weight": [2 * q_dim, hidden_size],
          f"{prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
          f"{prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
          f"{prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
          f"{prefix}.self_attn.q_norm.weight": [head_dim],
          f"{prefix}.self_attn.k_norm.weight": [head_dim],
          f"{prefix}.self_attn.indexer.index_qk_proj.weight": [
              (indexer_n_heads + indexer_kv_heads) * indexer_head_dim, hidden_size
          ],
          f"{prefix}.self_attn.indexer.q_layernorm.weight": [indexer_head_dim],
          f"{prefix}.self_attn.indexer.k_layernorm.weight": [indexer_head_dim],
      })
    else:
      layer_map.update({
          f"{prefix}.linear_attn.A_log": [linear_num_value_heads],
          f"{prefix}.linear_attn.dt_bias": [linear_num_value_heads],
          f"{prefix}.linear_attn.conv1d.weight": [conv_dim, 1, linear_conv_kernel_dim],
          f"{prefix}.linear_attn.in_proj_qkv.weight": [conv_dim, hidden_size],
          f"{prefix}.linear_attn.in_proj_z.weight": [gdn_v_dim, hidden_size],
          f"{prefix}.linear_attn.in_proj_b.weight": [linear_num_value_heads, hidden_size],
          f"{prefix}.linear_attn.in_proj_a.weight": [linear_num_value_heads, hidden_size],
          f"{prefix}.linear_attn.norm.weight": [linear_value_head_dim],
          f"{prefix}.linear_attn.out_proj.weight": [hidden_size, gdn_v_dim],
      })
    return layer_map

  for layer_idx in range(num_hidden_layers):
    mapping.update(_layer_shapes(f"{text_base}.layers.{layer_idx}", is_qsa=layer_types[layer_idx] == "full_attention"))
  ple_layer_ids = tcfg.get("ple_layer_ids", [2])
  if tcfg.get("use_ple_layer", bool(ple_layer_ids)):
    default_pid = (int(ple_layer_ids[0]) - 1) if ple_layer_ids else 1
    ple_layer_id = tcfg.get("ple_layer_id", default_pid)
    ple_pfx = f"{text_base}.layers.{ple_layer_id}.ple"
    ple_embed_dim = tcfg.get("ple_embed_dim", hidden_size)
    ple_conv_kernel_size = tcfg.get("ple_conv_kernel_size", 4)
    ngram_size = tcfg.get("ngram_size", 3)
    heads_per_ngram = tcfg.get("heads_per_ngram", 8)
    ngram_vocab_size_base = tcfg.get("ngram_vocab_size_base", 20000000)
    split_ngram_parts = max(1, int(tcfg.get("split_ngram_parts", 128)))
    divisible_by = tcfg.get("make_ngram_vocab_size_divisible_by", 128)
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    head_dim_per_ngram = ple_embed_dim // ngram_heads
    total_vocab = sum(_qwen3_8_flash_next_nth_prime_after(ngram_vocab_size_base - 1, head + 1)
                      for head in range(ngram_heads))
    padded_vocab_size = math.ceil(total_vocab / divisible_by) * divisible_by
    shard_rows = padded_vocab_size // split_ngram_parts
    mapping.update({
        f"{ple_pfx}.conv1d.weight": [hc_dim, 1, ple_conv_kernel_size],
        f"{ple_pfx}.key_proj.weight": [hc_dim, ple_embed_dim],
        f"{ple_pfx}.value_proj.weight": [hidden_size, ple_embed_dim],
        f"{ple_pfx}.norm_key.weight": [hc_dim],
        f"{ple_pfx}.norm_query.weight": [hc_dim],
        f"{ple_pfx}.norm_conv.weight": [hc_dim],
        f"{ple_pfx}.ple_embedding.ngram_embedding.weight": [padded_vocab_size, head_dim_per_ngram],
    })
    shard_keys = tuple(f"{ple_pfx}.ple_embedding.ngram_embedding.shard_{shard}.weight"
                       for shard in range(split_ngram_parts))
    for shard_key in shard_keys:
      mapping[shard_key] = [shard_rows, head_dim_per_ngram]
    mapping[shard_keys] = tuple([shard_rows, head_dim_per_ngram] for _ in range(split_ngram_parts))
  mtp_num_layers = tcfg.get("mtp_num_hidden_layers", tcfg.get("mtp_num_layers", 1))
  if mtp_num_layers and mtp_num_layers > 0:
    mapping.update({
        "mtp.pre_fc_norm_embedding.weight": [hidden_size],
        "mtp.pre_fc_norm_hidden.weight": [hc_dim],
        "mtp.fc_embedding.weight": [hidden_size, hidden_size],
        "mtp.fc_hidden.weight": [hidden_size, hidden_size],
        "mtp.hyper_connection_mixer.hc_norm.weight": [hc_dim],
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
    })
    for mtp_idx in range(mtp_num_layers):
      mapping.update(_layer_shapes(f"mtp.layers.{mtp_idx}", is_qsa=True))
  return mapping


def _get_gemma4_layer_attention_dims(text_cfg: dict, layer_idx: int, is_global: bool):
  num_attention_heads = text_cfg["num_attention_heads"]
  num_key_value_heads = text_cfg["num_key_value_heads"]
  head_dim = text_cfg["head_dim"]
  global_head_dim = text_cfg.get("global_head_dim") or head_dim
  num_global_key_value_heads = text_cfg.get("num_global_key_value_heads") or num_key_value_heads
  per_layer_config = text_cfg.get("per_layer_config") or {}
  if isinstance(per_layer_config, list):
    layer_override = (
        per_layer_config[layer_idx]
        if layer_idx < len(per_layer_config) and isinstance(per_layer_config[layer_idx], dict) else {}
    )
  elif isinstance(per_layer_config, dict):
    layer_override = (per_layer_config.get(layer_idx) or per_layer_config.get(str(layer_idx))
                      or per_layer_config.get(f"{layer_idx:02d}") or {})
  else:
    layer_override = {}
  layer_heads = layer_override.get("num_attention_heads") or num_attention_heads
  layer_kv_heads = layer_override.get("num_key_value_heads") or (
      num_global_key_value_heads if is_global else num_key_value_heads
  )
  layer_head_dim = layer_override.get("head_dim") or (global_head_dim if is_global else head_dim)
  return layer_heads * layer_head_dim, layer_kv_heads * layer_head_dim, layer_head_dim


def GEMMA4_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_cfg = config.get("text_config", config)
  vision_cfg = config.get("vision_config", {})
  text_base = "model.language_model" if vision_cfg else "model"
  hidden_size = text_cfg["hidden_size"]
  intermediate_size = text_cfg["intermediate_size"]
  num_hidden_layers = text_cfg["num_hidden_layers"]
  vocab_size = text_cfg["vocab_size"]
  num_experts = text_cfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  expert_intermediate_size = text_cfg.get("moe_intermediate_size") or text_cfg.get("expert_intermediate_size")
  layer_types = text_cfg.get("layer_types", [])
  shapes[f"{text_base}.embed_tokens.weight"] = [vocab_size, hidden_size]
  shapes[f"{text_base}.norm.weight"] = [hidden_size]
  for layer_idx in range(num_hidden_layers):
    prefix = f"{text_base}.layers.{layer_idx}"
    is_global = (layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention") if layer_types else (
        layer_idx % 6 == 5
    )
    q_dim, kv_dim, norm_dim = _get_gemma4_layer_attention_dims(text_cfg, layer_idx, is_global)
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.self_attn.k_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.self_attn.v_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.layer_scalar"] = [1]
    if num_experts > 1:
      shapes[f"{prefix}.pre_feedforward_layernorm_2.weight"] = [hidden_size]
      shapes[f"{prefix}.post_feedforward_layernorm_1.weight"] = [hidden_size]
      shapes[f"{prefix}.post_feedforward_layernorm_2.weight"] = [hidden_size]
      shapes[f"{prefix}.router.scale"] = [hidden_size]
      shapes[f"{prefix}.router.proj.weight"] = [num_experts, hidden_size]
      shapes[f"{prefix}.router.per_expert_scale"] = [num_experts]
      shapes[f"{prefix}.experts.gate_up_proj"] = [num_experts, 2 * expert_intermediate_size, hidden_size]
      shapes[f"{prefix}.experts.down_proj"] = [num_experts, hidden_size, expert_intermediate_size]
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [intermediate_size, hidden_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [intermediate_size, hidden_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [hidden_size, intermediate_size]
  if vision_cfg:
    vis_hidden = vision_cfg["hidden_size"]
    vis_intermediate = vision_cfg["intermediate_size"]
    vis_num_layers = vision_cfg["num_hidden_layers"]
    vis_num_heads = vision_cfg["num_attention_heads"]
    vis_head_dim = vision_cfg["head_dim"]
    vis_q_dim = vis_num_heads * vis_head_dim
    vis_kv_dim = vision_cfg.get("num_key_value_heads", vis_num_heads) * vis_head_dim
    vis_pos_emb_size = vision_cfg.get("position_embedding_size", 10240)
    vis_patch_size = vision_cfg.get("patch_size", 16)
    patch_flat = 3 * vis_patch_size * vis_patch_size
    shapes["model.vision_tower.patch_embedder.input_proj.weight"] = [vis_hidden, patch_flat]
    shapes["model.vision_tower.patch_embedder.position_embedding_table"] = [2, vis_pos_emb_size, vis_hidden]
    shapes["model.vision_tower.std_scale"] = [vis_hidden]
    shapes["model.vision_tower.std_bias"] = [vis_hidden]
    shapes["model.embed_vision.embedding_projection.weight"] = [hidden_size, vis_hidden]
    for layer_idx in range(vis_num_layers):
      prefix = f"model.vision_tower.encoder.layers.{layer_idx}"
      shapes[f"{prefix}.self_attn.q_proj.linear.weight"] = [vis_q_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.k_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.v_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.o_proj.linear.weight"] = [vis_hidden, vis_q_dim]
      shapes[f"{prefix}.self_attn.q_norm.weight"] = [vis_head_dim]
      shapes[f"{prefix}.self_attn.k_norm.weight"] = [vis_head_dim]
      shapes[f"{prefix}.input_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.post_attention_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.mlp.gate_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{prefix}.mlp.up_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{prefix}.mlp.down_proj.linear.weight"] = [vis_hidden, vis_intermediate]
  return shapes


def GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_cfg = config.get("text_config", config)
  vision_cfg = config.get("vision_config", {})
  text_base = "model.language_model" if vision_cfg else "model"
  hidden_size = text_cfg["hidden_size"]
  intermediate_size = text_cfg["intermediate_size"]
  num_hidden_layers = text_cfg["num_hidden_layers"]
  vocab_size = text_cfg["vocab_size"]
  layer_types = text_cfg.get("layer_types", [])
  ple_dim = text_cfg.get("hidden_size_per_layer_input", 0) or 0
  vocab_ple = text_cfg.get("vocab_size_per_layer_input", 0) or 0
  num_kv_shared = text_cfg.get("num_kv_shared_layers", 0) or 0
  first_shared = max(0, num_hidden_layers - num_kv_shared) if num_kv_shared > 0 else num_hidden_layers
  use_double_wide_mlp = bool(text_cfg.get("use_double_wide_mlp", False))
  shapes[f"{text_base}.embed_tokens.weight"] = [vocab_size, hidden_size]
  shapes[f"{text_base}.norm.weight"] = [hidden_size]
  if ple_dim > 0:
    shapes[f"{text_base}.embed_tokens_per_layer.weight"] = [vocab_ple, num_hidden_layers * ple_dim]
    shapes[f"{text_base}.per_layer_model_projection.weight"] = [num_hidden_layers * ple_dim, hidden_size]
    shapes[f"{text_base}.per_layer_projection_norm.weight"] = [ple_dim]
  for layer_idx in range(num_hidden_layers):
    prefix = f"{text_base}.layers.{layer_idx}"
    is_global = layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention"
    is_shared = num_kv_shared > 0 and layer_idx >= first_shared
    q_dim, kv_dim, norm_dim = _get_gemma4_layer_attention_dims(text_cfg, layer_idx, is_global)
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [norm_dim]
    if not is_shared:
      shapes[f"{prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
      shapes[f"{prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
      shapes[f"{prefix}.self_attn.k_norm.weight"] = [norm_dim]
      shapes[f"{prefix}.self_attn.v_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.layer_scalar"] = [1]
    mlp_dim = intermediate_size * 2 if (is_shared and use_double_wide_mlp) else intermediate_size
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [mlp_dim, hidden_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [mlp_dim, hidden_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [hidden_size, mlp_dim]
    if ple_dim > 0:
      shapes[f"{prefix}.per_layer_input_gate.weight"] = [ple_dim, hidden_size]
      shapes[f"{prefix}.per_layer_projection.weight"] = [hidden_size, ple_dim]
      shapes[f"{prefix}.post_per_layer_input_norm.weight"] = [hidden_size]
  return shapes


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


def GEMMA3_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_config = config["text_config"]
  vision_config = config["vision_config"]
  lm_hidden_size = text_config["hidden_size"]
  lm_intermediate_size = text_config["intermediate_size"]
  lm_num_layers = text_config["num_hidden_layers"]
  lm_head_dim = text_config["head_dim"]
  lm_q_dim = text_config["num_attention_heads"] * lm_head_dim
  lm_kv_dim = text_config["num_key_value_heads"] * lm_head_dim
  vision_hidden_size = vision_config["hidden_size"]
  vision_intermediate_size = vision_config["intermediate_size"]
  vision_num_layers = vision_config["num_hidden_layers"]
  vision_patch_size = vision_config["patch_size"]
  vision_num_channels = vision_config["num_channels"]
  vision_num_positions = (vision_config["image_size"] // vision_patch_size) ** 2
  vocab_size = text_config["vocab_size"]

  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = [
      vision_hidden_size, vision_num_channels, vision_patch_size, vision_patch_size
  ]
  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.bias"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.embeddings.position_embedding.weight"] = [
      vision_num_positions, vision_hidden_size
  ]
  for layer_idx in range(vision_num_layers):
    prefix = f"model.vision_tower.vision_model.encoder.layers.{layer_idx}"
    shapes[f"{prefix}.layer_norm1.weight"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm1.bias"] = [vision_hidden_size]
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
      shapes[f"{prefix}.self_attn.{projection}.weight"] = [vision_hidden_size, vision_hidden_size]
      shapes[f"{prefix}.self_attn.{projection}.bias"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm2.weight"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm2.bias"] = [vision_hidden_size]
    shapes[f"{prefix}.mlp.fc1.weight"] = [vision_intermediate_size, vision_hidden_size]
    shapes[f"{prefix}.mlp.fc1.bias"] = [vision_intermediate_size]
    shapes[f"{prefix}.mlp.fc2.weight"] = [vision_hidden_size, vision_intermediate_size]
    shapes[f"{prefix}.mlp.fc2.bias"] = [vision_hidden_size]

  shapes["model.vision_tower.vision_model.post_layernorm.weight"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.post_layernorm.bias"] = [vision_hidden_size]
  shapes["model.multi_modal_projector.mm_input_projection_weight"] = [vision_hidden_size, lm_hidden_size]
  shapes["model.multi_modal_projector.mm_soft_emb_norm.weight"] = [vision_hidden_size]
  shapes["model.language_model.embed_tokens.weight"] = [vocab_size, lm_hidden_size]
  for layer_idx in range(lm_num_layers):
    prefix = f"model.language_model.layers.{layer_idx}"
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [lm_q_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.q_proj.bias"] = [lm_q_dim]
    shapes[f"{prefix}.self_attn.k_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.k_proj.bias"] = [lm_kv_dim]
    shapes[f"{prefix}.self_attn.v_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.v_proj.bias"] = [lm_kv_dim]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [lm_hidden_size, lm_q_dim]
    shapes[f"{prefix}.self_attn.o_proj.bias"] = [lm_hidden_size]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [lm_head_dim]
    shapes[f"{prefix}.self_attn.k_norm.weight"] = [lm_head_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"{prefix}.mlp.gate_proj.bias"] = [lm_intermediate_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"{prefix}.mlp.up_proj.bias"] = [lm_intermediate_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [lm_hidden_size, lm_intermediate_size]
    shapes[f"{prefix}.mlp.down_proj.bias"] = [lm_hidden_size]
  shapes["model.language_model.norm.weight"] = [lm_hidden_size]
  shapes["lm_head.weight"] = [vocab_size, lm_hidden_size]
  return shapes


def DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE(config):
  hidden_size = config["hidden_size"]
  vocab_size = config["vocab_size"]
  num_hidden_layers = config["num_hidden_layers"]
  q_lora_rank = config.get("q_lora_rank", 1024)
  o_lora_rank = config.get("o_lora_rank", 1024) * config.get("o_groups", 8)
  num_attention_heads = config["num_attention_heads"]
  head_dim = config["head_dim"]
  moe_intermediate_size = config["moe_intermediate_size"]
  n_routed_experts = config["n_routed_experts"]
  hc_mult = config.get("hc_mult", config.get("mhc_expansion_rate", 4))
  hc_dim = (2 * hc_mult) + (hc_mult**2)
  num_hash_layers = config.get("num_hash_layers", config.get("first_num_hash_layers", 3))
  index_n_heads = config.get("index_n_heads", config.get("indexer_n_heads", 64))
  index_head_dim = config.get("index_head_dim", config.get("indexer_head_dim", 128))

  mapping = {
      "embed.weight": [vocab_size, hidden_size],
      "norm.weight": [hidden_size],
      "head.weight": [vocab_size, hidden_size],
      "hc_head_fn": [hc_mult, hidden_size * hc_mult],
      "hc_head_base": [hc_mult],
      "hc_head_scale": [1],
  }

  def _compressor_shapes(comp_prefix, comp_dim, norm_dim, ratio):
    return {
        f"{comp_prefix}.wgate.weight": [comp_dim, hidden_size],
        f"{comp_prefix}.wkv.weight": [comp_dim, hidden_size],
        f"{comp_prefix}.norm.weight": [norm_dim],
        f"{comp_prefix}.ape": [ratio, comp_dim],
    }

  def _layer_shapes(layer_prefix, is_hash, ratio):
    layer_mapping = {
        f"{layer_prefix}.attn_norm.weight": [hidden_size],
        f"{layer_prefix}.ffn_norm.weight": [hidden_size],
        f"{layer_prefix}.attn.wq_a.weight": [q_lora_rank, hidden_size],
        f"{layer_prefix}.attn.q_norm.weight": [q_lora_rank],
        f"{layer_prefix}.attn.wq_b.weight": [num_attention_heads * head_dim, q_lora_rank],
        f"{layer_prefix}.attn.wkv.weight": [
            config.get("num_key_value_heads", 1) * config.get("head_dim", 64), hidden_size
        ],
        f"{layer_prefix}.attn.kv_norm.weight": [head_dim],
        f"{layer_prefix}.attn.attn_sink": [num_attention_heads],
        f"{layer_prefix}.attn.wo_a.weight": [
            o_lora_rank, (num_attention_heads * head_dim) // config.get("o_groups", 8)
        ],
        f"{layer_prefix}.attn.wo_b.weight": [hidden_size, o_lora_rank],
        f"{layer_prefix}.hc_attn_fn": [hc_dim, hidden_size * hc_mult],
        f"{layer_prefix}.hc_attn_base": [hc_dim],
        f"{layer_prefix}.hc_attn_scale": [3],
        f"{layer_prefix}.hc_ffn_fn": [hc_dim, hidden_size * hc_mult],
        f"{layer_prefix}.hc_ffn_base": [hc_dim],
        f"{layer_prefix}.hc_ffn_scale": [3],
        f"{layer_prefix}.ffn.gate.weight": [n_routed_experts, hidden_size],
        f"{layer_prefix}.ffn.gate.bias": [n_routed_experts],
    }
    if is_hash:
      layer_mapping[f"{layer_prefix}.ffn.gate.tid2eid"] = [vocab_size, config.get("num_experts_per_tok", 2)]
    if ratio > 0:
      comp_dim = 2 * head_dim if ratio == 4 else head_dim
      layer_mapping.update(_compressor_shapes(f"{layer_prefix}.attn.compressor", comp_dim, head_dim, ratio))
      if ratio == 4:
        layer_mapping.update(
            _compressor_shapes(f"{layer_prefix}.attn.indexer.compressor", 2 * index_head_dim, index_head_dim, ratio)
        )
        layer_mapping[f"{layer_prefix}.attn.indexer.weights_proj.weight"] = [index_n_heads, hidden_size]
        layer_mapping[f"{layer_prefix}.attn.indexer.wq_b.weight"] = [index_n_heads * index_head_dim, q_lora_rank]
    for exp_idx in range(n_routed_experts):
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w1.weight"] = [moe_intermediate_size, hidden_size]
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w2.weight"] = [hidden_size, moe_intermediate_size]
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w3.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w1.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w3.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w2.weight"] = [hidden_size, moe_intermediate_size]
    return layer_mapping

  compress_ratios = config["compress_ratios"]
  for layer_idx in range(num_hidden_layers):
    mapping.update(_layer_shapes(f"layers.{layer_idx}", is_hash=layer_idx < num_hash_layers,
                                 ratio=compress_ratios[layer_idx]))

  mtp_num_layers = config.get("num_nextn_predict_layers", config.get("num_mtp_layers", config.get("mtp_num_layers", 0)))
  for mtp_idx in range(mtp_num_layers):
    mtp_prefix = f"mtp.{mtp_idx}"
    mapping.update(_layer_shapes(mtp_prefix, is_hash=False, ratio=0))
    mapping.update({
        f"{mtp_prefix}.enorm.weight": [hidden_size],
        f"{mtp_prefix}.hnorm.weight": [hidden_size],
        f"{mtp_prefix}.norm.weight": [hidden_size],
        f"{mtp_prefix}.e_proj.weight": [hidden_size, hidden_size],
        f"{mtp_prefix}.h_proj.weight": [hidden_size, hidden_size],
        (f"{mtp_prefix}.e_proj.weight", f"{mtp_prefix}.h_proj.weight"): (
            [hidden_size, hidden_size], [hidden_size, hidden_size]
        ),
        f"{mtp_prefix}.hc_head_fn": [hc_mult, hidden_size * hc_mult],
        f"{mtp_prefix}.hc_head_base": [hc_mult],
        f"{mtp_prefix}.hc_head_scale": [1],
    })
  return mapping


HF_SHAPE = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma3-4b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-12b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-27b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma4-26b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-31b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e2b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e4b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-1.5b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-7b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-0.6b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b-thinking-2507": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-8b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-32b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b-Instruct": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-70b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-405b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b-base": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-235b-a22b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-480b-a35b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "deepseek2-16b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3.2-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek4-284b": DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE,
    "deepseek4-1600b": DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x7b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x22b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-20b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-120b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-9b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-a3b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-35b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-397b-a17b": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-2.4t-a95b": QWEN3_8_HF_WEIGHTS_TO_SHAPE,
    "qwen3-next-80b-a3b": QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-2b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-4b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "qwen3-vl-30b-a3b": QWEN3_VL_HF_WEIGHTS_TO_SHAPE,
    "hy4-preview": HY4_HF_WEIGHTS_TO_SHAPE,
}
