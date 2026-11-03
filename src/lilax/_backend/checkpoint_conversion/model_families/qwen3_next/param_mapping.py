"""Qwen3-Next parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mapping from MaxText to HuggingFace Qwen3-Next weight paths."""
  num_main_layers = config["num_hidden_layers"]
  num_experts = config["num_experts"]
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  shared_specs = [
      ("input_layernorm-scale", "input_layernorm.weight"),
      ("post_attention_layernorm-scale", "post_attention_layernorm.weight"),
      ("mlp-routed_experts-gate-kernel", "mlp.gate.weight"),
      ("mlp-shared_expert-wi_0-kernel", "mlp.shared_expert.gate_proj.weight"),
      ("mlp-shared_expert-wi_1-kernel", "mlp.shared_expert.up_proj.weight"),
      ("mlp-shared_expert-wo-kernel", "mlp.shared_expert.down_proj.weight"),
      ("mlp-shared_expert_gate-kernel", "mlp.shared_expert_gate.weight"),
  ]
  local_specs = shared_specs + [
      ("attention-in_proj_qkvz-kernel", "linear_attn.in_proj_qkvz.weight"),
      ("attention-in_proj_ba-kernel", "linear_attn.in_proj_ba.weight"),
      ("attention-conv1d-kernel", "linear_attn.conv1d.weight"),
      ("attention-A_log", "linear_attn.A_log"),
      ("attention-dt_bias", "linear_attn.dt_bias"),
      ("attention-norm-rms_norm-scale", "linear_attn.norm.weight"),
      ("attention-out_proj-kernel", "linear_attn.out_proj.weight"),
  ]
  global_specs = shared_specs + [
      ("attention-attention-query-kernel", "self_attn.q_proj.weight"),
      ("attention-attention-key-kernel", "self_attn.k_proj.weight"),
      ("attention-attention-value-kernel", "self_attn.v_proj.weight"),
      ("attention-attention-out-kernel", "self_attn.o_proj.weight"),
      ("attention-attention-query_norm-scale", "self_attn.q_norm.weight"),
      ("attention-attention-key_norm-scale", "self_attn.k_norm.weight"),
  ]
  expert_specs = [
      ("mlp-routed_experts-wi_0", "gate_proj.weight"),
      ("mlp-routed_experts-wi_1", "up_proj.weight"),
      ("mlp-routed_experts-wo", "down_proj.weight"),
  ]
  if scan_layers:
    num_blocks = num_main_layers // layer_cycle_interval
    local_positions = list(range(layer_cycle_interval - 1))
    global_position = layer_cycle_interval - 1

    def hf_layer(block_idx, position, suffix):
      return f"model.layers.{block_idx * layer_cycle_interval + position}.{suffix}"

    local_prefix = "params-decoder-layers-local_layers"
    for subkey, suffix in local_specs:
      mapping[f"{local_prefix}-{subkey}"] = [
          [hf_layer(b, p, suffix) for p in local_positions] for b in range(num_blocks)
      ]
    for subkey, suffix in expert_specs:
      mapping[f"{local_prefix}-{subkey}"] = [
          [[hf_layer(b, p, f"mlp.experts.{e}.{suffix}") for p in local_positions] for b in range(num_blocks)]
          for e in range(num_experts)
      ]
    global_prefix = "params-decoder-layers-global_layer"
    for subkey, suffix in global_specs:
      mapping[f"{global_prefix}-{subkey}"] = [hf_layer(b, global_position, suffix) for b in range(num_blocks)]
    for subkey, suffix in expert_specs:
      mapping[f"{global_prefix}-{subkey}"] = [
          [hf_layer(b, global_position, f"mlp.experts.{e}.{suffix}") for b in range(num_blocks)]
          for e in range(num_experts)
      ]
    remainder_positions = range(num_main_layers - num_blocks * layer_cycle_interval)
    if remainder_positions:
      remainder_prefix = "params-decoder-layers_remainder-local_layers"
      for subkey, suffix in local_specs:
        mapping[f"{remainder_prefix}-{subkey}"] = [hf_layer(num_blocks, p, suffix) for p in remainder_positions]
      for subkey, suffix in expert_specs:
        mapping[f"{remainder_prefix}-{subkey}"] = [
            [hf_layer(num_blocks, p, f"mlp.experts.{e}.{suffix}") for p in remainder_positions]
            for e in range(num_experts)
        ]
  else:
    for i in range(num_main_layers):
      prefix = f"params-decoder-layers_{i}"
      is_full_attention_layer = (i % layer_cycle_interval + 1) % layer_cycle_interval == 0
      for subkey, suffix in global_specs if is_full_attention_layer else local_specs:
        mapping[f"{prefix}-{subkey}"] = f"model.layers.{i}.{suffix}"
      for subkey, suffix in expert_specs:
        mapping[f"{prefix}-{subkey}"] = [f"model.layers.{i}.mlp.experts.{e}.{suffix}" for e in range(num_experts)]
  return mapping


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3-Next."""
  def transpose(input_tensor, target_shape=None):
    return input_tensor.T

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    return input_tensor.T.reshape(target_shape)

  def permute_conv(input_tensor, target_shape=None):
    return input_tensor.transpose(2, 1, 0)

  hooks = {"params-decoder-logits_dense-kernel": transpose}
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  num_main_layers = config["num_hidden_layers"]
  if scan_layers:
    layer_prefixes = [
        ("params-decoder-layers-local_layers", False),
        ("params-decoder-layers-global_layer", True),
    ]
    if num_main_layers % layer_cycle_interval:
      layer_prefixes.append(("params-decoder-layers_remainder-local_layers", False))
  else:
    layer_prefixes = [
        (f"params-decoder-layers_{i}", (i % layer_cycle_interval + 1) % layer_cycle_interval == 0)
        for i in range(num_main_layers)
    ]
  for prefix, is_full_attention_layer in layer_prefixes:
    if is_full_attention_layer:
      for key in ["query", "key", "value", "out"]:
        hooks[f"{prefix}-attention-attention-{key}-kernel"] = reshape_kernel
    else:
      hooks[f"{prefix}-attention-in_proj_qkvz-kernel"] = transpose
      hooks[f"{prefix}-attention-in_proj_ba-kernel"] = transpose
      hooks[f"{prefix}-attention-out_proj-kernel"] = transpose
      hooks[f"{prefix}-attention-conv1d-kernel"] = permute_conv
    mlp_prefix = f"{prefix}-mlp"
    hooks[f"{mlp_prefix}-routed_experts-gate-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_0-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_1-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wo-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert_gate-kernel"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wi_0"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wi_1"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wo"] = transpose
  return hooks
