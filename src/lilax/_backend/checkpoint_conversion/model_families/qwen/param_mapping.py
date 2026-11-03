"""Qwen 2.5 / Qwen 3 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", config.get("num_local_experts", 0))
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  if scan_layers:
    mapping.update({
        "params-decoder-layers-pre_self_attention_layer_norm-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query-bias": [
            f"model.layers.{i}.self_attn.q_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key-bias": [
            f"model.layers.{i}.self_attn.k_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-value-bias": [
            f"model.layers.{i}.self_attn.v_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query_norm-scale": [
            f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key_norm-scale": [
            f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-post_self_attention_layer_norm-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
        ],
    })
    if num_experts > 1:
      mapping.update({
          "params-decoder-layers-moe_block-gate-kernel": [
              f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)
          ],
          "params-decoder-layers-moe_block-wi_0": [
              [f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
          "params-decoder-layers-moe_block-wi_1": [
              [f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
          "params-decoder-layers-moe_block-wo": [
              [f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
      })
    else:
      mapping.update({
          "params-decoder-layers-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)],
          "params-decoder-layers-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)],
          "params-decoder-layers-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)],
      })
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers-layers_{i}" if getattr(maxtext_config, "use_m3_model", False) else f"params-decoder-layers_{i}"
      mapping.update({
          f"{prefix}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
          f"{prefix}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
          f"{prefix}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
          f"{prefix}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
          f"{prefix}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
          f"{prefix}-self_attention-query-bias": f"model.layers.{i}.self_attn.q_proj.bias",
          f"{prefix}-self_attention-key-bias": f"model.layers.{i}.self_attn.k_proj.bias",
          f"{prefix}-self_attention-value-bias": f"model.layers.{i}.self_attn.v_proj.bias",
          f"{prefix}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
          f"{prefix}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
          f"{prefix}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
      })
      if num_experts > 1:
        mapping.update({
            f"{prefix}-moe_block-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
            f"{prefix}-moe_block-wi_0": [
                f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)
            ],
            f"{prefix}-moe_block-wi_1": [
                f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)
            ],
            f"{prefix}-moe_block-wo": [
                f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)
            ],
        })
      else:
        mapping.update({
            f"{prefix}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
            f"{prefix}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
            f"{prefix}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
        })
  return mapping


def QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", config.get("num_local_experts", 0))

  def pad_embedding_layer(input_tensor, target_shape):
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]
    if source_vocab_size == target_vocab_size:
      return input_tensor
    if saving_to_hf:
      return input_tensor[:target_vocab_size, :]
    padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    padded_tensor[:source_vocab_size, :] = input_tensor
    return padded_tensor

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def reshape_expert_kernel(input_tensor, target_shape=None):
    if input_tensor.ndim == 3:
      return input_tensor.transpose(0, 2, 1)
    return input_tensor.transpose(1, 0)

  def reshape_bias(input_tensor, target_shape=None):
    return input_tensor.reshape(target_shape)

  mapping = {
      "params-token_embedder-embedding": pad_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
  }
  kernel_hooks = [
      "self_attention-query-kernel", "self_attention-key-kernel", "self_attention-value-kernel",
      "self_attention-out-kernel", "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
  ]
  bias_hooks = ["self_attention-query-bias", "self_attention-key-bias", "self_attention-value-bias"]
  moe_gate_hooks = ["moe_block-gate-kernel", "moe_block-wi_0-kernel", "moe_block-wi_1-kernel", "moe_block-wo-kernel"]
  moe_expert_hooks = ["moe_block-wi_0", "moe_block-wi_1", "moe_block-wo"]
  if scan_layers:
    for key in kernel_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_kernel
    for key in bias_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_bias
    if num_experts > 1:
      for key in moe_gate_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_kernel
      for key in moe_expert_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_expert_kernel
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers-layers_{i}" if getattr(maxtext_config, "use_m3_model", False) else f"params-decoder-layers_{i}"
      for key in kernel_hooks:
        mapping[f"{prefix}-{key}"] = reshape_kernel
      for key in bias_hooks:
        mapping[f"{prefix}-{key}"] = reshape_bias
      if num_experts > 1:
        for key in moe_gate_hooks:
          mapping[f"{prefix}-{key}"] = reshape_kernel
        for key in moe_expert_hooks:
          mapping[f"{prefix}-{key}"] = reshape_expert_kernel
  return mapping


def QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN(target_shape=None):
  """Creates parameter transformation functions for Qwen3."""
  return {}
