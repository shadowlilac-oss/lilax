"""OLMo 3 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  n_layers = config["num_hidden_layers"]
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  if scan_layers:
    for cycle_idx in range(layer_cycle_interval):
      hf_indices = range(cycle_idx, n_layers, layer_cycle_interval)
      prefix = f"params-decoder-layers-layers_{cycle_idx}"
      mapping.update({
          f"{prefix}-attention-query-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices],
          f"{prefix}-attention-key-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices],
          f"{prefix}-attention-value-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices],
          f"{prefix}-attention-out-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices],
          f"{prefix}-attention-query_norm-scale": [f"model.layers.{i}.self_attn.q_norm.weight" for i in hf_indices],
          f"{prefix}-attention-key_norm-scale": [f"model.layers.{i}.self_attn.k_norm.weight" for i in hf_indices],
          f"{prefix}-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in hf_indices],
          f"{prefix}-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in hf_indices],
          f"{prefix}-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in hf_indices],
          f"{prefix}-post_self_attention_layer_norm-scale": [
              f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
          ],
          f"{prefix}-post_mlp_layer_norm-scale": [
              f"model.layers.{i}.post_feedforward_layernorm.weight" for i in hf_indices
          ],
      })
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"model.layers.{i}"
      mapping.update({
          f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.weight",
          f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.weight",
          f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.weight",
          f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.weight",
          f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
          f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
          f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight",
          f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight",
          f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight",
          f"{prefix}-post_self_attention_layer_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
          f"{prefix}-post_mlp_layer_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
      })
  return mapping


def OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def adapt_olmo3_qk_norm(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def pad_hf_embedding_layer(input_tensor, target_shape):
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]
    if source_vocab_size == target_vocab_size:
      return input_tensor
    if saving_to_hf:
      return input_tensor[:target_vocab_size, :]
    padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    padded_tensor[:source_vocab_size, :] = input_tensor
    return padded_tensor

  hooks = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }
  kernel_keys = [
      "attention-query-kernel", "attention-key-kernel", "attention-value-kernel", "attention-out-kernel",
      "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
  ]
  norm_keys = [
      "post_self_attention_layer_norm-scale", "post_mlp_layer_norm-scale",
      "attention-query_norm-scale", "attention-key_norm-scale",
  ]
  cycle_len = getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", 4)
  n_layers = config["num_hidden_layers"]
  if scan_layers:
    prefixes = (f"params-decoder-layers-layers_{i}" for i in range(cycle_len))
  else:
    prefixes = (f"params-decoder-layers_{i}" for i in range(n_layers))
  for prefix in prefixes:
    for key in kernel_keys:
      hooks[f"{prefix}-{key}"] = reshape_kernel
    for key in norm_keys:
      if "attention-" in key and "_norm-scale" in key:
        hooks[f"{prefix}-{key}"] = adapt_olmo3_qk_norm
      else:
        hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer
  return hooks
