"""Gemma 2 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mappings between MaxText and Hugging Face Gemma 2 parameters."""
  nlayers = config["num_hidden_layers"]
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }
  parameters = [
      ("pre_self_attention_norm", "scale", "input_layernorm.weight"),
      ("mlp", "wo-kernel", "mlp.down_proj.weight"),
      ("mlp", "wi_1-kernel", "mlp.up_proj.weight"),
      ("mlp", "wi_0-kernel", "mlp.gate_proj.weight"),
      ("post_self_attention_norm", "scale", "post_attention_layernorm.weight"),
      ("post_ffw_norm", "scale", "post_feedforward_layernorm.weight"),
      ("pre_ffw_norm", "scale", "pre_feedforward_layernorm.weight"),
      ("self_attention", "key-kernel", "self_attn.k_proj.weight"),
      ("self_attention", "out-kernel", "self_attn.o_proj.weight"),
      ("self_attention", "query-kernel", "self_attn.q_proj.weight"),
      ("self_attention", "value-kernel", "self_attn.v_proj.weight"),
  ]
  if scan_layers:
    for attention_type, start in (("global", 1), ("local", 0)):
      for component, suffix, hf_name in parameters:
        key = f"params-decoder-layers-{component}_{attention_type}-{suffix}"
        mapping[key] = [f"model.layers.{index}.{hf_name}" for index in range(start, nlayers, 2)]
  else:
    for layer_index in range(nlayers // 2):
      for attention_type, offset in (("global", 1), ("local", 0)):
        for component, suffix, hf_name in parameters:
          key = f"params-decoder-layers_{layer_index}-{component}_{attention_type}-{suffix}"
          mapping[key] = f"model.layers.{layer_index * 2 + offset}.{hf_name}"
  return mapping


def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Gemma 2 conversion."""
  nlayers = config["num_hidden_layers"]

  def pad_hf_embedding_layer(input_tensor, target_shape):
    normalizer = np.dtype("float32").type(config["hidden_size"] ** 0.5)
    if saving_to_hf:
      target_tensor = input_tensor[: target_shape[0], : target_shape[1]]
      target_tensor = target_tensor / normalizer
      return target_tensor.astype(input_tensor.dtype)
    else:
      target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
      target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
      target_tensor = target_tensor * normalizer
      return target_tensor.astype(input_tensor.dtype)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    else:
      return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    if saving_to_hf:
      return (input_tensor - 1.0).reshape(target_shape)
    else:
      return (input_tensor + 1.0).reshape(target_shape)

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)
    else:
      depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)

  query_hook_chain = [reshape_kernel, scale_query_layer]
  mapping = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }
  layer_hooks = [
      ("self_attention_global-query-kernel", query_hook_chain),
      ("self_attention_local-query-kernel", query_hook_chain),
      ("self_attention_global-key-kernel", reshape_kernel),
      ("self_attention_local-key-kernel", reshape_kernel),
      ("self_attention_global-value-kernel", reshape_kernel),
      ("self_attention_local-value-kernel", reshape_kernel),
      ("mlp_global-wo-kernel", reshape_kernel),
      ("mlp_global-wi_1-kernel", reshape_kernel),
      ("mlp_global-wi_0-kernel", reshape_kernel),
      ("self_attention_global-out-kernel", reshape_kernel),
      ("mlp_local-wo-kernel", reshape_kernel),
      ("mlp_local-wi_1-kernel", reshape_kernel),
      ("mlp_local-wi_0-kernel", reshape_kernel),
      ("self_attention_local-out-kernel", reshape_kernel),
      ("pre_self_attention_norm_global-scale", scale_rmsnorm_layer),
      ("post_self_attention_norm_global-scale", scale_rmsnorm_layer),
      ("post_ffw_norm_global-scale", scale_rmsnorm_layer),
      ("pre_ffw_norm_global-scale", scale_rmsnorm_layer),
      ("pre_self_attention_norm_local-scale", scale_rmsnorm_layer),
      ("post_self_attention_norm_local-scale", scale_rmsnorm_layer),
      ("post_ffw_norm_local-scale", scale_rmsnorm_layer),
      ("pre_ffw_norm_local-scale", scale_rmsnorm_layer),
  ]
  prefixes = ["params-decoder-layers-"] if scan_layers else [
      f"params-decoder-layers_{index}-" for index in range(nlayers // 2)
  ]
  for prefix in prefixes:
    for suffix, hook in layer_hooks:
      mapping[prefix + suffix] = hook
  return mapping
