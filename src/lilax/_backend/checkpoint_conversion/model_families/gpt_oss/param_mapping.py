"""GPT-OSS parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np
import jax.numpy as jnp


def GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  n_layers = config["num_hidden_layers"]
  interval = maxtext_config.inhomogeneous_layer_cycle_interval
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  params = [
      ("pre_self_attention_layer_norm-scale", "input_layernorm.weight"),
      ("post_self_attention_layer_norm-scale", "post_attention_layernorm.weight"),
      ("GptOssAttention-query-kernel", "self_attn.q_proj.weight"),
      ("GptOssAttention-query-bias", "self_attn.q_proj.bias"),
      ("GptOssAttention-key-kernel", "self_attn.k_proj.weight"),
      ("GptOssAttention-key-bias", "self_attn.k_proj.bias"),
      ("GptOssAttention-value-kernel", "self_attn.v_proj.weight"),
      ("GptOssAttention-value-bias", "self_attn.v_proj.bias"),
      ("GptOssAttention-out-kernel", "self_attn.o_proj.weight"),
      ("GptOssAttention-out-bias", "self_attn.o_proj.bias"),
      ("GptOssAttention-sinks", "self_attn.sinks"),
      ("GptOssMlp-gate-kernel", "mlp.router.weight"), ("GptOssMlp-gate-bias", "mlp.router.bias"),
      ("GptOssMlp-wo", "mlp.experts.down_proj"), ("GptOssMlp-wo_bias", "mlp.experts.down_proj_bias"),
      (("GptOssMlp-wi_0", "GptOssMlp-wi_1"), "mlp.experts.gate_up_proj"),
      (("GptOssMlp-wi_0_bias", "GptOssMlp-wi_1_bias"), "mlp.experts.gate_up_proj_bias"),
  ]
  for idx in range(interval if scan_layers else n_layers):
    prefix = f"params-decoder-layers-layers_{idx}" if scan_layers else f"params-decoder-layers_{idx}"
    for mt, hf in params:
      key = tuple(f"{prefix}-{part}" for part in mt) if isinstance(mt, tuple) else f"{prefix}-{mt}"
      mapping[key] = ([f"model.layers.{i}.{hf}" for i in range(idx, n_layers, interval)]
                      if scan_layers else f"model.layers.{idx}.{hf}")
  return mapping


def GPT_OSS_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def transpose(x, target_shape=None):
    return x.T

  def reshape_kernel(x, target_shape):
    if saving_to_hf:
      return x.reshape(np.flip(np.array(target_shape))).T
    return x.T.reshape(target_shape)

  def reshape_bias(x, target_shape=None):
    return x.reshape(target_shape)

  def interleave(x, target_shape=None):
    if saving_to_hf:
      wi_0, wi_1 = x
      result = jnp.empty(target_shape, dtype=wi_0.dtype)
      result = result.at[..., ::2].set(wi_0)
      return result.at[..., 1::2].set(wi_1)
    return np.stack([x[..., ::2], x[..., 1::2]], axis=-1)

  hooks = {"params-decoder-logits_dense-kernel": transpose}
  indices = range(maxtext_config.inhomogeneous_layer_cycle_interval if scan_layers else config["num_hidden_layers"])
  for idx in indices:
    prefix = f"params-decoder-layers-layers_{idx}" if scan_layers else f"params-decoder-layers_{idx}"
    for key in ("query", "key", "value"):
      hooks[f"{prefix}-GptOssAttention-{key}-kernel"] = reshape_kernel
      hooks[f"{prefix}-GptOssAttention-{key}-bias"] = reshape_bias
    hooks[f"{prefix}-GptOssAttention-out-kernel"] = reshape_kernel
    hooks[f"{prefix}-GptOssMlp-gate-kernel"] = transpose
    hooks[(f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1")] = interleave
    hooks[(f"{prefix}-GptOssMlp-wi_0_bias", f"{prefix}-GptOssMlp-wi_1_bias")] = interleave
  return hooks
