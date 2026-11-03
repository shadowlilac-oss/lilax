"""Hy4 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def HY4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
      return cfg.get(key, default)
    return getattr(cfg, key, default)

  n_layers = (getattr(maxtext_config, "num_decoder_layers", None)
              or getattr(maxtext_config, "base_num_decoder_layers", None) or _get(config, "num_hidden_layers", 78))
  first_num_dense = int(getattr(maxtext_config, "first_num_dense_layers", _get(config, "first_num_dense_layers", 1)))
  cycle = int(getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", 4) or 4)
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-hc_head-hc_fn": "model.hc_head.hc_head_fn",
      "params-decoder-hc_head-hc_base": "model.hc_head.hc_head_base",
      "params-decoder-hc_head-hc_scale": "model.hc_head.hc_head_scale",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }

  def _is_full_indexer(layer_idx):
    if layer_idx < first_num_dense:
      return True
    return ((layer_idx - first_num_dense) % cycle) == 0

  def _add_layer(prefix, get_hf_key, is_moe, is_full_idx):
    keys = {
        "pre_self_attention_layer_norm-scale": "input_layernorm.weight",
        "post_self_attention_layer_norm-scale": "post_attention_layernorm.weight",
        "attn_hc-fn": "hc_attn_layer.hc_pre.hc_fn", "attn_hc-base": "hc_attn_layer.hc_pre.hc_base",
        "attn_hc-scale": "hc_attn_layer.hc_pre.hc_scale", "ffn_hc-fn": "hc_mlp_layer.hc_pre.hc_fn",
        "ffn_hc-base": "hc_mlp_layer.hc_pre.hc_base", "ffn_hc-scale": "hc_mlp_layer.hc_pre.hc_scale",
        "self_attention-wq_a-kernel": "self_attn.q_a_proj.weight",
        "self_attention-q_norm-scale": "self_attn.q_a_layernorm.weight",
        "self_attention-wq_b-kernel": "self_attn.q_b_proj.weight",
        "self_attention-wkv_a-kernel": "self_attn.kv_a_proj_with_mqa.weight",
        "self_attention-kv_norm-scale": "self_attn.kv_a_layernorm.weight",
        "self_attention-wkv_b-kernel": "self_attn.kv_b_proj.weight",
        "self_attention-gate_proj-kernel": "self_attn.linear_gate.weight",
        "self_attention-out-kernel": "self_attn.o_proj.weight", "self_attention-sinks": "self_attn.learnable_sink_param",
    }
    if is_full_idx:
      keys |= {"self_attention-indexer-wq_b-kernel": "self_attn.indexer.wq_b.weight",
               "self_attention-indexer-wk-kernel": "self_attn.indexer.wk.weight",
               "self_attention-indexer-k_norm-scale": "self_attn.indexer.k_norm.weight",
               "self_attention-indexer-k_norm-bias": "self_attn.indexer.k_norm.bias",
               "self_attention-indexer-weights_proj-kernel": "self_attn.indexer.weights_proj.weight"}
    if not is_moe:
      keys |= {"mlp-wi_0-kernel": "mlp.gate_proj.weight", "mlp-wi_1-kernel": "mlp.up_proj.weight",
               "mlp-wo-kernel": "mlp.down_proj.weight"}
    else:
      keys |= {"moe_block-MoeBlock_0-gate-kernel": "mlp.gate.weight",
               "moe_block-MoeBlock_0-wo": "mlp.experts.down_proj",
               "moe_block-shared_experts-wi_0-kernel": "mlp.shared_experts.gate_proj.weight",
               "moe_block-shared_experts-wi_1-kernel": "mlp.shared_experts.up_proj.weight",
               "moe_block-shared_experts-wo-kernel": "mlp.shared_experts.down_proj.weight"}
      mapping[f"{prefix.replace('params-', 'MoEBiasVar-')}-moe_block-MoeBlock_0-gate-bias"] = get_hf_key(
          "mlp.gate.e_score_correction_bias")
      mapping[(f"{prefix}-moe_block-MoeBlock_0-wi_0", f"{prefix}-moe_block-MoeBlock_0-wi_1")] = get_hf_key(
          "mlp.experts.gate_up_proj")
    for mt_key, hf_key in keys.items():
      mapping[f"{prefix}-{mt_key}"] = get_hf_key(hf_key)

  def _make_unrolled_hf_key_fn(layer_idx):
    return lambda target: f"model.layers.{layer_idx}.{target}"

  def _make_scanned_hf_key_fn(layer_indices):
    return lambda target: [f"model.layers.{idx}.{target}" for idx in layer_indices]

  num_prefix = min(n_layers, first_num_dense)
  moe_layers = max(0, n_layers - num_prefix)
  num_blocks = (moe_layers // cycle) if scan_layers else 0
  if not scan_layers or num_blocks == 0:
    for i in range(n_layers):
      _add_layer(f"params-decoder-layers_{i}", _make_unrolled_hf_key_fn(i), i >= first_num_dense, _is_full_indexer(i))
  else:
    for i in range(num_prefix):
      _add_layer(f"params-decoder-layers_{i}", _make_unrolled_hf_key_fn(i), False, True)
    for j in range(cycle):
      block_indices = [num_prefix + b * cycle + j for b in range(num_blocks)]
      _add_layer(f"params-decoder-scanned_blocks-layers_{j}", _make_scanned_hf_key_fn(block_indices), True, j == 0)
    for i in range(num_prefix + num_blocks * cycle, n_layers):
      _add_layer(f"params-decoder-layers_{i}", _make_unrolled_hf_key_fn(i), True, _is_full_indexer(i))
  return mapping


def HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
      return cfg.get(key, default)
    return getattr(cfg, key, default)

  n_layers = (getattr(maxtext_config, "num_decoder_layers", None)
              or getattr(maxtext_config, "base_num_decoder_layers", None) or _get(config, "num_hidden_layers", 78))
  first_num_dense = int(getattr(maxtext_config, "first_num_dense_layers", _get(config, "first_num_dense_layers", 1)))
  cycle = int(getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", 4) or 4)

  def reshape_kernel(input_tensor, target_shape=None):
    if saving_to_hf:
      if target_shape is not None:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      return input_tensor.T
    return input_tensor.T.reshape(target_shape) if target_shape is not None else input_tensor.T

  def transpose_expert(input_tensor, target_shape=None):
    del target_shape
    return input_tensor.swapaxes(-1, -2)

  def process_wi_0_wi_1(input_tensor, target_shape=None):
    del target_shape
    if saving_to_hf:
      wi_0, wi_1 = input_tensor
      gate_up = np.concatenate([wi_0, wi_1], axis=-1)
      return gate_up.swapaxes(-1, -2)
    gate, up = np.split(input_tensor, 2, axis=-2)
    gate = gate.swapaxes(-1, -2)
    up = up.swapaxes(-1, -2)
    return np.stack([gate, up], axis=-1)

  hooks = {"params-decoder-hc_head-hc_fn": reshape_kernel, "params-decoder-logits_dense-kernel": reshape_kernel}

  def _is_full_indexer(layer_idx):
    if layer_idx < first_num_dense:
      return True
    return ((layer_idx - first_num_dense) % cycle) == 0

  def _attach_layer_hooks(prefix, is_moe, is_full_idx):
    hooks[f"{prefix}-attn_hc-fn"] = reshape_kernel
    hooks[f"{prefix}-ffn_hc-fn"] = reshape_kernel
    for k in ("wq_a", "wq_b", "wkv_a", "wkv_b", "gate_proj", "out"):
      hooks[f"{prefix}-self_attention-{k}-kernel"] = reshape_kernel
    if is_full_idx:
      for k in ("wq_b", "wk", "weights_proj"):
        hooks[f"{prefix}-self_attention-indexer-{k}-kernel"] = reshape_kernel
    if not is_moe:
      for k in ("wi_0", "wi_1", "wo"):
        hooks[f"{prefix}-mlp-{k}-kernel"] = reshape_kernel
    else:
      hooks[f"{prefix}-moe_block-MoeBlock_0-gate-kernel"] = reshape_kernel
      hooks[(f"{prefix}-moe_block-MoeBlock_0-wi_0", f"{prefix}-moe_block-MoeBlock_0-wi_1")] = process_wi_0_wi_1
      hooks[f"{prefix}-moe_block-MoeBlock_0-wo"] = transpose_expert
      for k in ("wi_0", "wi_1", "wo"):
        hooks[f"{prefix}-moe_block-shared_experts-{k}-kernel"] = reshape_kernel

  num_prefix = min(n_layers, first_num_dense)
  moe_layers = max(0, n_layers - num_prefix)
  num_blocks = (moe_layers // cycle) if scan_layers else 0
  if not scan_layers or num_blocks == 0:
    for i in range(n_layers):
      _attach_layer_hooks(f"params-decoder-layers_{i}", i >= first_num_dense, _is_full_indexer(i))
  else:
    for i in range(num_prefix):
      _attach_layer_hooks(f"params-decoder-layers_{i}", False, True)
    for j in range(cycle):
      _attach_layer_hooks(f"params-decoder-scanned_blocks-layers_{j}", True, j == 0)
    for i in range(num_prefix + num_blocks * cycle, n_layers):
      _attach_layer_hooks(f"params-decoder-layers_{i}", True, _is_full_indexer(i))
  return hooks
