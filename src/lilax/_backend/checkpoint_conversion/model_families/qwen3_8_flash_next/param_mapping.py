"""Qwen3.8-Flash-Next parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def _qwen3_8_flash_next_hf_config_dict(config):
  if hasattr(config, "to_dict"):
    return config.to_dict()
  return config


def _qwen3_8_flash_next_get(cfg, key, default=None):
  if cfg is None:
    return default
  if isinstance(cfg, dict):
    if "text_config" in cfg and isinstance(cfg["text_config"], dict) and key in cfg["text_config"]:
      return cfg["text_config"][key]
    return cfg.get(key, default)
  return getattr(cfg, key, default)


def _qwen3_8_flash_next_text_base(config) -> str:
  if isinstance(config, dict):
    if config.get("language_model_only", False):
      return "model"
    if "text_config" in config or "vision_config" in config:
      return "model.language_model"
  return "model"


def _qwen3_8_flash_next_layer_sub_specs(is_qsa: bool):
  specs = [
      ("mlp-gate-linear-kernel", "mlp.gate.weight", "transpose"),
      ("mlp-experts-gate_up_proj", "mlp.experts.gate_up_proj", "experts"),
      ("mlp-experts-down_proj", "mlp.experts.down_proj", "experts"),
      ("mlp-shared_expert_gate-kernel", "mlp.shared_expert_gate.weight", "transpose"),
  ]
  for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
    specs.append((f"{hc}-hc_norm-scale", f"{hc}.hc_norm.weight", "copy"))
    for w in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
      specs.append((f"{hc}-{w}-kernel", f"{hc}.{w}.weight", "transpose"))
  for p in ("gate_proj", "up_proj", "down_proj"):
    specs.append((f"mlp-shared_expert-{p}-kernel", f"mlp.shared_expert.{p}.weight", "transpose"))
  if is_qsa:
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
      specs.append((f"self_attn-{p}-kernel", f"self_attn.{p}.weight", "transpose"))
    for p in ("q_norm", "k_norm"):
      specs.append((f"self_attn-{p}-scale", f"self_attn.{p}.weight", "copy"))
    specs.append(("self_attn-indexer-index_qk_proj-kernel", "self_attn.indexer.index_qk_proj.weight", "transpose"))
    for p in ("q_layernorm", "k_layernorm"):
      specs.append((f"self_attn-indexer-{p}-scale", f"self_attn.indexer.{p}.weight", "copy"))
  else:
    for p in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
      specs.append((f"linear_attn-{p}-kernel", f"linear_attn.{p}.weight", "transpose"))
    specs.extend([
        ("linear_attn-conv1d-kernel", "linear_attn.conv1d.weight", "conv1d"),
        ("linear_attn-norm-scale", "linear_attn.norm.weight", "copy"),
        ("linear_attn-A_log", "linear_attn.A_log", "copy"),
        ("linear_attn-dt_bias", "linear_attn.dt_bias", "copy"),
    ])
  return specs


def QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  config = _qwen3_8_flash_next_hf_config_dict(config)
  text_base = _qwen3_8_flash_next_text_base(config)
  n_layers = int(_qwen3_8_flash_next_get(maxtext_config, "num_decoder_layers", _qwen3_8_flash_next_get(
      maxtext_config, "base_num_decoder_layers", _qwen3_8_flash_next_get(config, "num_hidden_layers", 48))))
  cycle_interval = int(_qwen3_8_flash_next_get(maxtext_config, "inhomogeneous_layer_cycle_interval",
                                               _qwen3_8_flash_next_get(config, "full_attention_interval", 4)))
  ple_layer_ids = _qwen3_8_flash_next_get(config, "ple_layer_ids", [2])
  use_ple_layer = bool(_qwen3_8_flash_next_get(maxtext_config, "use_ple_layer", bool(ple_layer_ids)))
  default_pid = (int(ple_layer_ids[0]) - 1) if ple_layer_ids else 1
  pid = int(_qwen3_8_flash_next_get(maxtext_config, "ple_layer_id", default_pid))
  num_shards = int(_qwen3_8_flash_next_get(maxtext_config, "split_ngram_parts",
                                           _qwen3_8_flash_next_get(config, "split_ngram_parts", 128)))
  mtp_num_layers = int(_qwen3_8_flash_next_get(maxtext_config, "mtp_num_layers",
                                               _qwen3_8_flash_next_get(config, "mtp_num_hidden_layers", 1)))
  mapping = {
      "params-decoder-token_embedder-embedding": f"{text_base}.embed_tokens.weight",
      "params-lm_head-kernel": "lm_head.weight",
      "params-decoder-hyper_connection_mixer-hc_norm-scale": f"{text_base}.hyper_connection_mixer.hc_norm.weight",
      "params-decoder-hyper_connection_mixer-input_mix_weight_down-kernel":
          f"{text_base}.hyper_connection_mixer.input_mix_weight_down.weight",
      "params-decoder-hyper_connection_mixer-input_mix_weight_up-kernel":
          f"{text_base}.hyper_connection_mixer.input_mix_weight_up.weight",
  }
  if not scan_layers:
    for i in range(n_layers):
      is_qsa = (i % cycle_interval) == (cycle_interval - 1)
      prefix = f"params-decoder-layers_{i}"
      for mt_sub, hf_sub, _ in _qwen3_8_flash_next_layer_sub_specs(is_qsa):
        mapping[f"{prefix}-{mt_sub}"] = f"{text_base}.layers.{i}.{hf_sub}"
  else:
    num_cycles = n_layers // cycle_interval
    for p in range(cycle_interval):
      is_qsa = p == (cycle_interval - 1)
      blk = f"qsa_{p}" if is_qsa else f"gdn_{p}"
      prefix = f"params-decoder-cycles-{blk}"
      layer_indices = [c * cycle_interval + p for c in range(num_cycles)]
      for mt_sub, hf_sub, _ in _qwen3_8_flash_next_layer_sub_specs(is_qsa):
        mapping[f"{prefix}-{mt_sub}"] = [f"{text_base}.layers.{idx}.{hf_sub}" for idx in layer_indices]
  if use_ple_layer:
    ple_prefix = "params-decoder-ple" if scan_layers else f"params-decoder-layers_{pid}-ple"
    hf_ple_prefix = f"{text_base}.layers.{pid}.ple"
    mapping[f"{ple_prefix}-conv1d-kernel"] = f"{hf_ple_prefix}.conv1d.weight"
    mapping[f"{ple_prefix}-key_proj-kernel"] = f"{hf_ple_prefix}.key_proj.weight"
    mapping[f"{ple_prefix}-value_proj-kernel"] = f"{hf_ple_prefix}.value_proj.weight"
    mapping[f"{ple_prefix}-norm_key-scale"] = f"{hf_ple_prefix}.norm_key.weight"
    mapping[f"{ple_prefix}-norm_query-scale"] = f"{hf_ple_prefix}.norm_query.weight"
    mapping[f"{ple_prefix}-norm_conv-scale"] = f"{hf_ple_prefix}.norm_conv.weight"
    if num_shards > 1:
      mapping[f"{ple_prefix}-ple_embedding-ngram_embedding-embedding"] = tuple(
          f"{hf_ple_prefix}.ple_embedding.ngram_embedding.shard_{s}.weight" for s in range(num_shards))
    else:
      mapping[f"{ple_prefix}-ple_embedding-ngram_embedding-embedding"] = (
          f"{hf_ple_prefix}.ple_embedding.ngram_embedding.weight")
  if mtp_num_layers > 0:
    mapping["params-mtp-pre_fc_norm_embedding-scale"] = "mtp.pre_fc_norm_embedding.weight"
    mapping["params-mtp-pre_fc_norm_hidden-scale"] = "mtp.pre_fc_norm_hidden.weight"
    mapping["params-mtp-fc_embedding-kernel"] = "mtp.fc_embedding.weight"
    mapping["params-mtp-fc_hidden-kernel"] = "mtp.fc_hidden.weight"
    mapping["params-mtp-hyper_connection_mixer-hc_norm-scale"] = "mtp.hyper_connection_mixer.hc_norm.weight"
    mapping["params-mtp-hyper_connection_mixer-input_mix_weight_down-kernel"] = (
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight")
    mapping["params-mtp-hyper_connection_mixer-input_mix_weight_up-kernel"] = (
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight")
    for m in range(mtp_num_layers):
      mtp_layer_prefix = f"params-mtp-layers_{m}"
      for mt_sub, hf_sub, _ in _qwen3_8_flash_next_layer_sub_specs(is_qsa=True):
        mapping[f"{mtp_layer_prefix}-{mt_sub}"] = f"mtp.layers.{m}.{hf_sub}"
  return mapping


def QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  config = _qwen3_8_flash_next_hf_config_dict(config)
  n_layers = int(_qwen3_8_flash_next_get(maxtext_config, "num_decoder_layers", _qwen3_8_flash_next_get(
      maxtext_config, "base_num_decoder_layers", _qwen3_8_flash_next_get(config, "num_hidden_layers", 48))))
  cycle_interval = int(_qwen3_8_flash_next_get(maxtext_config, "inhomogeneous_layer_cycle_interval",
                                               _qwen3_8_flash_next_get(config, "full_attention_interval", 4)))
  ple_layer_ids = _qwen3_8_flash_next_get(config, "ple_layer_ids", [2])
  use_ple_layer = bool(_qwen3_8_flash_next_get(maxtext_config, "use_ple_layer", bool(ple_layer_ids)))
  default_pid = (int(ple_layer_ids[0]) - 1) if ple_layer_ids else 1
  pid = int(_qwen3_8_flash_next_get(maxtext_config, "ple_layer_id", default_pid))
  num_shards = int(_qwen3_8_flash_next_get(maxtext_config, "split_ngram_parts",
                                           _qwen3_8_flash_next_get(config, "split_ngram_parts", 128)))
  mtp_num_layers = int(_qwen3_8_flash_next_get(maxtext_config, "mtp_num_layers",
                                               _qwen3_8_flash_next_get(config, "mtp_num_hidden_layers", 1)))

  def reshape_kernel(input_tensor, target_shape=None):
    if saving_to_hf:
      return input_tensor.T
    return input_tensor.T.reshape(target_shape) if target_shape is not None else input_tensor.T

  def reshape_conv1d(input_tensor, target_shape=None):
    del target_shape
    return input_tensor.transpose(2, 1, 0)

  def reshape_experts(input_tensor, target_shape=None):
    del target_shape
    return input_tensor.transpose(0, 2, 1)

  def reshape_ple_ngram_shards(weights, target_shape=None):
    del target_shape
    if saving_to_hf:
      shard_len = weights.shape[0] // num_shards
      return tuple(weights[s * shard_len : (s + 1) * shard_len] for s in range(num_shards))
    if isinstance(weights, (tuple, list)):
      return np.concatenate([np.asarray(w) for w in weights], axis=0)
    return np.asarray(weights)

  kind_to_hook = {"transpose": reshape_kernel, "conv1d": reshape_conv1d, "experts": reshape_experts}
  hooks = {
      "params-lm_head-kernel": reshape_kernel,
      "params-decoder-hyper_connection_mixer-input_mix_weight_down-kernel": reshape_kernel,
      "params-decoder-hyper_connection_mixer-input_mix_weight_up-kernel": reshape_kernel,
  }

  def _attach_layer_hooks(prefix, is_qsa):
    for mt_sub, _, kind in _qwen3_8_flash_next_layer_sub_specs(is_qsa):
      if kind in kind_to_hook:
        hooks[f"{prefix}-{mt_sub}"] = kind_to_hook[kind]

  if not scan_layers:
    for i in range(n_layers):
      is_qsa = (i % cycle_interval) == (cycle_interval - 1)
      _attach_layer_hooks(f"params-decoder-layers_{i}", is_qsa)
  else:
    for p in range(cycle_interval):
      is_qsa = p == (cycle_interval - 1)
      blk = f"qsa_{p}" if is_qsa else f"gdn_{p}"
      _attach_layer_hooks(f"params-decoder-cycles-{blk}", is_qsa)
  if use_ple_layer:
    ple_prefix = "params-decoder-ple" if scan_layers else f"params-decoder-layers_{pid}-ple"
    hooks[f"{ple_prefix}-conv1d-kernel"] = reshape_conv1d
    hooks[f"{ple_prefix}-key_proj-kernel"] = reshape_kernel
    hooks[f"{ple_prefix}-value_proj-kernel"] = reshape_kernel
    if num_shards > 1:
      hooks[f"{ple_prefix}-ple_embedding-ngram_embedding-embedding"] = reshape_ple_ngram_shards
  if mtp_num_layers > 0:
    hooks["params-mtp-fc_embedding-kernel"] = reshape_kernel
    hooks["params-mtp-fc_hidden-kernel"] = reshape_kernel
    hooks["params-mtp-hyper_connection_mixer-input_mix_weight_down-kernel"] = reshape_kernel
    hooks["params-mtp-hyper_connection_mixer-input_mix_weight_up-kernel"] = reshape_kernel
    for m in range(mtp_num_layers):
      _attach_layer_hooks(f"params-mtp-layers_{m}", is_qsa=True)
  return hooks
