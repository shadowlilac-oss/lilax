"""DeepSeek V4 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
      return cfg.get(key, default)
    return getattr(cfg, key, default)

  n_layers = _get(config, "num_hidden_layers", getattr(maxtext_config, "base_num_decoder_layers", 43))
  num_experts = (maxtext_config.num_experts if getattr(maxtext_config, "num_experts", 0) > 0
                 else _get(config, "n_routed_experts", 256))
  num_hash_layers = _get(config, "num_hash_layers", getattr(maxtext_config, "first_num_hash_layers", 3))
  compress_ratios = _get(config, "compress_ratios", getattr(maxtext_config, "compress_ratios", None))

  def _get_compressor_type(layer_idx):
    if compress_ratios and layer_idx < len(compress_ratios):
      ratio = compress_ratios[layer_idx]
    else:
      ratio = 0 if layer_idx < 2 else (4 if layer_idx % 2 == 0 else 128)
    if ratio == 4:
      return "csa"
    elif ratio > 4:
      return "hca"
    return None

  mapping = {
      "params-token_embedder-embedding": "embed.weight", "params-decoder-decoder_norm-scale": "norm.weight",
      "params-decoder-logits_dense-kernel": "head.weight", "params-decoder-hc_head-hc_fn": "hc_head_fn",
      "params-decoder-hc_head-hc_base": "hc_head_base", "params-decoder-hc_head-hc_scale": "hc_head_scale",
  }

  def _add_layer(prefix, get_hf_key, is_hash, compressor_type=None):
    mapping[f"{prefix}-pre_self_attention_layer_norm-scale"] = get_hf_key("attn_norm.weight")
    mapping[f"{prefix}-post_self_attention_layer_norm-scale"] = get_hf_key("ffn_norm.weight")
    mapping[f"{prefix}-mhc_attention-mhc_norm-scale"] = None
    mapping[f"{prefix}-mhc_mlp-mhc_norm-scale"] = None
    for mod, hf_attr in [("mhc_attention", "hc_attn"), ("mhc_mlp", "hc_ffn")]:
      for part in ["pre_alpha", "post_alpha", "res_alpha"]:
        mapping[f"{prefix}-{mod}-{part}"] = get_hf_key(f"{hf_attr}_fn")
      for part in ["pre_beta", "post_beta", "res_beta"]:
        mapping[f"{prefix}-{mod}-{part}"] = get_hf_key(f"{hf_attr}_base")
      for part in ["pre_alpha_scale", "post_alpha_scale", "res_alpha_scale"]:
        mapping[f"{prefix}-{mod}-{part}"] = get_hf_key(f"{hf_attr}_scale")
    keys = {
        "self_attention-q_norm-scale": "attn.q_norm.weight", "self_attention-kv_norm-scale": "attn.kv_norm.weight",
        "self_attention-wq_a-kernel": "attn.wq_a.weight", "self_attention-wq_b-kernel": "attn.wq_b.weight",
        "self_attention-wkv-kernel": "attn.wkv.weight", "self_attention-sinks": "attn.attn_sink",
        "self_attention-o_a_proj-kernel": "attn.wo_a.weight", "self_attention-o_b_proj-kernel": "attn.wo_b.weight",
        "mlp-MoeBlock_0-gate-kernel": "ffn.gate.weight", "mlp-shared_experts-wi_0-kernel": "ffn.shared_experts.w1.weight",
        "mlp-shared_experts-wi_1-kernel": "ffn.shared_experts.w3.weight",
        "mlp-shared_experts-wo-kernel": "ffn.shared_experts.w2.weight",
    }
    for mt_key, hf_key in keys.items():
      mapping[f"{prefix}-{mt_key}"] = get_hf_key(hf_key)

    def _add_compressor_keys(mt_comp_prefix, hf_comp_prefix):
      mapping[f"{mt_comp_prefix}-gate_proj-kernel"] = get_hf_key(f"{hf_comp_prefix}.wgate.weight")
      mapping[f"{mt_comp_prefix}-kv_proj-kernel"] = get_hf_key(f"{hf_comp_prefix}.wkv.weight")
      mapping[f"{mt_comp_prefix}-kv_norm-scale"] = get_hf_key(f"{hf_comp_prefix}.norm.weight")
      mapping[f"{mt_comp_prefix}-position_bias"] = get_hf_key(f"{hf_comp_prefix}.ape")

    if compressor_type in ("csa", "both"):
      _add_compressor_keys(f"{prefix}-self_attention-csa_compressor", "attn.compressor")
      _add_compressor_keys(f"{prefix}-self_attention-csa_compressor-indexer", "attn.indexer.compressor")
      mapping[f"{prefix}-self_attention-csa_compressor-indexer-weights_proj-kernel"] = get_hf_key(
          "attn.indexer.weights_proj.weight")
      mapping[f"{prefix}-self_attention-csa_compressor-indexer-q_proj-kernel"] = get_hf_key("attn.indexer.wq_b.weight")
    if compressor_type in ("hca", "both"):
      _add_compressor_keys(f"{prefix}-self_attention-hca_compressor", "attn.compressor")
    if is_hash:
      mapping[f"{prefix.replace('params-', 'Tid2EidVar-')}-mlp-MoeBlock_0-tid2eid"] = get_hf_key("ffn.gate.tid2eid")
    else:
      mapping[f"{prefix.replace('params-', 'MoEBiasVar-')}-mlp-MoeBlock_0-gate-bias"] = get_hf_key("ffn.gate.bias")
    for mt, hf in (("wi_0", "w1"), ("wi_1", "w3"), ("wo", "w2")):
      mapping[f"{prefix}-mlp-MoeBlock_0-{mt}"] = get_hf_key(f"ffn.experts.{{e}}.{hf}.weight", per_expert=True)

  def _make_unrolled_hf_key_fn(layer_idx):
    def get_hf_key(target, per_expert=False):
      if per_expert:
        return [f"layers.{layer_idx}.{target.replace('{e}', str(e))}" for e in range(num_experts)]
      return f"layers.{layer_idx}.{target}"
    return get_hf_key

  def _make_scanned_hf_key_fn(layer_indices):
    def get_hf_key(target, per_expert=False):
      if per_expert:
        return [[f"layers.{idx}.{target.replace('{e}', str(e))}" for idx in layer_indices] for e in range(num_experts)]
      return [f"layers.{idx}.{target}" for idx in layer_indices]
    return get_hf_key

  if not scan_layers:
    for i in range(n_layers):
      _add_layer(f"params-decoder-layers_{i}", _make_unrolled_hf_key_fn(i), i < num_hash_layers, _get_compressor_type(i))
  else:
    for i in range(num_hash_layers):
      _add_layer(f"params-decoder-layers_{i}", _make_unrolled_hf_key_fn(i), True, _get_compressor_type(i))
    _add_layer("params-decoder-scanned_blocks-layers_0", _make_scanned_hf_key_fn(list(range(num_hash_layers, n_layers, 2))),
               False, "hca")
    _add_layer("params-decoder-scanned_blocks-layers_1", _make_scanned_hf_key_fn(list(range(num_hash_layers + 1, n_layers, 2))),
               False, "csa")

  def _make_mtp_hf_key_fn(mtp_idx):
    def get_hf_key(target, per_expert=False):
      if per_expert:
        return [f"mtp.{mtp_idx}.{target.replace('{e}', str(e))}" for e in range(num_experts)]
      return f"mtp.{mtp_idx}.{target}"
    return get_hf_key

  for j in range(maxtext_config.mtp_num_layers):
    mtp_prefix = f"params-mtp_block-mtp_layer_{j + 1}"
    mapping[f"{mtp_prefix}-mtp_{j + 1}_embedding_norm-scale"] = f"mtp.{j}.enorm.weight"
    mapping[f"{mtp_prefix}-mtp_{j + 1}_hidden_state_norm-scale"] = f"mtp.{j}.hnorm.weight"
    mapping[f"{mtp_prefix}-mtp_{j + 1}_final_norm-scale"] = f"mtp.{j}.norm.weight"
    mapping[f"{mtp_prefix}-mtp_{j + 1}_projection-kernel"] = (f"mtp.{j}.e_proj.weight", f"mtp.{j}.h_proj.weight")
    mapping[f"{mtp_prefix}-hc_head-hc_fn"] = f"mtp.{j}.hc_head_fn"
    mapping[f"{mtp_prefix}-hc_head-hc_base"] = f"mtp.{j}.hc_head_base"
    mapping[f"{mtp_prefix}-hc_head-hc_scale"] = f"mtp.{j}.hc_head_scale"
    _add_layer(f"{mtp_prefix}-mtp_{j + 1}_transformer_layer", _make_mtp_hf_key_fn(j), False, None)
  return mapping


def DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
      return cfg.get(key, default)
    return getattr(cfg, key, default)

  n_layers = _get(config, "num_hidden_layers", getattr(maxtext_config, "base_num_decoder_layers", 43))
  num_hash_layers = _get(config, "num_hash_layers", getattr(maxtext_config, "first_num_hash_layers", 3))

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    try:
      return input_tensor.T.reshape(target_shape)
    except Exception as e:
      raise ValueError(f"RESHAPE FAILED: {str(e)} for target {target_shape} with input {input_tensor.shape}") from e

  def reshape_o_a_proj(input_tensor, target_shape):
    if saving_to_hf:
      heads, head_dim, in_dim = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
      if target_shape[0] == in_dim:
        return input_tensor.transpose(2, 0, 1).reshape(target_shape)
      return input_tensor.transpose(0, 2, 1).reshape(target_shape)
    heads, head_dim, in_dim = target_shape[0], target_shape[1], target_shape[2]
    if input_tensor.shape[0] == in_dim:
      return input_tensor.reshape(in_dim, heads, head_dim).transpose(1, 2, 0)
    return input_tensor.reshape(heads, in_dim, head_dim).transpose(0, 2, 1)

  mhc_composite_fns = {
      "alpha": lambda w, *args, **kwargs: np.concatenate(w, axis=-1).T,
      "beta": lambda w, *args, **kwargs: np.concatenate([w[0], w[1], w[2].flatten()], axis=-1),
      "scale": lambda w, *args, **kwargs: np.concatenate(w, axis=-1),
  }

  def mhc_slice_hook(mode, part_idx):
    def hook(tensor, target_shape=None):
      del target_shape
      if mode == "scale":
        return tensor[part_idx : part_idx + 1]
      t = tensor.T if mode == "alpha" else tensor
      k = int(np.sqrt(1 + t.shape[-1]) - 1)
      if part_idx == 0:
        return t[..., :k]
      elif part_idx == 1:
        return t[..., k : 2 * k]
      res = t[..., 2 * k :]
      return res.reshape(k, k) if mode == "beta" else res
    return hook

  def mhc_dummy_norm(input_tensor, target_shape=None):
    dtype = input_tensor.dtype if input_tensor is not None else np.float32
    return np.ones(target_shape, dtype=dtype)

  def unpad_hf_embedding_layer(input_tensor, target_shape):
    target_vocab_size = target_shape[0]
    if input_tensor.shape[0] == target_vocab_size:
      return input_tensor[:target_vocab_size, :]
    return input_tensor[:, :target_vocab_size].T

  def unpad_logits_layer(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor[:, : target_shape[0]].T
    return input_tensor[: target_shape[1], :].T

  def reshape_mtp_projection(weights, target_shape=None):
    del target_shape
    if saving_to_hf:
      half = weights.shape[0] // 2
      return weights[:half, :].T, weights[half:, :].T
    e_proj, h_proj = weights
    return np.concatenate([e_proj.T, h_proj.T], axis=0)

  mapping = {"params-token_embedder-embedding": unpad_hf_embedding_layer,
             "params-decoder-logits_dense-kernel": unpad_logits_layer, "params-decoder-hc_head-hc_fn": reshape_kernel}

  def _attach_layer_hooks(prefix):
    mapping[f"{prefix}-self_attention-o_a_proj-kernel"] = reshape_o_a_proj
    mapping[f"{prefix}-mhc_attention-mhc_norm-scale"] = mhc_dummy_norm
    mapping[f"{prefix}-mhc_mlp-mhc_norm-scale"] = mhc_dummy_norm
    for key in ("self_attention-wq_a-kernel", "self_attention-wq_b-kernel", "self_attention-wkv-kernel",
                "self_attention-o_b_proj-kernel", "mlp-MoeBlock_0-gate-kernel", "mlp-MoeBlock_0-wi_0",
                "mlp-MoeBlock_0-wi_1", "mlp-MoeBlock_0-wo", "mlp-shared_experts-wi_0-kernel",
                "mlp-shared_experts-wi_1-kernel", "mlp-shared_experts-wo-kernel",
                "self_attention-csa_compressor-gate_proj-kernel", "self_attention-csa_compressor-kv_proj-kernel",
                "self_attention-csa_compressor-indexer-gate_proj-kernel",
                "self_attention-csa_compressor-indexer-kv_proj-kernel",
                "self_attention-csa_compressor-indexer-weights_proj-kernel",
                "self_attention-csa_compressor-indexer-q_proj-kernel", "self_attention-hca_compressor-gate_proj-kernel",
                "self_attention-hca_compressor-kv_proj-kernel"):
      mapping[f"{prefix}-{key}"] = reshape_kernel
    for mod in ("mhc_attention", "mhc_mlp"):
      for mode, suffix in (("alpha", "alpha"), ("beta", "beta"), ("scale", "alpha_scale")):
        keys = (f"{prefix}-{mod}-pre_{suffix}", f"{prefix}-{mod}-post_{suffix}", f"{prefix}-{mod}-res_{suffix}")
        if saving_to_hf:
          mapping[keys] = mhc_composite_fns[mode]
        else:
          for idx, k in enumerate(keys):
            mapping[k] = mhc_slice_hook(mode, idx)

  prefixes = [f"params-decoder-layers_{i}" for i in range(n_layers if not scan_layers else num_hash_layers)]
  if scan_layers:
    prefixes.extend(["params-decoder-scanned_blocks-layers_0", "params-decoder-scanned_blocks-layers_1"])
  for prefix in prefixes:
    _attach_layer_hooks(prefix)
  for j in range(maxtext_config.mtp_num_layers):
    mtp_prefix = f"params-mtp_block-mtp_layer_{j + 1}"
    mapping[f"{mtp_prefix}-mtp_{j + 1}_projection-kernel"] = reshape_mtp_projection
    mapping[f"{mtp_prefix}-hc_head-hc_fn"] = reshape_kernel
    _attach_layer_hooks(f"{mtp_prefix}-mtp_{j + 1}_transformer_layer")
  return mapping


DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_MAPPING = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING
DEEPSEEKV4_MAXTEXT_TO_HF_PARAM_HOOK_FN = DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN
