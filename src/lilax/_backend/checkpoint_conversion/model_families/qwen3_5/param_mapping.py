"""Qwen3.5 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


def QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False, *, hf_prefix="model.language_model"):
  num_main_layers = getattr(maxtext_config, "num_decoder_layers", None) or config["text_config"]["num_hidden_layers"]
  cycle = maxtext_config.inhomogeneous_layer_cycle_interval
  is_quantized = getattr(maxtext_config, "weight_dtype", None) == "float8_e4m3fn"
  num_experts = getattr(maxtext_config, "num_experts", None) or config.get("text_config", config).get("num_experts", 1)
  mapping = {
      "params-token_embedder-embedding": f"{hf_prefix}.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": f"{hf_prefix}.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  shared = [("input_layernorm-scale", "input_layernorm.weight"),
            ("post_attention_layernorm-scale", "post_attention_layernorm.weight")]
  experts = []
  if num_experts > 1:
    shared += [("mlp-routed_experts-gate-kernel", "mlp.gate.weight"),
               ("mlp-shared_expert-wi_0-kernel", "mlp.shared_expert.gate_proj.weight"),
               ("mlp-shared_expert-wi_1-kernel", "mlp.shared_expert.up_proj.weight"),
               ("mlp-shared_expert-wo-kernel", "mlp.shared_expert.down_proj.weight"),
               ("mlp-shared_expert_gate-kernel", "mlp.shared_expert_gate.weight")]
    if is_quantized:
      num_experts = config.get("text_config", config).get("num_experts", 256)
      shared += [(f"mlp-shared_expert-{mt}-kernel_scale", f"mlp.shared_expert.{hf}.weight_scale_inv")
                 for mt, hf in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj"))]
      experts = [(f"mlp-routed_experts-{mt}{scale}", f"{hf}.{weight}")
                 for mt, hf in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj"))
                 for scale, weight in (("", "weight"), ("_scale", "weight_scale_inv"))]
    else:
      shared.append(("mlp-routed_experts-wo", "mlp.experts.down_proj"))
  else:
    shared += [(f"mlp-{mt}-kernel", f"mlp.{hf}.weight")
               for mt, hf in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj"))]
  local = shared + [
      ("attention-in_proj_qkvz-kernel", ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight")),
      ("attention-in_proj_ba-kernel", ("linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight")),
      ("attention-conv1d-kernel", "linear_attn.conv1d.weight"), ("attention-A_log", "linear_attn.A_log"),
      ("attention-dt_bias", "linear_attn.dt_bias"), ("attention-norm-rms_norm-scale", "linear_attn.norm.weight"),
      ("attention-out_proj-kernel", "linear_attn.out_proj.weight"),
  ]
  global_specs = shared + [(f"attention-attention-{mt}-kernel", f"self_attn.{hf}_proj.weight")
                           for mt, hf in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o"))]
  global_specs += [("attention-attention-query_norm-scale", "self_attn.q_norm.weight"),
                   ("attention-attention-key_norm-scale", "self_attn.k_norm.weight")]
  if is_quantized:
    local += [("attention-in_proj_qkvz-kernel_scale",
               ("linear_attn.in_proj_qkv.weight_scale_inv", "linear_attn.in_proj_z.weight_scale_inv")),
              ("attention-out_proj-kernel_scale", "linear_attn.out_proj.weight_scale_inv")]
    global_specs += [(f"attention-attention-{mt}-kernel_scale", f"self_attn.{hf}_proj.weight_scale_inv")
                     for mt, hf in (("query", "q"), ("key", "k"), ("value", "v"), ("out", "o"))]

  def hf_entry(layer, suffix):
    if isinstance(suffix, tuple):
      return tuple(f"{hf_prefix}.layers.{layer}.{s}" for s in suffix)
    return f"{hf_prefix}.layers.{layer}.{suffix}"

  fused = num_experts > 1 and not is_quantized
  if scan_layers:
    blocks = num_main_layers // cycle
    groups = [("params-decoder-layers-local_layers", local,
               [[b * cycle + p for p in range(cycle - 1)] for b in range(blocks)]),
              ("params-decoder-layers-global_layer", global_specs, [b * cycle + cycle - 1 for b in range(blocks)])]
    remainder = list(range(blocks * cycle, num_main_layers))
    if remainder:
      groups.append(("params-decoder-layers_remainder-local_layers", local, remainder))
    for prefix, specs, layers in groups:
      def entries(suffix):
        return [[hf_entry(i, suffix) for i in layer] if isinstance(layer, list) else hf_entry(layer, suffix)
                for layer in layers]
      for subkey, suffix in specs:
        mapping[f"{prefix}-{subkey}"] = entries(suffix)
      for subkey, suffix in experts:
        mapping[f"{prefix}-{subkey}"] = [entries(f"mlp.experts.{e}.{suffix}") for e in range(num_experts)]
      if fused:
        mapping[(f"{prefix}-mlp-routed_experts-wi_0", f"{prefix}-mlp-routed_experts-wi_1")] = entries(
            "mlp.experts.gate_up_proj")
  else:
    for i in range(num_main_layers):
      prefix = f"params-decoder-layers_{i}"
      specs = global_specs if (i % cycle + 1) % cycle == 0 else local
      for subkey, suffix in specs:
        mapping[f"{prefix}-{subkey}"] = hf_entry(i, suffix)
      for subkey, suffix in experts:
        mapping[f"{prefix}-{subkey}"] = [hf_entry(i, f"mlp.experts.{e}.{suffix}") for e in range(num_experts)]
      if fused:
        mapping[(f"{prefix}-mlp-routed_experts-wi_0", f"{prefix}-mlp-routed_experts-wi_1")] = hf_entry(
            i, "mlp.experts.gate_up_proj")
  if maxtext_config.use_multimodal and "vision_config" in config:
    encoder = "params-vision_encoder-Qwen3_5MoeVisionEncoder_0"
    projector = "params-vision_encoder-Qwen3_5MoeVisionProjector_0-merger"
    mapping[f"{encoder}-patch_embed-proj-kernel"] = "model.visual.patch_embed.proj.weight"
    mapping[f"{encoder}-patch_embed-proj-bias"] = "model.visual.patch_embed.proj.bias"
    mapping[f"{encoder}-pos_embed_interpolate-pos_embed"] = "model.visual.pos_embed.weight"
    for i in range(config["vision_config"]["depth"]):
      prefix = f"{encoder}-blocks_{i}"
      hf = f"model.visual.blocks.{i}"
      for mt, suffix in (("ln1-scale", "norm1.weight"), ("ln1-bias", "norm1.bias"),
                         ("ln2-scale", "norm2.weight"), ("ln2-bias", "norm2.bias"),
                         ("attn-attn-out-kernel", "attn.proj.weight"), ("attn-attn-out-bias", "attn.proj.bias"),
                         ("mlp-kernel", "mlp.linear_fc1.weight"), ("mlp-bias", "mlp.linear_fc1.bias"),
                         ("mlp_out-kernel", "mlp.linear_fc2.weight"), ("mlp_out-bias", "mlp.linear_fc2.bias")):
        mapping[f"{prefix}-{mt}"] = f"{hf}.{suffix}"
      for part in ("query", "key", "value"):
        mapping[f"{prefix}-attn-attn-{part}-kernel"] = f"{hf}.attn.qkv.weight"
        mapping[f"{prefix}-attn-attn-{part}-bias"] = f"{hf}.attn.qkv.bias"
    for mt, suffix in (("ln_q-scale", "norm.weight"), ("ln_q-bias", "norm.bias"),
                       ("mlp_0-kernel", "linear_fc1.weight"), ("mlp_0-bias", "linear_fc1.bias"),
                       ("mlp_2-kernel", "linear_fc2.weight"), ("mlp_2-bias", "linear_fc2.bias")):
      mapping[f"{projector}-{mt}"] = f"model.visual.merger.{suffix}"
  return mapping


def QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def transpose(input_tensor, target_shape=None):
    return input_tensor.T

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def permute_conv(input_tensor, target_shape=None):
    return input_tensor.transpose(2, 1, 0)

  def transpose_expert(input_tensor, target_shape=None):
    return input_tensor.transpose(0, 2, 1)

  def process_wi_0_wi_1(input_tensor, target_shape=None):
    if saving_to_hf:
      wi_0, wi_1 = input_tensor
      return np.concatenate([wi_0, wi_1], axis=-1).swapaxes(-1, -2)
    gate, up = np.split(input_tensor, 2, axis=-2)
    return np.stack([gate.swapaxes(-1, -2), up.swapaxes(-1, -2)], axis=-1)

  text_cfg = config.get("text_config", config)
  H_k, H_v = text_cfg["linear_num_key_heads"], text_cfg["linear_num_value_heads"]
  D_k, D_v = text_cfg["linear_key_head_dim"], text_cfg["linear_value_head_dim"]
  V_per_K = H_v // H_k

  def concat_qkvz_and_transpose(input_tensor, target_shape=None):
    if saving_to_hf:
      t_r = input_tensor.T.reshape(H_k, D_k + D_k + V_per_K * D_v + V_per_K * D_v, -1)
      q = t_r[:, :D_k, :].reshape(H_k * D_k, -1)
      k = t_r[:, D_k:2 * D_k, :].reshape(H_k * D_k, -1)
      v = t_r[:, 2 * D_k:2 * D_k + V_per_K * D_v, :].reshape(H_v * D_v, -1)
      z = t_r[:, 2 * D_k + V_per_K * D_v:, :].reshape(H_v * D_v, -1)
      return np.concatenate([q, k, v], axis=0), z
    qkv, z = input_tensor
    q = qkv[:H_k * D_k, :].reshape(H_k, D_k, -1)
    k = qkv[H_k * D_k:2 * H_k * D_k, :].reshape(H_k, D_k, -1)
    v = qkv[2 * H_k * D_k:, :].reshape(H_k, V_per_K * D_v, -1)
    z = z.reshape(H_k, V_per_K * D_v, -1)
    return np.concatenate([q, k, v, z], axis=1).reshape(-1, qkv.shape[-1]).T

  raw_block_size = getattr(maxtext_config, "weight_block_size", None)
  if raw_block_size is None and isinstance(config, dict):
    raw_block_size = config.get("quantization_config", {}).get("weight_block_size", 128)
  weight_block_size = raw_block_size[0] if isinstance(raw_block_size, (list, tuple)) else (raw_block_size or 128)

  def concat_qkvz_scales_and_transpose(input_tensor, target_shape=None):
    d_k = max(1, D_k // weight_block_size)
    d_v = max(1, V_per_K * D_v // weight_block_size)
    if saving_to_hf:
      t_m = input_tensor.T
      t_r = t_m.reshape(H_k, -1, t_m.shape[-1])
      q = t_r[:, :d_k, :].reshape(H_k * d_k, -1)
      k = t_r[:, d_k:2 * d_k, :].reshape(H_k * d_k, -1)
      v = t_r[:, 2 * d_k:2 * d_k + d_v, :].reshape(H_v * max(1, D_v // weight_block_size), -1)
      z = t_r[:, 2 * d_k + d_v:, :].reshape(H_v * max(1, D_v // weight_block_size), -1)
      return np.concatenate([q, k, v], axis=0), z
    qkv, z = input_tensor
    q = qkv[:H_k * d_k, :].reshape(H_k, d_k, -1)
    k = qkv[H_k * d_k:2 * H_k * d_k, :].reshape(H_k, d_k, -1)
    v = qkv[2 * H_k * d_k:, :].reshape(H_k, d_v, -1)
    z = z.reshape(H_k, d_v, -1)
    return np.concatenate([q, k, v, z], axis=1).reshape(-1, qkv.shape[-1]).T

  def concat_ba_and_transpose(input_tensor, target_shape=None):
    if saving_to_hf:
      t_r = input_tensor.T.reshape(H_k, V_per_K * 2, -1)
      return t_r[:, :V_per_K, :].reshape(H_v, -1), t_r[:, V_per_K:, :].reshape(H_v, -1)
    b, a = input_tensor
    return np.concatenate([b.reshape(H_k, V_per_K, -1), a.reshape(H_k, V_per_K, -1)], axis=1).reshape(
        -1, b.shape[-1]).T

  is_quantized = getattr(maxtext_config, "weight_dtype", None) == "float8_e4m3fn"
  cycle = maxtext_config.inhomogeneous_layer_cycle_interval
  num_layers = getattr(maxtext_config, "num_decoder_layers", None) or config["text_config"]["num_hidden_layers"]
  num_experts = getattr(maxtext_config, "num_experts", None) or text_cfg.get("num_experts", 1)
  hooks = {"params-decoder-logits_dense-kernel": transpose}
  if scan_layers:
    prefixes = [("params-decoder-layers-local_layers", False), ("params-decoder-layers-global_layer", True)]
    if num_layers % cycle:
      prefixes.append(("params-decoder-layers_remainder-local_layers", False))
  else:
    prefixes = [(f"params-decoder-layers_{i}", (i % cycle + 1) % cycle == 0) for i in range(num_layers)]
  for prefix, full in prefixes:
    if full:
      for key in ("query", "key", "value", "out"):
        hooks[f"{prefix}-attention-attention-{key}-kernel"] = reshape_kernel
        if is_quantized:
          hooks[f"{prefix}-attention-attention-{key}-kernel_scale"] = reshape_kernel
    else:
      hooks[f"{prefix}-attention-in_proj_qkvz-kernel"] = concat_qkvz_and_transpose
      hooks[f"{prefix}-attention-in_proj_ba-kernel"] = concat_ba_and_transpose
      hooks[f"{prefix}-attention-out_proj-kernel"] = transpose
      hooks[f"{prefix}-attention-conv1d-kernel"] = permute_conv
      if is_quantized:
        hooks[f"{prefix}-attention-in_proj_qkvz-kernel_scale"] = concat_qkvz_scales_and_transpose
        hooks[f"{prefix}-attention-out_proj-kernel_scale"] = transpose
    mlp = f"{prefix}-mlp"
    if num_experts > 1:
      for key in ("routed_experts-gate-kernel", "shared_expert-wi_0-kernel", "shared_expert-wi_1-kernel",
                  "shared_expert-wo-kernel", "shared_expert_gate-kernel"):
        hooks[f"{mlp}-{key}"] = transpose
      if is_quantized:
        for key in ("shared_expert-wi_0-kernel_scale", "shared_expert-wi_1-kernel_scale",
                    "shared_expert-wo-kernel_scale", "routed_experts-wi_0", "routed_experts-wi_0_scale",
                    "routed_experts-wi_1", "routed_experts-wi_1_scale", "routed_experts-wo", "routed_experts-wo_scale"):
          hooks[f"{mlp}-{key}"] = transpose
      else:
        hooks[(f"{mlp}-routed_experts-wi_0", f"{mlp}-routed_experts-wi_1")] = process_wi_0_wi_1
        hooks[f"{mlp}-routed_experts-wo"] = transpose_expert
    else:
      for key in ("wi_0", "wi_1", "wo"):
        hooks[f"{mlp}-{key}-kernel"] = reshape_kernel
  vision = config.get("vision_config", None)
  if vision and maxtext_config.use_multimodal:
    hidden_size = vision["hidden_size"]

    def reshape_conv3d_patch_embed(input_tensor, target_shape):
      if saving_to_hf:
        return input_tensor.transpose(4, 3, 0, 1, 2)
      return input_tensor.transpose(2, 3, 4, 1, 0)

    def split_qkv(input_tensor, target_shape, index, bias):
      if saving_to_hf:
        raise NotImplementedError("Use fusion hook for MaxText->HF")
      end = (index + 1) * hidden_size if index < 2 else None
      part = input_tensor[index * hidden_size:end]
      return part.reshape(target_shape) if bias else part.T.reshape(target_shape)

    def reshape_vision_attn_out(input_tensor, target_shape):
      if saving_to_hf:
        return input_tensor.reshape(hidden_size, hidden_size).T
      return input_tensor.T.reshape(target_shape)

    encoder = "params-vision_encoder-Qwen3_5MoeVisionEncoder_0"
    hooks[f"{encoder}-patch_embed-proj-kernel"] = reshape_conv3d_patch_embed
    for i in range(vision["depth"]):
      prefix = f"{encoder}-blocks_{i}"
      for index, part in enumerate(("query", "key", "value")):
        hooks[f"{prefix}-attn-attn-{part}-kernel"] = (
            lambda tensor, shape, index=index: split_qkv(tensor, shape, index, False))
        hooks[f"{prefix}-attn-attn-{part}-bias"] = (
            lambda tensor, shape, index=index: split_qkv(tensor, shape, index, True))
      hooks[f"{prefix}-attn-attn-out-kernel"] = reshape_vision_attn_out
      hooks[f"{prefix}-mlp-kernel"] = reshape_kernel
      hooks[f"{prefix}-mlp_out-kernel"] = reshape_kernel
    projector = "params-vision_encoder-Qwen3_5MoeVisionProjector_0-merger"
    hooks[f"{projector}-mlp_0-kernel"] = reshape_kernel
    hooks[f"{projector}-mlp_2-kernel"] = reshape_kernel
  return hooks
