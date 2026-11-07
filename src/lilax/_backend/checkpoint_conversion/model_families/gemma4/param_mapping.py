"""Gemma 4 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np
import jax.numpy as jnp


def GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  vcfg = config.get("vision_config", {})
  text_base = "model.language_model" if vcfg else "model"
  num_experts = tcfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  mapping = {
      "params-token_embedder-embedding": f"{text_base}.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": f"{text_base}.norm.weight",
  }
  if vcfg:
    mapping.update({
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel":
            "model.vision_tower.patch_embedder.input_proj.weight",
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param":
            "model.vision_tower.patch_embedder.position_embedding_table",
        "params-vision_encoder-Gemma4VisionProjector_0-projection-kernel":
            "model.embed_vision.embedding_projection.weight",
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_scale": "model.vision_tower.std_scale",
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-std_bias": "model.vision_tower.std_bias",
    })
    vision_specs = [
        ("attention-query-kernel", "self_attn.q_proj.linear.weight"),
        ("attention-key-kernel", "self_attn.k_proj.linear.weight"),
        ("attention-value-kernel", "self_attn.v_proj.linear.weight"),
        ("attention-out-kernel", "self_attn.o_proj.linear.weight"),
        ("attention-query_norm-scale", "self_attn.q_norm.weight"),
        ("attention-key_norm-scale", "self_attn.k_norm.weight"),
        ("pre_attention_norm-scale", "input_layernorm.weight"),
        ("post_attention_norm-scale", "post_attention_layernorm.weight"),
        ("pre_ffw_norm-scale", "pre_feedforward_layernorm.weight"),
        ("post_ffw_norm-scale", "post_feedforward_layernorm.weight"),
        ("mlp-wi_0-kernel", "mlp.gate_proj.linear.weight"),
        ("mlp-wi_1-kernel", "mlp.up_proj.linear.weight"),
        ("mlp-wo-kernel", "mlp.down_proj.linear.weight"),
    ]
    for i in range(vcfg.get("num_hidden_layers", 0)):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      for mt, hf in vision_specs:
        mapping[f"{prefix}-{mt}"] = f"model.vision_tower.encoder.layers.{i}.{hf}"

  specs = [
      ("self_attention-query-kernel", "self_attn.q_proj.weight", "always"),
      ("self_attention-key-kernel", "self_attn.k_proj.weight", "always"),
      ("self_attention-out-kernel", "self_attn.o_proj.weight", "always"),
      ("self_attention-query_norm-scale", "self_attn.q_norm.weight", "always"),
      ("self_attention-key_norm-scale", "self_attn.k_norm.weight", "always"),
      ("pre_self_attention_norm-scale", "input_layernorm.weight", "always"),
      ("post_self_attention_norm-scale", "post_attention_layernorm.weight", "always"),
      ("pre_ffw_norm-scale", "pre_feedforward_layernorm.weight", "always"),
      ("post_ffw_norm-scale", "post_feedforward_layernorm.weight", "always"),
      ("mlp-pre_feedforward_layernorm_2-scale", "pre_feedforward_layernorm_2.weight", "moe"),
      ("mlp-post_feedforward_layernorm_1-scale", "post_feedforward_layernorm_1.weight", "moe"),
      ("mlp-post_feedforward_layernorm_2-scale", "post_feedforward_layernorm_2.weight", "moe"),
      ("mlp-pre_forward_scale_2", "router.scale", "moe"),
      ("mlp-wi_0-kernel", "mlp.gate_proj.weight", "dense"),
      ("mlp-wi_1-kernel", "mlp.up_proj.weight", "dense"),
      ("mlp-wo-kernel", "mlp.down_proj.weight", "dense"),
      ("mlp-moe_block-MoeBlock_0-gate-kernel", "router.proj.weight", "moe"),
      ("mlp-moe_block-MoeBlock_0-wi_0", "experts.gate_up_proj", "moe"),
      ("mlp-moe_block-MoeBlock_0-wi_1", "experts.gate_up_proj", "moe"),
      ("mlp-moe_block-MoeBlock_0-wo", "experts.down_proj", "moe"),
      ("mlp-moe_block-MoeBlock_0-per_expert_scale", "router.per_expert_scale", "moe"),
      ("mlp-moe_block-shared_experts-wi_0-kernel", "mlp.gate_proj.weight", "moe"),
      ("mlp-moe_block-shared_experts-wi_1-kernel", "mlp.up_proj.weight", "moe"),
      ("mlp-moe_block-shared_experts-wo-kernel", "mlp.down_proj.weight", "moe"),
      ("layer_scalar", "layer_scalar", "always"),
  ]
  if maxtext_config.v_norm_with_scale:
    specs.append(("self_attention-value_norm-scale", "self_attn.v_norm.weight", "always"))
  specs = [(mt, hf) for mt, hf, gate in specs
           if gate == "always" or (gate == "dense" and num_experts == 1) or (gate == "moe" and num_experts > 1)]

  def hf_layer(i, suffix):
    return f"{text_base}.layers.{i}.{suffix}"

  if scan_layers:
    num_blocks = nlayers // 6
    for mt, hf in specs:
      mapping[f"params-decoder-scanned_blocks-local_layers-{mt}"] = [
          [hf_layer(b * 6 + i, hf) for i in range(5)] for b in range(num_blocks)
      ]
      mapping[f"params-decoder-scanned_blocks-global_layer-{mt}"] = [
          hf_layer(b * 6 + 5, hf) for b in range(num_blocks)
      ]
    mapping["params-decoder-scanned_blocks-local_layers-self_attention-value-kernel"] = [
        [hf_layer(b * 6 + i, "self_attn.v_proj.weight") for i in range(5)] for b in range(num_blocks)
    ]
    if not maxtext_config.share_kv_projections:
      mapping["params-decoder-scanned_blocks-global_layer-self_attention-value-kernel"] = [
          hf_layer(b * 6 + 5, "self_attn.v_proj.weight") for b in range(num_blocks)
      ]
    if nlayers % 6:
      prefix = "params-decoder-layers_remainder-local_layers"
      for mt, hf in specs:
        mapping[f"{prefix}-{mt}"] = [hf_layer(num_blocks * 6 + i, hf) for i in range(nlayers % 6)]
      mapping[f"{prefix}-self_attention-value-kernel"] = [
          hf_layer(num_blocks * 6 + i, "self_attn.v_proj.weight") for i in range(nlayers % 6)
      ]
  else:
    for i in range(nlayers):
      prefix = f"params-decoder-layers_{i}"
      for mt, hf in specs:
        mapping[f"{prefix}-{mt}"] = hf_layer(i, hf)
      if not maxtext_config.share_kv_projections or i % 6 != 5:
        mapping[f"{prefix}-self_attention-value-kernel"] = hf_layer(i, "self_attn.v_proj.weight")
  return mapping


def GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  share_kv_projections = maxtext_config.share_kv_projections
  hooks = {}

  def pad_hf_embedding_layer(input_tensor, target_shape):
    normalizer = np.dtype("float32").type(tcfg["hidden_size"] ** 0.5)
    if saving_to_hf:
      target_tensor = input_tensor[: target_shape[0], : target_shape[1]] / normalizer
      return target_tensor.astype(input_tensor.dtype)
    target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
    return (target_tensor * normalizer).astype(input_tensor.dtype)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def split_moe_wi0(input_tensor, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Saving to HF for fused gate_up_proj requires custom concat hook.")
    _, two_ff, _ = input_tensor.shape
    return input_tensor[:, : two_ff // 2, :].transpose(0, 2, 1)

  def split_moe_wi1(input_tensor, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Saving to HF for fused gate_up_proj requires custom concat hook.")
    _, two_ff, _ = input_tensor.shape
    return input_tensor[:, two_ff // 2 :, :].transpose(0, 2, 1)

  def reshape_moe_wo(input_tensor, target_shape):
    return input_tensor.transpose(0, 2, 1)

  def moe_gate_up_hook(weight_list, target_shape):
    wi_0 = jnp.asarray(weight_list[0])
    wi_1 = jnp.asarray(weight_list[1])
    return jnp.swapaxes(jnp.concatenate([wi_0, wi_1], axis=-1), -2, -1)

  hooks["params-token_embedder-embedding"] = pad_hf_embedding_layer
  hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm_layer
  kernel_keys = [
      "self_attention-query-kernel", "self_attention-key-kernel", "self_attention-value-kernel",
      "self_attention-out-kernel", "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
      "mlp-moe_block-shared_experts-wi_0-kernel", "mlp-moe_block-shared_experts-wi_1-kernel",
      "mlp-moe_block-shared_experts-wo-kernel",
  ]
  moe_kernel_keys = ["mlp-moe_block-MoeBlock_0-gate-kernel"]
  norm_keys = ["self_attention-query_norm-scale", "self_attention-key_norm-scale"]
  if maxtext_config.v_norm_with_scale:
    norm_keys.append("self_attention-value_norm-scale")
  norm_keys.extend([
      "pre_self_attention_norm-scale", "post_self_attention_norm-scale", "pre_ffw_norm-scale", "post_ffw_norm-scale",
  ])
  num_experts = tcfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  if num_experts > 1:
    norm_keys.extend([
        "mlp-pre_feedforward_layernorm_2-scale", "mlp-post_feedforward_layernorm_1-scale",
        "mlp-post_feedforward_layernorm_2-scale",
    ])

  vcfg = config.get("vision_config", {})
  if vcfg:
    nvis = vcfg.get("num_hidden_layers", 0)

    def reshape_vision_patch(input_tensor, target_shape):
      return input_tensor.T

    def reshape_pos_emb(input_tensor, target_shape):
      return input_tensor.transpose(1, 0, 2)

    hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel"] = reshape_vision_patch
    hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param"] = reshape_pos_emb
    hooks["params-vision_encoder-Gemma4VisionProjector_0-projection-kernel"] = reshape_kernel
    for i in range(nvis):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      for key in ("attention-query-kernel", "attention-key-kernel", "attention-value-kernel", "attention-out-kernel",
                  "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel"):
        hooks[f"{prefix}-{key}"] = reshape_kernel
      for key in ("attention-query_norm-scale", "attention-key_norm-scale", "pre_attention_norm-scale",
                  "post_attention_norm-scale", "pre_ffw_norm-scale", "post_ffw_norm-scale"):
        hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer

  def attach_hooks(prefix, is_global):
    for key in kernel_keys:
      if share_kv_projections and is_global and key == "self_attention-value-kernel":
        continue
      hooks[f"{prefix}-{key}"] = reshape_kernel
    for key in moe_kernel_keys:
      hooks[f"{prefix}-{key}"] = reshape_kernel
    for key in norm_keys:
      hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer
    if saving_to_hf and num_experts > 1:
      wi0_key = f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"
      wi1_key = f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"
      hooks[(wi0_key, wi1_key)] = moe_gate_up_hook
    else:
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_0"] = split_moe_wi0
      hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wi_1"] = split_moe_wi1
    hooks[f"{prefix}-mlp-moe_block-MoeBlock_0-wo"] = reshape_moe_wo

  if scan_layers:
    attach_hooks("params-decoder-scanned_blocks-local_layers", False)
    attach_hooks("params-decoder-scanned_blocks-global_layer", True)
    if nlayers % 6:
      attach_hooks("params-decoder-layers_remainder-local_layers", False)
  else:
    for i in range(nlayers):
      attach_hooks(f"params-decoder-layers_{i}", i % 6 == 5)
  return hooks


def GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  if scan_layers:
    raise NotImplementedError("Scan layers is not supported for the gemma4_small decoder block.")
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  num_kv_shared = tcfg.get("num_kv_shared_layers", 0) or 0
  first_shared = max(0, nlayers - num_kv_shared) if num_kv_shared > 0 else nlayers
  vcfg = config.get("vision_config", {})
  text_base = "model.language_model" if vcfg else "model"
  mapping = {
      "params-token_embedder-embedding": f"{text_base}.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": f"{text_base}.norm.weight",
  }
  ple_dim = tcfg.get("hidden_size_per_layer_input", 0) or 0
  if ple_dim > 0:
    mapping.update({
        "params-decoder-per_layer_embedder-embed_tokens_per_layer": f"{text_base}.embed_tokens_per_layer.weight",
        "params-decoder-per_layer_embedder-per_layer_model_projection-kernel": f"{text_base}.per_layer_model_projection.weight",
        "params-decoder-per_layer_embedder-per_layer_projection_norm-scale": f"{text_base}.per_layer_projection_norm.weight",
    })
  for lyr in range(nlayers):
    prefix = f"params-decoder-layers_{lyr}"
    hf_prefix = f"{text_base}.layers.{lyr}"
    is_shared = lyr >= first_shared > 0
    mapping.update({
        f"{prefix}-pre_self_attention_norm-scale": f"{hf_prefix}.input_layernorm.weight",
        f"{prefix}-post_self_attention_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
        f"{prefix}-pre_ffw_norm-scale": f"{hf_prefix}.pre_feedforward_layernorm.weight",
        f"{prefix}-post_ffw_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
        f"{prefix}-self_attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.weight",
        f"{prefix}-self_attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
        f"{prefix}-self_attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.weight",
        f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight",
        f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight",
        f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight",
        f"{prefix}-layer_scalar": f"{hf_prefix}.layer_scalar",
    })
    if not is_shared:
      mapping.update({
          f"{prefix}-self_attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.weight",
          f"{prefix}-self_attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.weight",
          f"{prefix}-self_attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
          f"{prefix}-self_attention-value_norm-scale": f"{hf_prefix}.self_attn.v_norm.weight"
          if maxtext_config.v_norm_with_scale else None,
      })
    if ple_dim > 0:
      mapping.update({
          f"{prefix}-per_layer_input_gate-kernel": f"{hf_prefix}.per_layer_input_gate.weight",
          f"{prefix}-per_layer_projection-kernel": f"{hf_prefix}.per_layer_projection.weight",
          f"{prefix}-post_per_layer_input_norm-scale": f"{hf_prefix}.post_per_layer_input_norm.weight",
      })
  if maxtext_config.use_multimodal and vcfg:
    mapping.update({
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-input_projection-kernel":
            "model.vision_tower.patch_embedder.input_proj.weight",
        "params-vision_encoder-Gemma4VisionEncoderLayer_0-vision_entry-pos_emb_param":
            "model.vision_tower.patch_embedder.position_embedding_table",
        "params-vision_encoder-Gemma4VisionProjector_0-projection-kernel": "model.embed_vision.embedding_projection.weight",
    })
    placeholder = "model.vision_tower.encoder.layers.0.input_layernorm.weight"
    for name in ("std_scale", "std_bias"):
      mapping[f"params-vision_encoder-Gemma4VisionEncoderLayer_0-{name}"] = (
          f"model.vision_tower.{name}" if vcfg.get("standardize", False) else placeholder
      )
    for i in range(vcfg.get("num_hidden_layers", 0)):
      prefix = f"params-vision_encoder-Gemma4VisionEncoderLayer_0-layer_{i}"
      hf_prefix = f"model.vision_tower.encoder.layers.{i}"
      mapping.update({
          f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.linear.weight",
          f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.linear.weight",
          f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.linear.weight",
          f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.linear.weight",
          f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
          f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
          f"{prefix}-pre_attention_norm-scale": f"{hf_prefix}.input_layernorm.weight",
          f"{prefix}-post_attention_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
          f"{prefix}-pre_ffw_norm-scale": f"{hf_prefix}.pre_feedforward_layernorm.weight",
          f"{prefix}-post_ffw_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
          f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.linear.weight",
          f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.linear.weight",
          f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.linear.weight",
      })
  return {key: value for key, value in mapping.items() if value is not None}


def GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  if scan_layers:
    raise NotImplementedError("Scan layers is not supported for the gemma4_small decoder block.")
  tcfg = config.get("text_config", config)
  nlayers = tcfg["num_hidden_layers"]
  num_kv_shared = tcfg.get("num_kv_shared_layers", 0) or 0
  first_shared = max(0, nlayers - num_kv_shared) if num_kv_shared > 0 else nlayers
  ple_dim = tcfg.get("hidden_size_per_layer_input", 0) or 0
  vcfg = config.get("vision_config", {})
  hooks = {}

  def pad_hf_embedding_layer(input_tensor, target_shape):
    normalizer = np.dtype("float32").type(tcfg["hidden_size"] ** 0.5)
    if saving_to_hf:
      return (input_tensor[: target_shape[0], : target_shape[1]] / normalizer).astype(input_tensor.dtype)
    target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    target_tensor[: input_tensor.shape[0], : input_tensor.shape[1]] = input_tensor
    return (target_tensor * normalizer).astype(input_tensor.dtype)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def reshape_to_target(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  hooks["params-token_embedder-embedding"] = pad_hf_embedding_layer
  hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm_layer
  if ple_dim > 0:
    hooks["params-decoder-per_layer_embedder-per_layer_model_projection-kernel"] = reshape_kernel
    hooks["params-decoder-per_layer_embedder-per_layer_projection_norm-scale"] = scale_rmsnorm_layer
  for lyr in range(nlayers):
    prefix = f"params-decoder-layers_{lyr}"
    is_shared = lyr >= first_shared > 0
    for name in ("pre_self_attention_norm", "post_self_attention_norm", "pre_ffw_norm", "post_ffw_norm"):
      hooks[f"{prefix}-{name}-scale"] = scale_rmsnorm_layer
    for key in ("query-kernel", "out-kernel"):
      hooks[f"{prefix}-self_attention-{key}"] = reshape_kernel
    hooks[f"{prefix}-self_attention-query_norm-scale"] = scale_rmsnorm_layer
    if not is_shared:
      for key in ("key-kernel", "value-kernel"):
        hooks[f"{prefix}-self_attention-{key}"] = reshape_kernel
      hooks[f"{prefix}-self_attention-key_norm-scale"] = scale_rmsnorm_layer
      if maxtext_config.v_norm_with_scale:
        hooks[f"{prefix}-self_attention-value_norm-scale"] = scale_rmsnorm_layer
    for key in ("wi_0-kernel", "wi_1-kernel", "wo-kernel"):
      hooks[f"{prefix}-mlp-{key}"] = reshape_kernel
    hooks[f"{prefix}-layer_scalar"] = reshape_to_target
    if ple_dim > 0:
      for key in ("per_layer_input_gate-kernel", "per_layer_projection-kernel"):
        hooks[f"{prefix}-{key}"] = reshape_kernel
      hooks[f"{prefix}-post_per_layer_input_norm-scale"] = scale_rmsnorm_layer
  if maxtext_config.use_multimodal and vcfg:
    big_hooks = GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=saving_to_hf)
    for key, fn in big_hooks.items():
      if isinstance(key, str) and key.startswith("params-vision_encoder-"):
        hooks[key] = fn
    if not vcfg.get("standardize", False):
      def make_ones(_input, target_shape):
        return np.ones(target_shape, dtype=np.float32)

      def make_zeros(_input, target_shape):
        return np.zeros(target_shape, dtype=np.float32)

      hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-std_scale"] = make_ones
      hooks["params-vision_encoder-Gemma4VisionEncoderLayer_0-std_bias"] = make_zeros
  return hooks
