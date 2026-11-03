"""Qwen3-VL parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np

from lilax._backend.checkpoint_conversion.model_families.qwen.param_mapping import (
    QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
)


def QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mapping from MaxText to HuggingFace Qwen3-VL weight paths."""
  mapping = {}
  n_layers_text = config["text_config"]["num_hidden_layers"]
  num_experts_text = config["text_config"].get("num_local_experts", 0)
  text_mapping = QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(
      config={"num_hidden_layers": n_layers_text, "num_experts": num_experts_text},
      maxtext_config=maxtext_config,
      scan_layers=scan_layers,
  )

  def replace_prefix(val):
    if isinstance(val, list):
      return [replace_prefix(v) for v in val]
    elif isinstance(val, str):
      return val.replace("model.", "model.language_model.")
    return val

  for key, value in text_mapping.items():
    mapping[key] = replace_prefix(value)

  if num_experts_text > 0 and not scan_layers:
    for i in range(n_layers_text):
      key0 = f"params-decoder-layers_{i}-moe_block-wi_0"
      key1 = f"params-decoder-layers_{i}-moe_block-wi_1"
      key_wo = f"params-decoder-layers_{i}-moe_block-wo"
      if key0 in mapping:
        del mapping[key0]
      if key1 in mapping:
        del mapping[key1]
      mapping[(key0, key1)] = f"model.language_model.layers.{i}.mlp.experts.gate_up_proj"
      mapping[key_wo] = f"model.language_model.layers.{i}.mlp.experts.down_proj"

  if maxtext_config.use_multimodal:
    vision_config = config["vision_config"]
    n_vision_layers = vision_config["depth"]
    mapping["params-vision_encoder-Qwen3VLVisionEncoder_0-patch_embed-proj-kernel"] = "model.visual.patch_embed.proj.weight"
    mapping["params-vision_encoder-Qwen3VLVisionEncoder_0-patch_embed-proj-bias"] = "model.visual.patch_embed.proj.bias"
    mapping["params-vision_encoder-Qwen3VLVisionEncoder_0-pos_embed_interpolate-pos_embed"] = "model.visual.pos_embed.weight"
    for i in range(n_vision_layers):
      prefix = f"params-vision_encoder-Qwen3VLVisionEncoder_0-blocks_{i}"
      hf_prefix = f"model.visual.blocks.{i}"
      mapping[f"{prefix}-ln1-scale"] = f"{hf_prefix}.norm1.weight"
      mapping[f"{prefix}-ln1-bias"] = f"{hf_prefix}.norm1.bias"
      mapping[f"{prefix}-ln2-scale"] = f"{hf_prefix}.norm2.weight"
      mapping[f"{prefix}-ln2-bias"] = f"{hf_prefix}.norm2.bias"
      mapping[(f"{prefix}-attn-attn-query-kernel", f"{prefix}-attn-attn-key-kernel", f"{prefix}-attn-attn-value-kernel")] = f"{hf_prefix}.attn.qkv.weight"
      mapping[(f"{prefix}-attn-attn-query-bias", f"{prefix}-attn-attn-key-bias", f"{prefix}-attn-attn-value-bias")] = f"{hf_prefix}.attn.qkv.bias"
      mapping[f"{prefix}-attn-attn-out-kernel"] = f"{hf_prefix}.attn.proj.weight"
      mapping[f"{prefix}-attn-attn-out-bias"] = f"{hf_prefix}.attn.proj.bias"
      mapping[f"{prefix}-mlp-kernel"] = f"{hf_prefix}.mlp.linear_fc1.weight"
      mapping[f"{prefix}-mlp-bias"] = f"{hf_prefix}.mlp.linear_fc1.bias"
      mapping[f"{prefix}-mlp_out-kernel"] = f"{hf_prefix}.mlp.linear_fc2.weight"
      mapping[f"{prefix}-mlp_out-bias"] = f"{hf_prefix}.mlp.linear_fc2.bias"

    deepstack_indexes = vision_config.get("deepstack_visual_indexes", [5, 11, 17])
    for merger_idx, _ in enumerate(deepstack_indexes):
      prefix = f"params-vision_encoder-Qwen3VLVisionEncoder_0-merger_{merger_idx}"
      hf_prefix = f"model.visual.deepstack_merger_list.{merger_idx}"
      mapping[f"{prefix}-ln_q-scale"] = f"{hf_prefix}.norm.weight"
      mapping[f"{prefix}-ln_q-bias"] = f"{hf_prefix}.norm.bias"
      mapping[f"{prefix}-mlp_0-kernel"] = f"{hf_prefix}.linear_fc1.weight"
      mapping[f"{prefix}-mlp_0-bias"] = f"{hf_prefix}.linear_fc1.bias"
      mapping[f"{prefix}-mlp_2-kernel"] = f"{hf_prefix}.linear_fc2.weight"
      mapping[f"{prefix}-mlp_2-bias"] = f"{hf_prefix}.linear_fc2.bias"

    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-ln_q-scale"] = "model.visual.merger.norm.weight"
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-ln_q-bias"] = "model.visual.merger.norm.bias"
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_0-kernel"] = "model.visual.merger.linear_fc1.weight"
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_0-bias"] = "model.visual.merger.linear_fc1.bias"
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_2-kernel"] = "model.visual.merger.linear_fc2.weight"
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_2-bias"] = "model.visual.merger.linear_fc2.bias"
  return mapping


def QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3-VL."""
  mapping = {}
  n_layers_text = config["text_config"]["num_hidden_layers"]
  num_experts_text = config["text_config"].get("num_local_experts", 0)
  text_hooks = QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      config={"num_hidden_layers": n_layers_text, "num_experts": num_experts_text},
      maxtext_config=maxtext_config, scan_layers=scan_layers, saving_to_hf=saving_to_hf,
  )

  def process_wi_0_wi_1_fused(input_tensor, target_shape=None):
    if saving_to_hf:
      wi_0, wi_1 = input_tensor
      return np.concatenate([wi_0, wi_1], axis=-1)
    gate, up = np.split(input_tensor, 2, axis=-1)
    return np.stack([gate, up], axis=-1)

  mapping.update(text_hooks)
  if num_experts_text > 0 and not scan_layers:
    for i in range(n_layers_text):
      composite_key = (f"params-decoder-layers_{i}-moe_block-wi_0", f"params-decoder-layers_{i}-moe_block-wi_1")
      mapping[composite_key] = process_wi_0_wi_1_fused
      mapping[f"params-decoder-layers_{i}-moe_block-wo"] = None

  if maxtext_config.use_multimodal:
    vision_config = config["vision_config"]
    n_vision_layers = vision_config["depth"]
    hidden_size = vision_config["hidden_size"]

    def reshape_kernel_vision(input_tensor, target_shape):
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      return input_tensor.T.reshape(target_shape)

    def reshape_conv3d_patch_embed(input_tensor, target_shape):
      if saving_to_hf:
        return input_tensor.transpose(4, 3, 0, 1, 2)
      return input_tensor.transpose(2, 3, 4, 1, 0)

    def process_qkv_vision(input_tensor, target_shape=None):
      if saving_to_hf:
        q, k, v = input_tensor
        q_hf = q.reshape(hidden_size, hidden_size).T
        k_hf = k.reshape(hidden_size, hidden_size).T
        v_hf = v.reshape(hidden_size, hidden_size).T
        return np.concatenate([q_hf, k_hf, v_hf], axis=0)
      q_hf = input_tensor[:hidden_size, :]
      k_hf = input_tensor[hidden_size : 2 * hidden_size, :]
      v_hf = input_tensor[2 * hidden_size :, :]
      q_mt = q_hf.T.reshape(target_shape[0])
      k_mt = k_hf.T.reshape(target_shape[1])
      v_mt = v_hf.T.reshape(target_shape[2])
      return np.stack([q_mt, k_mt, v_mt], axis=-1)

    def process_qkv_bias_vision(input_tensor, target_shape=None):
      if saving_to_hf:
        qb, kb, vb = input_tensor
        qb_hf = qb.reshape(hidden_size)
        kb_hf = kb.reshape(hidden_size)
        vb_hf = vb.reshape(hidden_size)
        return np.concatenate([qb_hf, kb_hf, vb_hf], axis=0)
      qb_hf = input_tensor[:hidden_size]
      kb_hf = input_tensor[hidden_size : 2 * hidden_size]
      vb_hf = input_tensor[2 * hidden_size :]
      qb_mt = qb_hf.reshape(target_shape[0])
      kb_mt = kb_hf.reshape(target_shape[1])
      vb_mt = vb_hf.reshape(target_shape[2])
      return np.stack([qb_mt, kb_mt, vb_mt], axis=-1)

    def reshape_vision_attn_out(input_tensor, target_shape):
      if saving_to_hf:
        return input_tensor.reshape(hidden_size, hidden_size).T
      return input_tensor.T.reshape(target_shape)

    mapping["params-vision_encoder-Qwen3VLVisionEncoder_0-patch_embed-proj-kernel"] = reshape_conv3d_patch_embed
    for i in range(n_vision_layers):
      prefix = f"params-vision_encoder-Qwen3VLVisionEncoder_0-blocks_{i}"
      mapping[(f"{prefix}-attn-attn-query-kernel", f"{prefix}-attn-attn-key-kernel", f"{prefix}-attn-attn-value-kernel")] = process_qkv_vision
      mapping[(f"{prefix}-attn-attn-query-bias", f"{prefix}-attn-attn-key-bias", f"{prefix}-attn-attn-value-bias")] = process_qkv_bias_vision
      mapping[f"{prefix}-attn-attn-out-kernel"] = reshape_vision_attn_out
      mapping[f"{prefix}-mlp-kernel"] = reshape_kernel_vision
      mapping[f"{prefix}-mlp_out-kernel"] = reshape_kernel_vision

    deepstack_indexes = vision_config.get("deepstack_visual_indexes", [5, 11, 17])
    for merger_idx, _ in enumerate(deepstack_indexes):
      prefix = f"params-vision_encoder-Qwen3VLVisionEncoder_0-merger_{merger_idx}"
      mapping[f"{prefix}-mlp_0-kernel"] = reshape_kernel_vision
      mapping[f"{prefix}-mlp_2-kernel"] = reshape_kernel_vision
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_0-kernel"] = reshape_kernel_vision
    mapping["params-vision_encoder-Qwen3VLVisionProjector_0-merger-mlp_2-kernel"] = reshape_kernel_vision
  return mapping
