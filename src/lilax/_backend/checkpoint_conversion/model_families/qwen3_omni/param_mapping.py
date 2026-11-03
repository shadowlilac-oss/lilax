"""Qwen3-Omni MoE parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np

from lilax._backend.checkpoint_conversion.model_families.qwen.param_mapping import (
    QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
)


def QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  thinker = config["thinker_config"]
  text = thinker["text_config"]
  mapping = QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(
      {"num_hidden_layers": text["num_hidden_layers"], "num_experts": text.get("num_experts", 0)},
      maxtext_config, scan_layers,
  )

  def add_prefix(value):
    if isinstance(value, list):
      return [add_prefix(v) for v in value]
    return f"thinker.{value}"

  mapping = {key: add_prefix(value) for key, value in mapping.items()}
  vision = thinker["vision_config"]
  vision_base = "params-vision_encoder-Qwen3OmniMoeVisionEncoder_0"
  projector = "params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger"
  mapping[f"{vision_base}-patch_embed-proj-kernel"] = "thinker.visual.patch_embed.proj.weight"
  mapping[f"{vision_base}-patch_embed-proj-bias"] = "thinker.visual.patch_embed.proj.bias"
  mapping[f"{vision_base}-pos_embed_interpolate-pos_embed"] = "thinker.visual.pos_embed.weight"
  vision_params = [
      ("ln1-scale", "norm1.weight"), ("ln1-bias", "norm1.bias"),
      ("ln2-scale", "norm2.weight"), ("ln2-bias", "norm2.bias"),
      ("attn-attn-query-kernel", "attn.qkv.weight"), ("attn-attn-query-bias", "attn.qkv.bias"),
      ("attn-attn-key-kernel", "attn.qkv.weight"), ("attn-attn-key-bias", "attn.qkv.bias"),
      ("attn-attn-value-kernel", "attn.qkv.weight"), ("attn-attn-value-bias", "attn.qkv.bias"),
      ("attn-attn-out-kernel", "attn.proj.weight"), ("attn-attn-out-bias", "attn.proj.bias"),
      ("mlp-kernel", "mlp.linear_fc1.weight"), ("mlp-bias", "mlp.linear_fc1.bias"),
      ("mlp_out-kernel", "mlp.linear_fc2.weight"), ("mlp_out-bias", "mlp.linear_fc2.bias"),
  ]
  for i in range(vision["depth"]):
    for mt, hf in vision_params:
      mapping[f"{vision_base}-blocks_{i}-{mt}"] = f"thinker.visual.blocks.{i}.{hf}"
  merger_params = [
      ("ln_q-scale", "ln_q.weight"), ("ln_q-bias", "ln_q.bias"),
      ("mlp_0-kernel", "mlp.0.weight"), ("mlp_0-bias", "mlp.0.bias"),
      ("mlp_2-kernel", "mlp.2.weight"), ("mlp_2-bias", "mlp.2.bias"),
  ]
  for i, _ in enumerate(vision.get("deepstack_visual_indexes", [8, 16, 24])):
    for mt, hf in merger_params:
      mapping[f"{vision_base}-merger_{i}-{mt}"] = f"thinker.visual.merger_list.{i}.{hf}"
  for mt, hf in merger_params:
    mapping[f"{projector}-{mt}"] = f"thinker.visual.merger.{hf}"

  audio_base = "params-audio_encoder-Qwen3OmniAudioEncoder_0"
  for i in range(1, 4):
    mapping[f"{audio_base}-conv2d{i}-kernel"] = f"thinker.audio_tower.conv2d{i}.weight"
    mapping[f"{audio_base}-conv2d{i}-bias"] = f"thinker.audio_tower.conv2d{i}.bias"
  mapping[f"{audio_base}-conv_out-kernel"] = "thinker.audio_tower.conv_out.weight"
  audio_params = [
      ("input_layer_norm-scale", "self_attn_layer_norm.weight"),
      ("input_layer_norm-bias", "self_attn_layer_norm.bias"),
      ("post_attention_layer_norm-scale", "final_layer_norm.weight"),
      ("post_attention_layer_norm-bias", "final_layer_norm.bias"),
      ("self_attention_audio-query-kernel", "self_attn.q_proj.weight"),
      ("self_attention_audio-query-bias", "self_attn.q_proj.bias"),
      ("self_attention_audio-key-kernel", "self_attn.k_proj.weight"),
      ("self_attention_audio-key-bias", "self_attn.k_proj.bias"),
      ("self_attention_audio-value-kernel", "self_attn.v_proj.weight"),
      ("self_attention_audio-value-bias", "self_attn.v_proj.bias"),
      ("self_attention_audio-out-kernel", "self_attn.out_proj.weight"),
      ("self_attention_audio-out-bias", "self_attn.out_proj.bias"),
      ("AudioMLP-wi-kernel", "fc1.weight"), ("AudioMLP-wi-bias", "fc1.bias"),
      ("AudioMLP-wo-kernel", "fc2.weight"), ("AudioMLP-wo-bias", "fc2.bias"),
  ]
  for i in range(thinker["audio_config"]["encoder_layers"]):
    for mt, hf in audio_params:
      mapping[f"{audio_base}-layers_{i}-{mt}"] = f"thinker.audio_tower.layers.{i}.{hf}"
  mapping[f"{audio_base}-layernorm_post-scale"] = "thinker.audio_tower.ln_post.weight"
  mapping[f"{audio_base}-layernorm_post-bias"] = "thinker.audio_tower.ln_post.bias"
  for i in (1, 2):
    mapping[f"params-audio_encoder-Qwen3OmniAudioProjector_0-proj{i}-kernel"] = f"thinker.audio_tower.proj{i}.weight"
    mapping[f"params-audio_encoder-Qwen3OmniAudioProjector_0-proj{i}-bias"] = f"thinker.audio_tower.proj{i}.bias"
  return mapping


def QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  thinker = config["thinker_config"]
  text = thinker["text_config"]
  hooks = QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      {"num_hidden_layers": text["num_hidden_layers"], "num_experts": text.get("num_experts", 0)},
      maxtext_config, scan_layers, saving_to_hf,
  )
  vision = thinker["vision_config"]
  hidden_size = vision["hidden_size"]
  audio = thinker["audio_config"]
  audio_hidden_size = audio["d_model"]

  def reshape_kernel(x, target_shape):
    if saving_to_hf:
      return x.reshape(np.flip(np.array(target_shape))).T
    return x.T.reshape(target_shape)

  def reshape_conv3d(x, target_shape):
    if saving_to_hf:
      return x.transpose(4, 3, 0, 1, 2)
    return x.transpose(2, 3, 4, 1, 0)

  def split_query(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[:hidden_size, :].T.reshape(target_shape)

  def split_key(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[hidden_size : 2 * hidden_size, :].T.reshape(target_shape)

  def split_value(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[2 * hidden_size :, :].T.reshape(target_shape)

  def split_query_bias(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[:hidden_size].reshape(target_shape)

  def split_key_bias(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[hidden_size : 2 * hidden_size].reshape(target_shape)

  def split_value_bias(x, target_shape):
    if saving_to_hf:
      raise NotImplementedError("Use fusion hook for MaxText->HF")
    return x[2 * hidden_size :].reshape(target_shape)

  def reshape_vision_out(x, target_shape):
    if saving_to_hf:
      return x.reshape(hidden_size, hidden_size).T
    return x.T.reshape(target_shape)

  vision_base = "params-vision_encoder-Qwen3OmniMoeVisionEncoder_0"
  hooks[f"{vision_base}-patch_embed-proj-kernel"] = reshape_conv3d
  for i in range(vision["depth"]):
    prefix = f"{vision_base}-blocks_{i}"
    for name, kernel, bias in (("query", split_query, split_query_bias), ("key", split_key, split_key_bias),
                               ("value", split_value, split_value_bias)):
      hooks[f"{prefix}-attn-attn-{name}-kernel"] = kernel
      hooks[f"{prefix}-attn-attn-{name}-bias"] = bias
    hooks[f"{prefix}-attn-attn-out-kernel"] = reshape_vision_out
    hooks[f"{prefix}-mlp-kernel"] = reshape_kernel
    hooks[f"{prefix}-mlp_out-kernel"] = reshape_kernel
  for i, _ in enumerate(vision.get("deepstack_visual_indexes", [8, 16, 24])):
    for layer in (0, 2):
      hooks[f"{vision_base}-merger_{i}-mlp_{layer}-kernel"] = reshape_kernel
  for layer in (0, 2):
    hooks[f"params-vision_encoder-Qwen3OmniMoeVisionProjector_0-merger-mlp_{layer}-kernel"] = reshape_kernel

  def reshape_conv2d(x, target_shape):
    if saving_to_hf:
      return x.transpose(3, 2, 0, 1)
    return x.transpose(2, 3, 1, 0)

  def reshape_audio_attention(x, target_shape):
    if saving_to_hf:
      return x.reshape(audio_hidden_size, audio_hidden_size).T
    return x.T.reshape(target_shape)

  def reshape_audio_bias(x, target_shape):
    if saving_to_hf:
      return x.reshape(audio_hidden_size)
    return x.reshape(target_shape)

  audio_base = "params-audio_encoder-Qwen3OmniAudioEncoder_0"
  for i in range(1, 4):
    hooks[f"{audio_base}-conv2d{i}-kernel"] = reshape_conv2d
  hooks[f"{audio_base}-conv_out-kernel"] = reshape_kernel
  for i in range(audio["encoder_layers"]):
    prefix = f"{audio_base}-layers_{i}"
    for name in ("query", "key", "value"):
      hooks[f"{prefix}-self_attention_audio-{name}-kernel"] = reshape_audio_attention
      hooks[f"{prefix}-self_attention_audio-{name}-bias"] = reshape_audio_bias
    hooks[f"{prefix}-self_attention_audio-out-kernel"] = reshape_audio_attention
    hooks[f"{prefix}-AudioMLP-wi-kernel"] = reshape_kernel
    hooks[f"{prefix}-AudioMLP-wo-kernel"] = reshape_kernel
  for i in (1, 2):
    hooks[f"params-audio_encoder-Qwen3OmniAudioProjector_0-proj{i}-kernel"] = reshape_kernel
  return hooks
