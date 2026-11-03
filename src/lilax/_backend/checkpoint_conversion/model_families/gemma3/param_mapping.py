"""Gemma 3 parameter mappings and transformation hooks for checkpoint conversion."""

import warnings
import numpy as np
import jax
import jax.numpy as jnp


def GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mappings between MaxText and Hugging Face Gemma 3 parameters."""
  ndec = config["text_config"]["num_hidden_layers"]
  nvis = config["vision_config"]["num_hidden_layers"]
  mapping = {
      "params-token_embedder-embedding": "model.language_model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.language_model.norm.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel":
          "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-bias":
          "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding":
          "model.vision_tower.vision_model.embeddings.position_embedding.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-scale":
          "model.vision_tower.vision_model.post_layernorm.weight",
      "params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoder_norm-bias":
          "model.vision_tower.vision_model.post_layernorm.bias",
      "params-vision_encoder-VisionEmbedder_0-mm_input_projection-w":
          "model.multi_modal_projector.mm_input_projection_weight",
      "params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale":
          "model.multi_modal_projector.mm_soft_emb_norm.weight",
  }
  vision_params = [
      ("LayerNorm_0-scale", "layer_norm1.weight"),
      ("LayerNorm_0-bias", "layer_norm1.bias"),
      ("LayerNorm_1-scale", "layer_norm2.weight"),
      ("LayerNorm_1-bias", "layer_norm2.bias"),
      ("MultiHeadDotProductAttention_0-query-kernel", "self_attn.q_proj.weight"),
      ("MultiHeadDotProductAttention_0-query-bias", "self_attn.q_proj.bias"),
      ("MultiHeadDotProductAttention_0-key-kernel", "self_attn.k_proj.weight"),
      ("MultiHeadDotProductAttention_0-key-bias", "self_attn.k_proj.bias"),
      ("MultiHeadDotProductAttention_0-value-kernel", "self_attn.v_proj.weight"),
      ("MultiHeadDotProductAttention_0-value-bias", "self_attn.v_proj.bias"),
      ("MultiHeadDotProductAttention_0-out-kernel", "self_attn.out_proj.weight"),
      ("MultiHeadDotProductAttention_0-out-bias", "self_attn.out_proj.bias"),
      ("MlpBlockViT_0-Dense_0-kernel", "mlp.fc1.weight"),
      ("MlpBlockViT_0-Dense_0-bias", "mlp.fc1.bias"),
      ("MlpBlockViT_0-Dense_1-kernel", "mlp.fc2.weight"),
      ("MlpBlockViT_0-Dense_1-bias", "mlp.fc2.bias"),
  ]
  for index in range(nvis):
    for mx, hf in vision_params:
      key = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{index}-{mx}"
      mapping[key] = f"model.vision_tower.vision_model.encoder.layers.{index}.{hf}"
  text_params = [
      ("pre_self_attention_norm-scale", "input_layernorm.weight"),
      ("post_self_attention_norm-scale", "post_attention_layernorm.weight"),
      ("self_attention-query_norm-scale", "self_attn.q_norm.weight"),
      ("self_attention-key_norm-scale", "self_attn.k_norm.weight"),
      ("pre_ffw_norm-scale", "pre_feedforward_layernorm.weight"),
      ("post_ffw_norm-scale", "post_feedforward_layernorm.weight"),
      ("self_attention-query-kernel", "self_attn.q_proj.weight"),
      ("self_attention-key-kernel", "self_attn.k_proj.weight"),
      ("self_attention-value-kernel", "self_attn.v_proj.weight"),
      ("self_attention-out-kernel", "self_attn.o_proj.weight"),
      ("mlp-wi_0-kernel", "mlp.gate_proj.weight"),
      ("mlp-wi_1-kernel", "mlp.up_proj.weight"),
      ("mlp-wo-kernel", "mlp.down_proj.weight"),
  ]
  if scan_layers:
    attention_pattern_length = 6
    num_remaining = ndec % attention_pattern_length
    num_scanned = ndec - num_remaining
    for block_index in range(attention_pattern_length):
      hf_indices = list(range(block_index, num_scanned, attention_pattern_length))
      for mx, hf in text_params:
        key = f"params-decoder-layers-layers_{block_index}-{mx}"
        mapping[key] = [f"model.language_model.layers.{index}.{hf}" for index in hf_indices]
    if num_remaining > 0:
      for remainder_index in range(num_remaining):
        hf_layer_index = num_scanned + remainder_index
        for mx, hf in text_params:
          key = f"params-decoder-layers_remainder-layers_{remainder_index}-{mx}"
          mapping[key] = f"model.language_model.layers.{hf_layer_index}.{hf}"
  else:
    for index in range(ndec):
      for mx, hf in text_params:
        key = f"params-decoder-layers_{index}-{mx}"
        mapping[key] = f"model.language_model.layers.{index}.{hf}"
  return mapping


def GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Gemma 3 conversion."""
  hooks = {}

  def pad_and_scale_embedding(input_tensor, target_shape):
    source_vocab_size, _ = input_tensor.shape
    target_vocab_size, target_hidden_size = target_shape
    normalizer = np.dtype("bfloat16").type(config["text_config"]["hidden_size"] ** 0.5)
    if saving_to_hf:
      scaled_tensor = (input_tensor / normalizer).astype(input_tensor.dtype)
    else:
      scaled_tensor = (input_tensor * normalizer).astype(input_tensor.dtype)
    if source_vocab_size > target_vocab_size:
      warnings.warn(
          f"source vocab={source_vocab_size} > target vocab={target_vocab_size}, truncate output layer for MaxText."
      )
      output_tensor = scaled_tensor[:target_vocab_size, :]
    elif source_vocab_size < target_vocab_size:
      warnings.warn(f"source vocab={source_vocab_size} < target vocab={target_vocab_size}, pad output layer for MaxText.")
      padding_shape = (target_vocab_size - source_vocab_size, target_hidden_size)
      padding = (
          jnp.zeros(padding_shape, dtype=scaled_tensor.dtype)
          if isinstance(scaled_tensor, jax.Array)
          else np.zeros(padding_shape, dtype=scaled_tensor.dtype)
      )
      output_tensor = (
          jnp.concatenate([scaled_tensor, padding], axis=0)
          if isinstance(scaled_tensor, jax.Array)
          else np.concatenate([scaled_tensor, padding], axis=0)
      )
    else:
      output_tensor = scaled_tensor
    return output_tensor

  def scale_rmsnorm(input_tensor, target_shape):
    if saving_to_hf:
      return (input_tensor - 1.0).reshape(target_shape)
    return (input_tensor + 1.0).reshape(target_shape)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped).T
    else:
      return input_tensor.T.reshape(target_shape)

  def vis_bias(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.flatten()
    else:
      return input_tensor.reshape(target_shape)

  def vision_patch(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.transpose(3, 2, 0, 1)
    else:
      return input_tensor.transpose(2, 3, 1, 0)

  def pos_embed(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.squeeze(0)
    return input_tensor[None, :, :]

  hooks["params-token_embedder-embedding"] = pad_and_scale_embedding
  hooks["params-decoder-decoder_norm-scale"] = scale_rmsnorm
  hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-embedding-kernel"] = vision_patch
  hooks["params-vision_encoder-Gemma3VisionEncoderLayer_0-pos_embedding"] = pos_embed
  hooks["params-vision_encoder-VisionEmbedder_0-mm_input_projection-w"] = lambda tensor, _: tensor
  hooks["params-vision_encoder-VisionEmbedder_0-mm_soft_embedding_norm-scale"] = scale_rmsnorm

  nlayers = config.get("text_config", {}).get("num_hidden_layers", 0)
  if scan_layers:
    attention_pattern_length = 6
    num_remaining = nlayers % attention_pattern_length
    prefixes = [f"params-decoder-layers-layers_{index}-" for index in range(attention_pattern_length)]
    if num_remaining > 0:
      prefixes += [f"params-decoder-layers_remainder-layers_{index}-" for index in range(num_remaining)]
  else:
    prefixes = [f"params-decoder-layers_{index}-" for index in range(nlayers)]
  for prefix in prefixes:
    hooks[prefix + "self_attention-query-kernel"] = reshape_kernel
    hooks[prefix + "self_attention-key-kernel"] = reshape_kernel
    hooks[prefix + "self_attention-value-kernel"] = reshape_kernel
    hooks[prefix + "self_attention-out-kernel"] = reshape_kernel
    for name in [
        "pre_self_attention_norm-scale", "post_self_attention_norm-scale", "self_attention-query_norm-scale",
        "self_attention-key_norm-scale", "pre_ffw_norm-scale", "post_ffw_norm-scale",
    ]:
      hooks[prefix + name] = scale_rmsnorm
    hooks[prefix + "mlp-wi_0-kernel"] = reshape_kernel
    hooks[prefix + "mlp-wi_1-kernel"] = reshape_kernel
    hooks[prefix + "mlp-wo-kernel"] = reshape_kernel

  nvis = config.get("vision_config", {}).get("num_hidden_layers", 0)
  for index in range(nvis):
    base = f"params-vision_encoder-Gemma3VisionEncoderLayer_0-Transformer-encoderblock_{index}-"
    for qkv in ["query", "key", "value"]:
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-kernel"] = reshape_kernel
      hooks[base + f"MultiHeadDotProductAttention_0-{qkv}-bias"] = vis_bias
    hooks[base + "MultiHeadDotProductAttention_0-out-kernel"] = reshape_kernel
    hooks[base + "MultiHeadDotProductAttention_0-out-bias"] = vis_bias
    for dense in ["Dense_0", "Dense_1"]:
      hooks[base + f"MlpBlockViT_0-{dense}-kernel"] = reshape_kernel
  return hooks
