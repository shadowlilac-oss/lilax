"""Parameter mappings and transformation hooks for checkpoint conversion."""

import warnings
import numpy as np
import jax
import jax.numpy as jnp


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


def LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns a mapping from MaxText to Hugging Face LLaMA 3.1 parameter names."""
  n_layers = config["num_hidden_layers"]
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
  }

  if scan_layers:
    mapping["params-decoder-layers-self_attention-query-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.q_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-key-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.k_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-value-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.v_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-self_attention-out-kernel"] = [
        f"model.layers.{layer_idx}.self_attn.o_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wi_0-kernel"] = [
        f"model.layers.{layer_idx}.mlp.gate_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wi_1-kernel"] = [
        f"model.layers.{layer_idx}.mlp.up_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-mlp-wo-kernel"] = [
        f"model.layers.{layer_idx}.mlp.down_proj.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"] = [
        f"model.layers.{layer_idx}.input_layernorm.weight" for layer_idx in range(n_layers)
    ]
    mapping["params-decoder-layers-post_self_attention_layer_norm-scale"] = [
        f"model.layers.{layer_idx}.post_attention_layernorm.weight" for layer_idx in range(n_layers)
    ]
  else:
    for layer_idx in range(n_layers):
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.q_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.k_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.v_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = (
          f"model.layers.{layer_idx}.self_attn.o_proj.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = f"model.layers.{layer_idx}.mlp.up_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = f"model.layers.{layer_idx}.mlp.down_proj.weight"
      mapping[f"params-decoder-layers_{layer_idx}-pre_self_attention_layer_norm-scale"] = (
          f"model.layers.{layer_idx}.input_layernorm.weight"
      )
      mapping[f"params-decoder-layers_{layer_idx}-post_self_attention_layer_norm-scale"] = (
          f"model.layers.{layer_idx}.post_attention_layernorm.weight"
      )

  return mapping


def LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for MaxText and Hugging Face."""
  nlayers = config["num_hidden_layers"]

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(config["head_dim"]))
      original_dtype = input_tensor.dtype
      output_tensor = input_tensor.astype(np.float32) * depth_scale
      return output_tensor.astype(original_dtype)
    else:
      depth_scale = np.dtype("float32").type(1 / np.sqrt(config["head_dim"]))
      original_dtype = input_tensor.dtype
      output_tensor = input_tensor.astype(np.float32) * depth_scale
      return output_tensor.astype(original_dtype)

  def adjust_rope(input_tensor, target_shape):
    arr = input_tensor
    if saving_to_hf:
      evens = arr[..., ::2]
      odds = arr[..., 1::2]
      return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)
    else:
      assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
      half_dim = arr.shape[-1] // 2
      first_half = arr[..., :half_dim]
      second_half = arr[..., half_dim:]
      return jax.numpy.stack([first_half, second_half], axis=-1).reshape(arr.shape)

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).transpose()
    else:
      return input_tensor.transpose().reshape(target_shape)

  query_hook_chain = [scale_query_layer, adjust_rope, reshape_kernel]
  key_hook_chain = [adjust_rope, reshape_kernel]
  if not saving_to_hf:
    query_hook_chain.reverse()
    key_hook_chain.reverse()

  hook_fns = {}
  hook_fns["params-decoder-logits_dense-kernel"] = reshape_kernel
  if scan_layers:
    hook_fns = {
        **hook_fns,
        "params-decoder-layers-self_attention-query-kernel": query_hook_chain,
        "params-decoder-layers-self_attention-key-kernel": key_hook_chain,
        "params-decoder-layers-self_attention-value-kernel": reshape_kernel,
        "params-decoder-layers-self_attention-out-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wi_0-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wi_1-kernel": reshape_kernel,
        "params-decoder-layers-mlp-wo-kernel": reshape_kernel,
    }
  else:
    for layer_idx in range(nlayers):
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = query_hook_chain
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = key_hook_chain
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = reshape_kernel
      hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = reshape_kernel
  return hook_fns


def LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN():
  """Returns LLaMA 3.1 weight transformations for vLLM."""
  def reorder_rope(arr):
    evens = arr[..., ::2]
    odds = arr[..., 1::2]
    return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

  def transform_query_kernel(arr):
    head_dim = arr.shape[-1]
    depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
    arr = arr * depth_scale
    return reorder_rope(arr)

  hook_fns = {
      "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
      "base.decoder.layers.self_attention.key.kernel": reorder_rope,
  }
  return hook_fns


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
    layers = [(f"params-decoder-layers_remainder-layers_{i}", num_blocks * 6 + i) for i in range(nlayers % 6)]
  else:
    layers = [(f"params-decoder-layers_{i}", i) for i in range(nlayers)]
  for prefix, i in layers:
    for mt, hf in specs:
      mapping[f"{prefix}-{mt}"] = hf_layer(i, hf)
    if not maxtext_config.share_kv_projections or i % 6 != 5:
      mapping[f"{prefix}-self_attention-value-kernel"] = hf_layer(i, "self_attn.v_proj.weight")
    elif scan_layers:
      mapping[f"{prefix}-self_attention-value-kernel"] = hf_layer(i, "self_attn.k_proj.weight")
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
    for rem_idx in range(nlayers % 6):
      real_layer_idx = 6 * (nlayers // 6) + rem_idx
      attach_hooks(f"params-decoder-layers_remainder-layers_{rem_idx}", real_layer_idx % 6 == 5)
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


def OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  n_layers = config["num_hidden_layers"]
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  if scan_layers:
    for cycle_idx in range(layer_cycle_interval):
      hf_indices = range(cycle_idx, n_layers, layer_cycle_interval)
      prefix = f"params-decoder-layers-layers_{cycle_idx}"
      mapping.update({
          f"{prefix}-attention-query-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices],
          f"{prefix}-attention-key-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices],
          f"{prefix}-attention-value-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices],
          f"{prefix}-attention-out-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices],
          f"{prefix}-attention-query_norm-scale": [f"model.layers.{i}.self_attn.q_norm.weight" for i in hf_indices],
          f"{prefix}-attention-key_norm-scale": [f"model.layers.{i}.self_attn.k_norm.weight" for i in hf_indices],
          f"{prefix}-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in hf_indices],
          f"{prefix}-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in hf_indices],
          f"{prefix}-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in hf_indices],
          f"{prefix}-post_self_attention_layer_norm-scale": [
              f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
          ],
          f"{prefix}-post_mlp_layer_norm-scale": [
              f"model.layers.{i}.post_feedforward_layernorm.weight" for i in hf_indices
          ],
      })
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"model.layers.{i}"
      mapping.update({
          f"{prefix}-attention-query-kernel": f"{hf_prefix}.self_attn.q_proj.weight",
          f"{prefix}-attention-key-kernel": f"{hf_prefix}.self_attn.k_proj.weight",
          f"{prefix}-attention-value-kernel": f"{hf_prefix}.self_attn.v_proj.weight",
          f"{prefix}-attention-out-kernel": f"{hf_prefix}.self_attn.o_proj.weight",
          f"{prefix}-attention-query_norm-scale": f"{hf_prefix}.self_attn.q_norm.weight",
          f"{prefix}-attention-key_norm-scale": f"{hf_prefix}.self_attn.k_norm.weight",
          f"{prefix}-mlp-wi_0-kernel": f"{hf_prefix}.mlp.gate_proj.weight",
          f"{prefix}-mlp-wi_1-kernel": f"{hf_prefix}.mlp.up_proj.weight",
          f"{prefix}-mlp-wo-kernel": f"{hf_prefix}.mlp.down_proj.weight",
          f"{prefix}-post_self_attention_layer_norm-scale": f"{hf_prefix}.post_attention_layernorm.weight",
          f"{prefix}-post_mlp_layer_norm-scale": f"{hf_prefix}.post_feedforward_layernorm.weight",
      })
  return mapping


def OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def scale_rmsnorm_layer(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def adapt_olmo3_qk_norm(input_tensor, target_shape):
    return input_tensor.reshape(target_shape)

  def pad_hf_embedding_layer(input_tensor, target_shape):
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]
    if source_vocab_size == target_vocab_size:
      return input_tensor
    if saving_to_hf:
      return input_tensor[:target_vocab_size, :]
    padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    padded_tensor[:source_vocab_size, :] = input_tensor
    return padded_tensor

  hooks = {
      "params-token_embedder-embedding": pad_hf_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
      "params-decoder-decoder_norm-scale": scale_rmsnorm_layer,
  }
  kernel_keys = [
      "attention-query-kernel", "attention-key-kernel", "attention-value-kernel", "attention-out-kernel",
      "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
  ]
  norm_keys = [
      "post_self_attention_layer_norm-scale", "post_mlp_layer_norm-scale",
      "attention-query_norm-scale", "attention-key_norm-scale",
  ]
  cycle_len = getattr(maxtext_config, "inhomogeneous_layer_cycle_interval", 4)
  n_layers = config["num_hidden_layers"]
  if scan_layers:
    prefixes = (f"params-decoder-layers-layers_{i}" for i in range(cycle_len))
  else:
    prefixes = (f"params-decoder-layers_{i}" for i in range(n_layers))
  for prefix in prefixes:
    for key in kernel_keys:
      hooks[f"{prefix}-{key}"] = reshape_kernel
    for key in norm_keys:
      if "attention-" in key and "_norm-scale" in key:
        hooks[f"{prefix}-{key}"] = adapt_olmo3_qk_norm
      else:
        hooks[f"{prefix}-{key}"] = scale_rmsnorm_layer
  return hooks


def QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", config.get("num_local_experts", 0))
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  if scan_layers:
    mapping.update({
        "params-decoder-layers-pre_self_attention_layer_norm-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query-bias": [
            f"model.layers.{i}.self_attn.q_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key-bias": [
            f"model.layers.{i}.self_attn.k_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-value-bias": [
            f"model.layers.{i}.self_attn.v_proj.bias" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-query_norm-scale": [
            f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-self_attention-key_norm-scale": [
            f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)
        ],
        "params-decoder-layers-post_self_attention_layer_norm-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
        ],
    })
    if num_experts > 1:
      mapping.update({
          "params-decoder-layers-moe_block-gate-kernel": [
              f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)
          ],
          "params-decoder-layers-moe_block-wi_0": [
              [f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
          "params-decoder-layers-moe_block-wi_1": [
              [f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
          "params-decoder-layers-moe_block-wo": [
              [f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight" for layer in range(n_layers)]
              for expert in range(num_experts)
          ],
      })
    else:
      mapping.update({
          "params-decoder-layers-mlp-wi_0-kernel": [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)],
          "params-decoder-layers-mlp-wi_1-kernel": [f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)],
          "params-decoder-layers-mlp-wo-kernel": [f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)],
      })
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers-layers_{i}" if getattr(maxtext_config, "use_m3_model", False) else f"params-decoder-layers_{i}"
      mapping.update({
          f"{prefix}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
          f"{prefix}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
          f"{prefix}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
          f"{prefix}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
          f"{prefix}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
          f"{prefix}-self_attention-query-bias": f"model.layers.{i}.self_attn.q_proj.bias",
          f"{prefix}-self_attention-key-bias": f"model.layers.{i}.self_attn.k_proj.bias",
          f"{prefix}-self_attention-value-bias": f"model.layers.{i}.self_attn.v_proj.bias",
          f"{prefix}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
          f"{prefix}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
          f"{prefix}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
      })
      if num_experts > 1:
        mapping.update({
            f"{prefix}-moe_block-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
            f"{prefix}-moe_block-wi_0": [
                f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)
            ],
            f"{prefix}-moe_block-wi_1": [
                f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)
            ],
            f"{prefix}-moe_block-wo": [
                f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)
            ],
        })
      else:
        mapping.update({
            f"{prefix}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
            f"{prefix}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
            f"{prefix}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
        })
  return mapping


def QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  n_layers = config["num_hidden_layers"]
  num_experts = config.get("num_experts", config.get("num_local_experts", 0))

  def pad_embedding_layer(input_tensor, target_shape):
    source_vocab_size = input_tensor.shape[0]
    target_vocab_size = target_shape[0]
    if source_vocab_size == target_vocab_size:
      return input_tensor
    if saving_to_hf:
      return input_tensor[:target_vocab_size, :]
    padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
    padded_tensor[:source_vocab_size, :] = input_tensor
    return padded_tensor

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      return input_tensor.reshape(np.flip(np.array(target_shape))).T
    return input_tensor.T.reshape(target_shape)

  def reshape_expert_kernel(input_tensor, target_shape=None):
    if input_tensor.ndim == 3:
      return input_tensor.transpose(0, 2, 1)
    return input_tensor.transpose(1, 0)

  def reshape_bias(input_tensor, target_shape=None):
    return input_tensor.reshape(target_shape)

  mapping = {
      "params-token_embedder-embedding": pad_embedding_layer,
      "params-decoder-logits_dense-kernel": reshape_kernel,
  }
  kernel_hooks = [
      "self_attention-query-kernel", "self_attention-key-kernel", "self_attention-value-kernel",
      "self_attention-out-kernel", "mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel",
  ]
  bias_hooks = ["self_attention-query-bias", "self_attention-key-bias", "self_attention-value-bias"]
  moe_gate_hooks = ["moe_block-gate-kernel", "moe_block-wi_0-kernel", "moe_block-wi_1-kernel", "moe_block-wo-kernel"]
  moe_expert_hooks = ["moe_block-wi_0", "moe_block-wi_1", "moe_block-wo"]
  if scan_layers:
    for key in kernel_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_kernel
    for key in bias_hooks:
      mapping[f"params-decoder-layers-{key}"] = reshape_bias
    if num_experts > 1:
      for key in moe_gate_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_kernel
      for key in moe_expert_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_expert_kernel
  else:
    for i in range(n_layers):
      prefix = f"params-decoder-layers-layers_{i}" if getattr(maxtext_config, "use_m3_model", False) else f"params-decoder-layers_{i}"
      for key in kernel_hooks:
        mapping[f"{prefix}-{key}"] = reshape_kernel
      for key in bias_hooks:
        mapping[f"{prefix}-{key}"] = reshape_bias
      if num_experts > 1:
        for key in moe_gate_hooks:
          mapping[f"{prefix}-{key}"] = reshape_kernel
        for key in moe_expert_hooks:
          mapping[f"{prefix}-{key}"] = reshape_expert_kernel
  return mapping


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


def _qwen3_8_wrap_config(config):
  if "text_config" in config:
    return config
  return {"text_config": config}


def QWEN3_8_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  return QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(
      _qwen3_8_wrap_config(config), maxtext_config, scan_layers=scan_layers, hf_prefix="model")


def QWEN3_8_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  return QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      _qwen3_8_wrap_config(config), maxtext_config, scan_layers, saving_to_hf)


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mapping from MaxText to HuggingFace Qwen3-Next weight paths."""
  num_main_layers = config["num_hidden_layers"]
  num_experts = config["num_experts"]
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  shared_specs = [
      ("input_layernorm-scale", "input_layernorm.weight"),
      ("post_attention_layernorm-scale", "post_attention_layernorm.weight"),
      ("mlp-routed_experts-gate-kernel", "mlp.gate.weight"),
      ("mlp-shared_expert-wi_0-kernel", "mlp.shared_expert.gate_proj.weight"),
      ("mlp-shared_expert-wi_1-kernel", "mlp.shared_expert.up_proj.weight"),
      ("mlp-shared_expert-wo-kernel", "mlp.shared_expert.down_proj.weight"),
      ("mlp-shared_expert_gate-kernel", "mlp.shared_expert_gate.weight"),
  ]
  local_specs = shared_specs + [
      ("attention-in_proj_qkvz-kernel", "linear_attn.in_proj_qkvz.weight"),
      ("attention-in_proj_ba-kernel", "linear_attn.in_proj_ba.weight"),
      ("attention-conv1d-kernel", "linear_attn.conv1d.weight"),
      ("attention-A_log", "linear_attn.A_log"),
      ("attention-dt_bias", "linear_attn.dt_bias"),
      ("attention-norm-rms_norm-scale", "linear_attn.norm.weight"),
      ("attention-out_proj-kernel", "linear_attn.out_proj.weight"),
  ]
  global_specs = shared_specs + [
      ("attention-attention-query-kernel", "self_attn.q_proj.weight"),
      ("attention-attention-key-kernel", "self_attn.k_proj.weight"),
      ("attention-attention-value-kernel", "self_attn.v_proj.weight"),
      ("attention-attention-out-kernel", "self_attn.o_proj.weight"),
      ("attention-attention-query_norm-scale", "self_attn.q_norm.weight"),
      ("attention-attention-key_norm-scale", "self_attn.k_norm.weight"),
  ]
  expert_specs = [
      ("mlp-routed_experts-wi_0", "gate_proj.weight"),
      ("mlp-routed_experts-wi_1", "up_proj.weight"),
      ("mlp-routed_experts-wo", "down_proj.weight"),
  ]
  if scan_layers:
    num_blocks = num_main_layers // layer_cycle_interval
    local_positions = list(range(layer_cycle_interval - 1))
    global_position = layer_cycle_interval - 1

    def hf_layer(block_idx, position, suffix):
      return f"model.layers.{block_idx * layer_cycle_interval + position}.{suffix}"

    local_prefix = "params-decoder-layers-local_layers"
    for subkey, suffix in local_specs:
      mapping[f"{local_prefix}-{subkey}"] = [
          [hf_layer(b, p, suffix) for p in local_positions] for b in range(num_blocks)
      ]
    for subkey, suffix in expert_specs:
      mapping[f"{local_prefix}-{subkey}"] = [
          [[hf_layer(b, p, f"mlp.experts.{e}.{suffix}") for p in local_positions] for b in range(num_blocks)]
          for e in range(num_experts)
      ]
    global_prefix = "params-decoder-layers-global_layer"
    for subkey, suffix in global_specs:
      mapping[f"{global_prefix}-{subkey}"] = [hf_layer(b, global_position, suffix) for b in range(num_blocks)]
    for subkey, suffix in expert_specs:
      mapping[f"{global_prefix}-{subkey}"] = [
          [hf_layer(b, global_position, f"mlp.experts.{e}.{suffix}") for b in range(num_blocks)]
          for e in range(num_experts)
      ]
    remainder_positions = range(num_main_layers - num_blocks * layer_cycle_interval)
    if remainder_positions:
      remainder_prefix = "params-decoder-layers_remainder-local_layers"
      for subkey, suffix in local_specs:
        mapping[f"{remainder_prefix}-{subkey}"] = [hf_layer(num_blocks, p, suffix) for p in remainder_positions]
      for subkey, suffix in expert_specs:
        mapping[f"{remainder_prefix}-{subkey}"] = [
            [hf_layer(num_blocks, p, f"mlp.experts.{e}.{suffix}") for p in remainder_positions]
            for e in range(num_experts)
        ]
  else:
    for i in range(num_main_layers):
      prefix = f"params-decoder-layers_{i}"
      is_full_attention_layer = (i % layer_cycle_interval + 1) % layer_cycle_interval == 0
      for subkey, suffix in global_specs if is_full_attention_layer else local_specs:
        mapping[f"{prefix}-{subkey}"] = f"model.layers.{i}.{suffix}"
      for subkey, suffix in expert_specs:
        mapping[f"{prefix}-{subkey}"] = [f"model.layers.{i}.mlp.experts.{e}.{suffix}" for e in range(num_experts)]
  return mapping


def QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3-Next."""
  def transpose(input_tensor, target_shape=None):
    return input_tensor.T

  def reshape_kernel(input_tensor, target_shape):
    if saving_to_hf:
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T
    return input_tensor.T.reshape(target_shape)

  def permute_conv(input_tensor, target_shape=None):
    return input_tensor.transpose(2, 1, 0)

  hooks = {"params-decoder-logits_dense-kernel": transpose}
  layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval
  num_main_layers = config["num_hidden_layers"]
  if scan_layers:
    layer_prefixes = [
        ("params-decoder-layers-local_layers", False),
        ("params-decoder-layers-global_layer", True),
    ]
    if num_main_layers % layer_cycle_interval:
      layer_prefixes.append(("params-decoder-layers_remainder-local_layers", False))
  else:
    layer_prefixes = [
        (f"params-decoder-layers_{i}", (i % layer_cycle_interval + 1) % layer_cycle_interval == 0)
        for i in range(num_main_layers)
    ]
  for prefix, is_full_attention_layer in layer_prefixes:
    if is_full_attention_layer:
      for key in ["query", "key", "value", "out"]:
        hooks[f"{prefix}-attention-attention-{key}-kernel"] = reshape_kernel
    else:
      hooks[f"{prefix}-attention-in_proj_qkvz-kernel"] = transpose
      hooks[f"{prefix}-attention-in_proj_ba-kernel"] = transpose
      hooks[f"{prefix}-attention-out_proj-kernel"] = transpose
      hooks[f"{prefix}-attention-conv1d-kernel"] = permute_conv
    mlp_prefix = f"{prefix}-mlp"
    hooks[f"{mlp_prefix}-routed_experts-gate-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_0-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wi_1-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert-wo-kernel"] = transpose
    hooks[f"{mlp_prefix}-shared_expert_gate-kernel"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wi_0"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wi_1"] = transpose
    hooks[f"{mlp_prefix}-routed_experts-wo"] = transpose
  return hooks


def WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  """Returns mapping from MaxText to HuggingFace Weaver weight paths."""
  qwen3_vl_mapping = QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers)
  mapping = {}

  def translate_hf_key(val):
    if isinstance(val, list):
      return [translate_hf_key(v) for v in val]
    elif isinstance(val, str):
      if val.startswith("model.language_model."):
        val = val[len("model.language_model.") :]
      elif val.startswith("model.visual."):
        val = val[len("model.visual.") :]
      elif val.startswith("model."):
        val = val[len("model.") :]
      val = val.replace("self_attn.q_proj", "self_attn.to_q")
      val = val.replace("self_attn.k_proj", "self_attn.to_k")
      val = val.replace("self_attn.v_proj", "self_attn.to_v")
      val = val.replace("self_attn.o_proj", "self_attn.to_out")
      val = val.replace("self_attn.q_norm", "self_attn.norm_q")
      val = val.replace("self_attn.k_norm", "self_attn.norm_k")
      return val
    return val

  for key, value in qwen3_vl_mapping.items():
    mapping[key] = translate_hf_key(value)
  return mapping


def WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Weaver."""
  return QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers, saving_to_hf)


def QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN(target_shape=None):
  """Creates parameter transformation functions for Qwen3."""
  return {}


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


def MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  num_experts = maxtext_config.num_experts
  if scan_layers:
    mapping.update({
        "params-decoder-layers-self_attention-query-kernel": [],
        "params-decoder-layers-self_attention-key-kernel": [],
        "params-decoder-layers-self_attention-value-kernel": [],
        "params-decoder-layers-self_attention-out-kernel": [],
        "params-decoder-layers-pre_self_attention_layer_norm-scale": [],
        "params-decoder-layers-post_self_attention_layer_norm-scale": [],
        "params-decoder-layers-MoeBlock_0-gate-kernel": [],
        "params-decoder-layers-MoeBlock_0-wi_0": [],
        "params-decoder-layers-MoeBlock_0-wi_1": [],
        "params-decoder-layers-MoeBlock_0-wo": [],
    })
    for i in range(config["num_hidden_layers"]):
      hf_prefix = f"model.layers.{i}"
      mapping["params-decoder-layers-self_attention-query-kernel"].append(f"{hf_prefix}.self_attn.q_proj.weight")
      mapping["params-decoder-layers-self_attention-key-kernel"].append(f"{hf_prefix}.self_attn.k_proj.weight")
      mapping["params-decoder-layers-self_attention-value-kernel"].append(f"{hf_prefix}.self_attn.v_proj.weight")
      mapping["params-decoder-layers-self_attention-out-kernel"].append(f"{hf_prefix}.self_attn.o_proj.weight")
      mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"].append(f"{hf_prefix}.input_layernorm.weight")
      mapping["params-decoder-layers-post_self_attention_layer_norm-scale"].append(
          f"{hf_prefix}.post_attention_layernorm.weight")
      mapping["params-decoder-layers-MoeBlock_0-gate-kernel"].append(f"{hf_prefix}.block_sparse_moe.gate.weight")
    for j in range(num_experts):
      w1_layers, w3_layers, w2_layers = [], [], []
      for i in range(config["num_hidden_layers"]):
        hf_prefix = f"model.layers.{i}"
        w1_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight")
        w3_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight")
        w2_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight")
      mapping["params-decoder-layers-MoeBlock_0-wi_0"].append(w1_layers)
      mapping["params-decoder-layers-MoeBlock_0-wi_1"].append(w3_layers)
      mapping["params-decoder-layers-MoeBlock_0-wo"].append(w2_layers)
  else:
    for i in range(config["num_hidden_layers"]):
      maxtext_prefix = f"params-decoder-layers_{i}"
      hf_prefix = f"model.layers.{i}"
      mapping[f"{maxtext_prefix}-self_attention-query-kernel"] = f"{hf_prefix}.self_attn.q_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-key-kernel"] = f"{hf_prefix}.self_attn.k_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-value-kernel"] = f"{hf_prefix}.self_attn.v_proj.weight"
      mapping[f"{maxtext_prefix}-self_attention-out-kernel"] = f"{hf_prefix}.self_attn.o_proj.weight"
      mapping[f"{maxtext_prefix}-pre_self_attention_layer_norm-scale"] = f"{hf_prefix}.input_layernorm.weight"
      mapping[f"{maxtext_prefix}-post_self_attention_layer_norm-scale"] = f"{hf_prefix}.post_attention_layernorm.weight"
      mapping[f"{maxtext_prefix}-MoeBlock_0-gate-kernel"] = f"{hf_prefix}.block_sparse_moe.gate.weight"
      mapping[f"{maxtext_prefix}-MoeBlock_0-wi_0"] = [
          f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight" for j in range(num_experts)]
      mapping[f"{maxtext_prefix}-MoeBlock_0-wi_1"] = [
          f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight" for j in range(num_experts)]
      mapping[f"{maxtext_prefix}-MoeBlock_0-wo"] = [
          f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight" for j in range(num_experts)]
  return mapping


def MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  hooks = {}

  def reshape_and_transpose_attention(x, target_shape):
    if saving_to_hf:
      return x.reshape(config["hidden_size"], -1).transpose()
    return x.transpose().reshape(target_shape)

  def reshape_kernel(x, target_shape):
    return x.transpose()

  def scale_query_layer(input_tensor, target_shape):
    if saving_to_hf:
      depth_scale = np.dtype("float32").type(np.sqrt(maxtext_config.head_dim))
      return (input_tensor * depth_scale).astype(input_tensor.dtype)
    depth_scale = np.dtype("float32").type(1 / np.sqrt(maxtext_config.head_dim))
    return (input_tensor * depth_scale).astype(input_tensor.dtype)

  query_hook_chain = [reshape_and_transpose_attention, scale_query_layer]
  if scan_layers:
    plan = [
        ("params-decoder-layers-self_attention-query-kernel", query_hook_chain),
        ("params-decoder-layers-self_attention-key-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-self_attention-value-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-self_attention-out-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers-MoeBlock_0-wi_0", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-wi_1", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-wo", reshape_kernel),
        ("params-decoder-layers-MoeBlock_0-gate-kernel", reshape_kernel),
    ]
  else:
    plan = [
        ("params-decoder-layers_{i}-self_attention-query-kernel", query_hook_chain),
        ("params-decoder-layers_{i}-self_attention-key-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-self_attention-value-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-self_attention-out-kernel", reshape_and_transpose_attention),
        ("params-decoder-layers_{i}-MoeBlock_0-wi_0", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-wi_1", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-wo", reshape_kernel),
        ("params-decoder-layers_{i}-MoeBlock_0-gate-kernel", reshape_kernel),
    ]
  plan.append(("params-decoder-logits_dense-kernel", reshape_kernel))
  for maxtext_pattern, op_func in plan:
    if "{i}" in maxtext_pattern:
      for i in range(config["num_hidden_layers"]):
        hooks[maxtext_pattern.format(i=i)] = op_func
    else:
      hooks[maxtext_pattern] = op_func
  return hooks


def DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  mt_layers = getattr(maxtext_config, "num_decoder_layers", None)
  n_layers = mt_layers if isinstance(mt_layers, int) else config["num_hidden_layers"]
  mt_dense = getattr(maxtext_config, "first_num_dense_layers", None)
  n_dense = mt_dense if isinstance(mt_dense, int) else config["first_k_dense_replace"]
  mt_experts = getattr(maxtext_config, "num_experts", None)
  n_experts = mt_experts if isinstance(mt_experts, int) else config.get("n_routed_experts", 0)
  mapping = {
      "params-token_embedder-embedding": "model.embed_tokens.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  attention = {
      "pre_self_attention_layer_norm-scale": "input_layernorm.weight",
      "post_self_attention_layer_norm-scale": "post_attention_layernorm.weight",
      "self_attention-kv_norm-scale": "self_attn.kv_a_layernorm.weight",
      "self_attention-wkv_a-kernel": "self_attn.kv_a_proj_with_mqa.weight",
      "self_attention-wkv_b-kernel": "self_attn.kv_b_proj.weight",
      "self_attention-out-kernel": "self_attn.o_proj.weight",
      "self_attention-query-kernel": "self_attn.q_proj.weight",
      "self_attention-q_norm-scale": "self_attn.q_a_layernorm.weight",
      "self_attention-wq_a-kernel": "self_attn.q_a_proj.weight",
      "self_attention-wq_b-kernel": "self_attn.q_b_proj.weight",
      "self_attention-indexer-k_norm-bias": "self_attn.indexer.k_norm.bias",
      "self_attention-indexer-k_norm-scale": "self_attn.indexer.k_norm.weight",
      "self_attention-indexer-weights_proj-kernel": "self_attn.indexer.weights_proj.weight",
      "self_attention-indexer-wk-kernel": "self_attn.indexer.wk.weight",
      "self_attention-indexer-wq_b-kernel": "self_attn.indexer.wq_b.weight",
  }
  dense = attention | {
      "mlp-wi_0-kernel": "mlp.gate_proj.weight", "mlp-wi_1-kernel": "mlp.up_proj.weight",
      "mlp-wo-kernel": "mlp.down_proj.weight",
  }
  moe = attention | {
      "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel": "mlp.shared_experts.gate_proj.weight",
      "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel": "mlp.shared_experts.up_proj.weight",
      "DeepSeekMoeBlock_0-shared_experts-wo-kernel": "mlp.shared_experts.down_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel": "mlp.gate.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias": "mlp.gate.e_score_correction_bias",
  }
  experts = {
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_0": "gate_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_1": "up_proj.weight",
      "DeepSeekMoeBlock_0-MoeBlock_0-wo": "down_proj.weight",
  }
  bias = "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias"
  if scan_layers:
    for key, hf in dense.items():
      mapping[f"params-decoder-dense_layers-{key}"] = [f"model.layers.{i}.{hf}" for i in range(n_dense)]
    for key, hf in moe.items():
      mapping[f"params-decoder-moe_layers-{key}"] = [f"model.layers.{i}.{hf}" for i in range(n_dense, n_layers)]
    mapping[f"MoEBiasVar-decoder-moe_layers-{bias}"] = [
        f"model.layers.{i}.mlp.gate.e_score_correction_bias" for i in range(n_dense, n_layers)]
    for key, hf in experts.items():
      mapping[f"params-decoder-moe_layers-{key}"] = [
          [f"model.layers.{i}.mlp.experts.{e}.{hf}" for i in range(n_dense, n_layers)] for e in range(n_experts)]
  else:
    for i in range(n_dense):
      for key, hf in dense.items():
        mapping[f"params-decoder-dense_layers_{i}-{key}"] = f"model.layers.{i}.{hf}"
    for i in range(n_dense, n_layers):
      prefix = f"params-decoder-moe_layers_{i - n_dense}"
      for key, hf in moe.items():
        mapping[f"{prefix}-{key}"] = f"model.layers.{i}.{hf}"
      mapping[f"MoEBiasVar-decoder-moe_layers_{i - n_dense}-{bias}"] = (
          f"model.layers.{i}.mlp.gate.e_score_correction_bias")
      for key, hf in experts.items():
        mapping[f"{prefix}-{key}"] = [f"model.layers.{i}.mlp.experts.{e}.{hf}" for e in range(n_experts)]
  mtp_layers = getattr(maxtext_config, "mtp_num_layers", 0)
  if not isinstance(mtp_layers, int):
    mtp_layers = 0
  for j in range(mtp_layers):
    i = config["num_hidden_layers"] + j
    prefix = f"params-mtp_block-mtp_layer_{j + 1}"
    layer = f"{prefix}-mtp_{j + 1}_transformer_layer"
    mapping[f"{prefix}-mtp_{j + 1}_embedding_norm-scale"] = f"model.layers.{i}.enorm.weight"
    mapping[f"{prefix}-mtp_{j + 1}_hidden_state_norm-scale"] = f"model.layers.{i}.hnorm.weight"
    mapping[f"{prefix}-mtp_{j + 1}_projection-kernel"] = f"model.layers.{i}.eh_proj.weight"
    mapping[f"{prefix}-mtp_{j + 1}_final_norm-scale"] = f"model.layers.{i}.shared_head.norm.weight"
    for key, hf in moe.items():
      mapping[f"{layer}-{key}"] = f"model.layers.{i}.{hf}"
    mapping[f"MoEBiasVar-mtp_block-mtp_layer_{j + 1}-mtp_{j + 1}_transformer_layer-{bias}"] = (
        f"model.layers.{i}.mlp.gate.e_score_correction_bias")
    for key, hf in experts.items():
      mapping[f"{layer}-{key}"] = [f"model.layers.{i}.mlp.experts.{e}.{hf}" for e in range(n_experts)]
  return mapping


def DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def reshape_kernel(x, target_shape):
    if saving_to_hf:
      return x.reshape(np.flip(np.array(target_shape))).T
    return x.T.reshape(target_shape)

  mt_layers = getattr(maxtext_config, "num_decoder_layers", None)
  n_layers = mt_layers if isinstance(mt_layers, int) else config["num_hidden_layers"]
  mt_dense = getattr(maxtext_config, "first_num_dense_layers", None)
  n_dense = mt_dense if isinstance(mt_dense, int) else config["first_k_dense_replace"]
  attention = {
      "self_attention-wkv_a-kernel", "self_attention-wkv_b-kernel", "self_attention-out-kernel",
      "self_attention-query-kernel", "self_attention-wq_a-kernel", "self_attention-wq_b-kernel",
      "self_attention-indexer-weights_proj-kernel", "self_attention-indexer-wk-kernel",
      "self_attention-indexer-wq_b-kernel",
  }
  dense = attention | {"mlp-wi_0-kernel", "mlp-wi_1-kernel", "mlp-wo-kernel"}
  moe = attention | {
      "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel", "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
      "DeepSeekMoeBlock_0-shared_experts-wo-kernel", "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",
      "DeepSeekMoeBlock_0-MoeBlock_0-wi_0", "DeepSeekMoeBlock_0-MoeBlock_0-wi_1",
      "DeepSeekMoeBlock_0-MoeBlock_0-wo",
  }
  hooks = {"params-decoder-logits_dense-kernel": reshape_kernel}
  if scan_layers:
    for key in dense:
      hooks[f"params-decoder-dense_layers-{key}"] = reshape_kernel
    for key in moe:
      hooks[f"params-decoder-moe_layers-{key}"] = reshape_kernel
  else:
    for i in range(n_dense):
      for key in dense:
        hooks[f"params-decoder-dense_layers_{i}-{key}"] = reshape_kernel
    for i in range(n_layers - n_dense):
      for key in moe:
        hooks[f"params-decoder-moe_layers_{i}-{key}"] = reshape_kernel
  mtp_layers = getattr(maxtext_config, "mtp_num_layers", 0)
  if not isinstance(mtp_layers, int):
    mtp_layers = 0
  for j in range(mtp_layers):
    prefix = f"params-mtp_block-mtp_layer_{j + 1}"
    hooks[f"{prefix}-mtp_{j + 1}_projection-kernel"] = reshape_kernel
    for key in moe:
      hooks[f"{prefix}-mtp_{j + 1}_transformer_layer-{key}"] = reshape_kernel
  return hooks


def DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN():
  return {}


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


def MING_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  num_layers = maxtext_config.num_decoder_layers
  num_experts = maxtext_config.num_experts
  cycle = maxtext_config.inhomogeneous_layer_cycle_interval
  first_dense = maxtext_config.first_num_dense_layers
  mapping = {
      "params-token_embedder-embedding": "model.word_embeddings.weight",
      "params-decoder-decoder_norm-scale": "model.norm.weight",
      "params-decoder-logits_dense-kernel": "lm_head.weight",
  }
  kda_keys = {f"kda-{n}-kernel": f"attention.{n}.weight" for n in (
      "q_proj", "k_proj", "v_proj", "f_proj", "b_proj", "g_proj", "o_proj", "q_conv1d", "k_conv1d", "v_conv1d"
  )}
  kda_keys |= {"kda-A_log": "attention.A_log", "kda-dt_bias": "attention.dt_bias",
               "kda-o_norm-scale": "attention.o_norm.weight"}
  mla_keys = {
      "self_attention-wq_a-kernel": "attention.q_a_proj.weight",
      "self_attention-q_norm-scale": "attention.q_a_layernorm.weight",
      "self_attention-wq_b-kernel": "attention.q_b_proj.weight",
      "self_attention-wkv_a-kernel": "attention.kv_a_proj_with_mqa.weight",
      "self_attention-kv_norm-scale": "attention.kv_a_layernorm.weight",
      "self_attention-wkv_b-kernel": "attention.kv_b_proj.weight",
      "self_attention-out-kernel": "attention.dense.weight",
      "self_attention-g_proj-kernel": "attention.g_proj.weight",
  }
  dense_keys = {"mlp-wi_0-kernel": "mlp.gate_proj.weight", "mlp-wi_1-kernel": "mlp.up_proj.weight",
                "mlp-wo-kernel": "mlp.down_proj.weight"}
  moe_keys = {
      "moe_block-MoeBlock_0-gate-kernel": "mlp.gate.weight",
      "moe_block-shared_experts-wi_0-kernel": "mlp.shared_experts.gate_proj.weight",
      "moe_block-shared_experts-wi_1-kernel": "mlp.shared_experts.up_proj.weight",
      "moe_block-shared_experts-wo-kernel": "mlp.shared_experts.down_proj.weight",
  }
  expert_keys = {"moe_block-MoeBlock_0-wi_0": "gate_proj.weight",
                 "moe_block-MoeBlock_0-wi_1": "up_proj.weight", "moe_block-MoeBlock_0-wo": "down_proj.weight"}
  layer_types = [((i + 1) % cycle == 0 or i >= num_layers // cycle * cycle,
                  num_experts > 1 and i >= first_dense) for i in range(num_layers)]
  unrolled_layers = range(num_layers)
  if scan_layers:
    num_prefix = min(num_layers, -(-first_dense // cycle) * cycle)
    num_blocks = (num_layers - num_prefix) // cycle
    remainder_start = num_prefix + num_blocks * cycle
    unrolled_layers = list(range(num_prefix)) + list(range(remainder_start, num_layers))
    for j in range(cycle):
      hf_layers = [num_prefix + b * cycle + j for b in range(num_blocks)]
      if not hf_layers:
        continue
      is_mla, is_moe = layer_types[hf_layers[0]]
      assert is_moe and all(layer_types[i] == (is_mla, is_moe) for i in hf_layers)
      prefix = f"params-decoder-scanned_blocks-layers_{j}"
      layer_keys = {"pre_self_attention_layer_norm-scale": "input_layernorm.weight",
                    "post_self_attention_layer_norm-scale": "post_attention_layernorm.weight"}
      layer_keys |= mla_keys if is_mla else kda_keys
      layer_keys |= moe_keys
      for mt_key, hf_key in layer_keys.items():
        mapping[f"{prefix}-{mt_key}"] = [f"model.layers.{i}.{hf_key}" for i in hf_layers]
      mapping[f"MoEBiasVar-decoder-scanned_blocks-layers_{j}-moe_block-MoeBlock_0-gate-bias"] = [
          f"model.layers.{i}.mlp.gate.expert_bias" for i in hf_layers
      ]
      for mt_key, hf_key in expert_keys.items():
        mapping[f"{prefix}-{mt_key}"] = [
            [f"model.layers.{i}.mlp.experts.{e}.{hf_key}" for i in hf_layers] for e in range(num_experts)
        ]
  for i in unrolled_layers:
    is_mla, is_moe = layer_types[i]
    prefix = f"params-decoder-layers_{i}"
    hf = f"model.layers.{i}"
    mapping[f"{prefix}-pre_self_attention_layer_norm-scale"] = f"{hf}.input_layernorm.weight"
    mapping[f"{prefix}-post_self_attention_layer_norm-scale"] = f"{hf}.post_attention_layernorm.weight"
    for mt_key, hf_key in (mla_keys if is_mla else kda_keys).items():
      mapping[f"{prefix}-{mt_key}"] = f"{hf}.{hf_key}"
    if is_moe:
      for mt_key, hf_key in moe_keys.items():
        mapping[f"{prefix}-{mt_key}"] = f"{hf}.{hf_key}"
      mapping[f"MoEBiasVar-decoder-layers_{i}-moe_block-MoeBlock_0-gate-bias"] = f"{hf}.mlp.gate.expert_bias"
      for mt_key, hf_key in expert_keys.items():
        mapping[f"{prefix}-{mt_key}"] = [f"{hf}.mlp.experts.{e}.{hf_key}" for e in range(num_experts)]
    else:
      for mt_key, hf_key in dense_keys.items():
        mapping[f"{prefix}-{mt_key}"] = f"{hf}.{hf_key}"
  if getattr(maxtext_config, "use_audio", False):
    enc = "params-audio_encoder-MingWhisperEncoder_0"
    for n in ("conv1", "conv2"):
      mapping[f"{enc}-{n}-kernel"] = f"model.audio.{n}.weight"
      mapping[f"{enc}-{n}-bias"] = f"model.audio.{n}.bias"
    mapping[f"{enc}-positional_embedding"] = "model.audio.positional_embedding"
    mapping[f"{enc}-ln_post-scale"] = "model.audio.ln_post.weight"
    mapping[f"{enc}-ln_post-bias"] = "model.audio.ln_post.bias"
    for j in range(maxtext_config.encoder_layers_for_audio):
      blk, hf = f"{enc}-blocks_{j}", f"model.audio.blocks.{j}"
      for ln in ("attn_ln", "mlp_ln"):
        mapping[f"{blk}-{ln}-scale"] = f"{hf}.{ln}.weight"
        mapping[f"{blk}-{ln}-bias"] = f"{hf}.{ln}.bias"
      for n in ("query", "key", "value", "out"):
        mapping[f"{blk}-attn-{n}-kernel"] = f"{hf}.attn.{n}.weight"
        if n != "key":
          mapping[f"{blk}-attn-{n}-bias"] = f"{hf}.attn.{n}.bias"
      for mt_n, hf_idx in (("mlp_fc1", 0), ("mlp_fc2", 2)):
        mapping[f"{blk}-{mt_n}-kernel"] = f"{hf}.mlp.{hf_idx}.weight"
        mapping[f"{blk}-{mt_n}-bias"] = f"{hf}.mlp.{hf_idx}.bias"
    proj = "params-audio_encoder-MingAudioProjector_0"
    for k in range(maxtext_config.num_conv_layers_for_audio):
      mapping[f"{proj}-conv_{k}-kernel"] = f"linear_proj_audio.{2 * k}.weight"
      mapping[f"{proj}-conv_{k}-bias"] = f"linear_proj_audio.{2 * k}.bias"
    linear_idx = 2 * maxtext_config.num_conv_layers_for_audio + 1
    mapping[f"{proj}-linear-kernel"] = f"linear_proj_audio.{linear_idx}.weight"
    mapping[f"{proj}-linear-bias"] = f"linear_proj_audio.{linear_idx}.bias"
  return mapping


def MING_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  def linear(x, target_shape):
    if saving_to_hf:
      return x.reshape(np.flip(np.array(target_shape))).T
    return x.T.reshape(target_shape)

  def conv1d(x, target_shape):
    del target_shape
    return np.transpose(x, (2, 1, 0))

  mapping = MING_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers)
  hooks = {}
  for mt_key in mapping:
    if (mt_key.endswith("conv1d-kernel") or "-conv1-kernel" in mt_key or "-conv2-kernel" in mt_key
        or ("MingAudioProjector_0-conv_" in mt_key and mt_key.endswith("-kernel"))):
      hooks[mt_key] = conv1d
    elif mt_key.endswith("-kernel") or mt_key.endswith(("MoeBlock_0-wi_0", "MoeBlock_0-wi_1", "MoeBlock_0-wo")):
      hooks[mt_key] = linear
  return hooks


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


PARAM_MAPPING = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-e2b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gemma4-e4b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "ming-3.0-audio-tiny": MING_MAXTEXT_TO_HF_PARAM_MAPPING,
    "hy4-preview": HY4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek4-284b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek4-1600b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-2b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-4b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-vl-30b-a3b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-20b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "gpt-oss-120b": GPT_OSS_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-397b-a17b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-a3b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-35b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.5-9b": QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3.8-2.4t-a95b": QWEN3_8_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_MAPPING,
    "weaver-mini": WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING,
    "weaver-max": WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-8b-Instruct": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_MAPPING,
}

HOOK_FNS = {
    "gemma2-2b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-9b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma2-27b": GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-4b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-12b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma3-27b": GEMMA3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-26b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-31b": GEMMA4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-e2b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gemma4-e4b": GEMMA4_SMALL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "ming-3.0-audio-tiny": MING_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "hy4-preview": HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-tiny": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-284b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek4-1600b": DEEPSEEK_V4_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-1.5b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen2.5-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-0.6b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-1.7b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-4b-thinking-2507": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-8b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-14b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-32b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-2b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-4b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-vl-30b-a3b": QWEN3_VL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-omni-30b-a3b": QWEN3_OMNI_MOE_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-20b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "gpt-oss-120b": GPT_OSS_TO_HF_PARAM_HOOK_FN,
    "deepseek2-16b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "deepseek3.2-671b": DEEPSEEK_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-next-80b-a3b": QWEN3_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-flash-next-mini": QWEN3_8_FLASH_NEXT_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-397b-a17b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-397b-a17b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-a3b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-a3b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-35b-fp8": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.5-9b": QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3.8-2.4t-a95b": QWEN3_8_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x7b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "mixtral-8x22b": MIXTRAL_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "weaver-mini": WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "weaver-max": WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-30b-a3b-base": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-235b-a22b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "qwen3-coder-480b-a35b": QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-7b-pt": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "olmo3-32b": OLMO3_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-8b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-8b-Instruct": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-70b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    "llama3.1-405b": LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN,
}

VLLM_HOOK_FNS = {
    "deepseek3": DEEPSEEK_NNX_TO_VLLM_PARAM_HOOK_FN,
    "qwen3": QWEN3_NNX_TO_VLLM_PARAM_HOOK_FN,
    "llama3.1": LLAMA31_NNX_TO_VLLM_PARAM_HOOK_FN,
}
