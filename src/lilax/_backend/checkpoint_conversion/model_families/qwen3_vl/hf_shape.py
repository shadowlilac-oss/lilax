"""Hugging Face Qwen3-VL weight shapes for checkpoint conversion."""

from lilax._backend.checkpoint_conversion.model_families.qwen.hf_shape import QWEN_HF_WEIGHTS_TO_SHAPE


def QWEN3_VL_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen3-VL weights path and weights shape."""
  text_shapes = QWEN_HF_WEIGHTS_TO_SHAPE(config["text_config"])
  vl_shapes = {}
  for key, value in text_shapes.items():
    if key.startswith("model."):
      new_key = key.replace("model.", "model.language_model.")
      vl_shapes[new_key] = value
    else:
      vl_shapes[key] = value
  vision_config = config["vision_config"]
  v_depth = vision_config["depth"]
  v_hidden_size = vision_config["hidden_size"]
  v_intermediate_size = vision_config["intermediate_size"]
  v_out_hidden_size = vision_config["out_hidden_size"]
  vl_shapes["model.visual.patch_embed.proj.weight"] = [v_hidden_size, 3, 2, 16, 16]
  vl_shapes["model.visual.patch_embed.proj.bias"] = [v_hidden_size]
  vl_shapes["model.visual.pos_embed.weight"] = [vision_config["num_position_embeddings"], v_hidden_size]
  for layer_idx in range(v_depth):
    prefix = f"model.visual.blocks.{layer_idx}"
    vl_shapes[f"{prefix}.norm1.weight"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm1.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm2.weight"] = [v_hidden_size]
    vl_shapes[f"{prefix}.norm2.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.attn.qkv.weight"] = [3 * v_hidden_size, v_hidden_size]
    vl_shapes[f"{prefix}.attn.qkv.bias"] = [3 * v_hidden_size]
    vl_shapes[f"{prefix}.attn.proj.weight"] = [v_hidden_size, v_hidden_size]
    vl_shapes[f"{prefix}.attn.proj.bias"] = [v_hidden_size]
    vl_shapes[f"{prefix}.mlp.linear_fc1.weight"] = [v_intermediate_size, v_hidden_size]
    vl_shapes[f"{prefix}.mlp.linear_fc1.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.mlp.linear_fc2.weight"] = [v_hidden_size, v_intermediate_size]
    vl_shapes[f"{prefix}.mlp.linear_fc2.bias"] = [v_hidden_size]
  deepstack_indexes = vision_config.get("deepstack_visual_indexes", [5, 11, 17])
  for merger_idx, _ in enumerate(deepstack_indexes):
    prefix = f"model.visual.deepstack_merger_list.{merger_idx}"
    vl_shapes[f"{prefix}.norm.weight"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.norm.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc1.weight"] = [v_intermediate_size, v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc1.bias"] = [v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc2.weight"] = [v_out_hidden_size, v_intermediate_size]
    vl_shapes[f"{prefix}.linear_fc2.bias"] = [v_out_hidden_size]
  vl_shapes["model.visual.merger.norm.weight"] = [v_hidden_size]
  vl_shapes["model.visual.merger.norm.bias"] = [v_hidden_size]
  vl_shapes["model.visual.merger.linear_fc1.weight"] = [v_intermediate_size, v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc1.bias"] = [v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc2.weight"] = [v_out_hidden_size, v_intermediate_size]
  vl_shapes["model.visual.merger.linear_fc2.bias"] = [v_out_hidden_size]
  return vl_shapes
