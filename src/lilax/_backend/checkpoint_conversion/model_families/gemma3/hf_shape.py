"""Hugging Face Gemma 3 weight shapes for checkpoint conversion."""


def GEMMA3_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_config = config["text_config"]
  vision_config = config["vision_config"]
  lm_hidden_size = text_config["hidden_size"]
  lm_intermediate_size = text_config["intermediate_size"]
  lm_num_layers = text_config["num_hidden_layers"]
  lm_head_dim = text_config["head_dim"]
  lm_q_dim = text_config["num_attention_heads"] * lm_head_dim
  lm_kv_dim = text_config["num_key_value_heads"] * lm_head_dim
  vision_hidden_size = vision_config["hidden_size"]
  vision_intermediate_size = vision_config["intermediate_size"]
  vision_num_layers = vision_config["num_hidden_layers"]
  vision_patch_size = vision_config["patch_size"]
  vision_num_channels = vision_config["num_channels"]
  vision_num_positions = (vision_config["image_size"] // vision_patch_size) ** 2
  vocab_size = text_config["vocab_size"]

  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.weight"] = [
      vision_hidden_size, vision_num_channels, vision_patch_size, vision_patch_size
  ]
  shapes["model.vision_tower.vision_model.embeddings.patch_embedding.bias"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.embeddings.position_embedding.weight"] = [
      vision_num_positions, vision_hidden_size
  ]
  for layer_idx in range(vision_num_layers):
    prefix = f"model.vision_tower.vision_model.encoder.layers.{layer_idx}"
    shapes[f"{prefix}.layer_norm1.weight"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm1.bias"] = [vision_hidden_size]
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
      shapes[f"{prefix}.self_attn.{projection}.weight"] = [vision_hidden_size, vision_hidden_size]
      shapes[f"{prefix}.self_attn.{projection}.bias"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm2.weight"] = [vision_hidden_size]
    shapes[f"{prefix}.layer_norm2.bias"] = [vision_hidden_size]
    shapes[f"{prefix}.mlp.fc1.weight"] = [vision_intermediate_size, vision_hidden_size]
    shapes[f"{prefix}.mlp.fc1.bias"] = [vision_intermediate_size]
    shapes[f"{prefix}.mlp.fc2.weight"] = [vision_hidden_size, vision_intermediate_size]
    shapes[f"{prefix}.mlp.fc2.bias"] = [vision_hidden_size]

  shapes["model.vision_tower.vision_model.post_layernorm.weight"] = [vision_hidden_size]
  shapes["model.vision_tower.vision_model.post_layernorm.bias"] = [vision_hidden_size]
  shapes["model.multi_modal_projector.mm_input_projection_weight"] = [vision_hidden_size, lm_hidden_size]
  shapes["model.multi_modal_projector.mm_soft_emb_norm.weight"] = [vision_hidden_size]
  shapes["model.language_model.embed_tokens.weight"] = [vocab_size, lm_hidden_size]
  for layer_idx in range(lm_num_layers):
    prefix = f"model.language_model.layers.{layer_idx}"
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [lm_q_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.q_proj.bias"] = [lm_q_dim]
    shapes[f"{prefix}.self_attn.k_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.k_proj.bias"] = [lm_kv_dim]
    shapes[f"{prefix}.self_attn.v_proj.weight"] = [lm_kv_dim, lm_hidden_size]
    shapes[f"{prefix}.self_attn.v_proj.bias"] = [lm_kv_dim]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [lm_hidden_size, lm_q_dim]
    shapes[f"{prefix}.self_attn.o_proj.bias"] = [lm_hidden_size]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [lm_head_dim]
    shapes[f"{prefix}.self_attn.k_norm.weight"] = [lm_head_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [lm_hidden_size]
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"{prefix}.mlp.gate_proj.bias"] = [lm_intermediate_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [lm_intermediate_size, lm_hidden_size]
    shapes[f"{prefix}.mlp.up_proj.bias"] = [lm_intermediate_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [lm_hidden_size, lm_intermediate_size]
    shapes[f"{prefix}.mlp.down_proj.bias"] = [lm_hidden_size]
  shapes["model.language_model.norm.weight"] = [lm_hidden_size]
  shapes["lm_head.weight"] = [vocab_size, lm_hidden_size]
  return shapes
