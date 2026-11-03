"""Hugging Face Gemma 4 weight shapes for checkpoint conversion."""


def _get_gemma4_layer_attention_dims(text_cfg: dict, layer_idx: int, is_global: bool):
  num_attention_heads = text_cfg["num_attention_heads"]
  num_key_value_heads = text_cfg["num_key_value_heads"]
  head_dim = text_cfg["head_dim"]
  global_head_dim = text_cfg.get("global_head_dim") or head_dim
  num_global_key_value_heads = text_cfg.get("num_global_key_value_heads") or num_key_value_heads
  per_layer_config = text_cfg.get("per_layer_config") or {}
  if isinstance(per_layer_config, list):
    layer_override = (
        per_layer_config[layer_idx]
        if layer_idx < len(per_layer_config) and isinstance(per_layer_config[layer_idx], dict) else {}
    )
  elif isinstance(per_layer_config, dict):
    layer_override = (per_layer_config.get(layer_idx) or per_layer_config.get(str(layer_idx))
                      or per_layer_config.get(f"{layer_idx:02d}") or {})
  else:
    layer_override = {}
  layer_heads = layer_override.get("num_attention_heads") or num_attention_heads
  layer_kv_heads = layer_override.get("num_key_value_heads") or (
      num_global_key_value_heads if is_global else num_key_value_heads
  )
  layer_head_dim = layer_override.get("head_dim") or (global_head_dim if is_global else head_dim)
  return layer_heads * layer_head_dim, layer_kv_heads * layer_head_dim, layer_head_dim


def GEMMA4_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_cfg = config.get("text_config", config)
  vision_cfg = config.get("vision_config", {})
  text_base = "model.language_model" if vision_cfg else "model"
  hidden_size = text_cfg["hidden_size"]
  intermediate_size = text_cfg["intermediate_size"]
  num_hidden_layers = text_cfg["num_hidden_layers"]
  vocab_size = text_cfg["vocab_size"]
  num_experts = text_cfg.get("num_experts")
  num_experts = num_experts if num_experts is not None else 1
  expert_intermediate_size = text_cfg.get("moe_intermediate_size") or text_cfg.get("expert_intermediate_size")
  layer_types = text_cfg.get("layer_types", [])
  shapes[f"{text_base}.embed_tokens.weight"] = [vocab_size, hidden_size]
  shapes[f"{text_base}.norm.weight"] = [hidden_size]
  for layer_idx in range(num_hidden_layers):
    prefix = f"{text_base}.layers.{layer_idx}"
    is_global = (layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention") if layer_types else (
        layer_idx % 6 == 5
    )
    q_dim, kv_dim, norm_dim = _get_gemma4_layer_attention_dims(text_cfg, layer_idx, is_global)
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.self_attn.k_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.self_attn.v_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.layer_scalar"] = [1]
    if num_experts > 1:
      shapes[f"{prefix}.pre_feedforward_layernorm_2.weight"] = [hidden_size]
      shapes[f"{prefix}.post_feedforward_layernorm_1.weight"] = [hidden_size]
      shapes[f"{prefix}.post_feedforward_layernorm_2.weight"] = [hidden_size]
      shapes[f"{prefix}.router.scale"] = [hidden_size]
      shapes[f"{prefix}.router.proj.weight"] = [num_experts, hidden_size]
      shapes[f"{prefix}.router.per_expert_scale"] = [num_experts]
      shapes[f"{prefix}.experts.gate_up_proj"] = [num_experts, 2 * expert_intermediate_size, hidden_size]
      shapes[f"{prefix}.experts.down_proj"] = [num_experts, hidden_size, expert_intermediate_size]
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [intermediate_size, hidden_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [intermediate_size, hidden_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [hidden_size, intermediate_size]
  if vision_cfg:
    vis_hidden = vision_cfg["hidden_size"]
    vis_intermediate = vision_cfg["intermediate_size"]
    vis_num_layers = vision_cfg["num_hidden_layers"]
    vis_num_heads = vision_cfg["num_attention_heads"]
    vis_head_dim = vision_cfg["head_dim"]
    vis_q_dim = vis_num_heads * vis_head_dim
    vis_kv_dim = vision_cfg.get("num_key_value_heads", vis_num_heads) * vis_head_dim
    vis_pos_emb_size = vision_cfg.get("position_embedding_size", 10240)
    vis_patch_size = vision_cfg.get("patch_size", 16)
    patch_flat = 3 * vis_patch_size * vis_patch_size
    shapes["model.vision_tower.patch_embedder.input_proj.weight"] = [vis_hidden, patch_flat]
    shapes["model.vision_tower.patch_embedder.position_embedding_table"] = [2, vis_pos_emb_size, vis_hidden]
    shapes["model.vision_tower.std_scale"] = [vis_hidden]
    shapes["model.vision_tower.std_bias"] = [vis_hidden]
    shapes["model.embed_vision.embedding_projection.weight"] = [hidden_size, vis_hidden]
    for layer_idx in range(vis_num_layers):
      prefix = f"model.vision_tower.encoder.layers.{layer_idx}"
      shapes[f"{prefix}.self_attn.q_proj.linear.weight"] = [vis_q_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.k_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.v_proj.linear.weight"] = [vis_kv_dim, vis_hidden]
      shapes[f"{prefix}.self_attn.o_proj.linear.weight"] = [vis_hidden, vis_q_dim]
      shapes[f"{prefix}.self_attn.q_norm.weight"] = [vis_head_dim]
      shapes[f"{prefix}.self_attn.k_norm.weight"] = [vis_head_dim]
      shapes[f"{prefix}.input_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.post_attention_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [vis_hidden]
      shapes[f"{prefix}.mlp.gate_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{prefix}.mlp.up_proj.linear.weight"] = [vis_intermediate, vis_hidden]
      shapes[f"{prefix}.mlp.down_proj.linear.weight"] = [vis_hidden, vis_intermediate]
  return shapes


def GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config):
  shapes = {}
  text_cfg = config.get("text_config", config)
  vision_cfg = config.get("vision_config", {})
  text_base = "model.language_model" if vision_cfg else "model"
  hidden_size = text_cfg["hidden_size"]
  intermediate_size = text_cfg["intermediate_size"]
  num_hidden_layers = text_cfg["num_hidden_layers"]
  vocab_size = text_cfg["vocab_size"]
  layer_types = text_cfg.get("layer_types", [])
  ple_dim = text_cfg.get("hidden_size_per_layer_input", 0) or 0
  vocab_ple = text_cfg.get("vocab_size_per_layer_input", 0) or 0
  num_kv_shared = text_cfg.get("num_kv_shared_layers", 0) or 0
  first_shared = max(0, num_hidden_layers - num_kv_shared) if num_kv_shared > 0 else num_hidden_layers
  use_double_wide_mlp = bool(text_cfg.get("use_double_wide_mlp", False))
  shapes[f"{text_base}.embed_tokens.weight"] = [vocab_size, hidden_size]
  shapes[f"{text_base}.norm.weight"] = [hidden_size]
  if ple_dim > 0:
    shapes[f"{text_base}.embed_tokens_per_layer.weight"] = [vocab_ple, num_hidden_layers * ple_dim]
    shapes[f"{text_base}.per_layer_model_projection.weight"] = [num_hidden_layers * ple_dim, hidden_size]
    shapes[f"{text_base}.per_layer_projection_norm.weight"] = [ple_dim]
  for layer_idx in range(num_hidden_layers):
    prefix = f"{text_base}.layers.{layer_idx}"
    is_global = layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention"
    is_shared = num_kv_shared > 0 and layer_idx >= first_shared
    q_dim, kv_dim, norm_dim = _get_gemma4_layer_attention_dims(text_cfg, layer_idx, is_global)
    shapes[f"{prefix}.self_attn.q_proj.weight"] = [q_dim, hidden_size]
    shapes[f"{prefix}.self_attn.o_proj.weight"] = [hidden_size, q_dim]
    shapes[f"{prefix}.self_attn.q_norm.weight"] = [norm_dim]
    if not is_shared:
      shapes[f"{prefix}.self_attn.k_proj.weight"] = [kv_dim, hidden_size]
      shapes[f"{prefix}.self_attn.v_proj.weight"] = [kv_dim, hidden_size]
      shapes[f"{prefix}.self_attn.k_norm.weight"] = [norm_dim]
      shapes[f"{prefix}.self_attn.v_norm.weight"] = [norm_dim]
    shapes[f"{prefix}.input_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_attention_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.pre_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.post_feedforward_layernorm.weight"] = [hidden_size]
    shapes[f"{prefix}.layer_scalar"] = [1]
    mlp_dim = intermediate_size * 2 if (is_shared and use_double_wide_mlp) else intermediate_size
    shapes[f"{prefix}.mlp.gate_proj.weight"] = [mlp_dim, hidden_size]
    shapes[f"{prefix}.mlp.up_proj.weight"] = [mlp_dim, hidden_size]
    shapes[f"{prefix}.mlp.down_proj.weight"] = [hidden_size, mlp_dim]
    if ple_dim > 0:
      shapes[f"{prefix}.per_layer_input_gate.weight"] = [ple_dim, hidden_size]
      shapes[f"{prefix}.per_layer_projection.weight"] = [hidden_size, ple_dim]
      shapes[f"{prefix}.post_per_layer_input_norm.weight"] = [hidden_size]
  return shapes
