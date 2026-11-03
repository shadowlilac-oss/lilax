"""Hugging Face Qwen3.8-Flash-Next weight shapes for checkpoint conversion."""

import math


def _qwen3_8_flash_next_is_prime(number: int) -> bool:
  if number < 2:
    return False
  if number < 4:
    return True
  if number % 2 == 0 or number % 3 == 0:
    return False
  divisor = 5
  while divisor * divisor <= number:
    if number % divisor == 0 or number % (divisor + 2) == 0:
      return False
    divisor += 6
  return True


def _qwen3_8_flash_next_nth_prime_after(start: int, nth: int) -> int:
  candidate = start
  found = 0
  while found < nth:
    candidate += 1
    if _qwen3_8_flash_next_is_prime(candidate):
      found += 1
  return candidate


def QWEN3_8_FLASH_NEXT_HF_WEIGHTS_TO_SHAPE(config):
  if hasattr(config, "to_dict"):
    config = config.to_dict()
  tcfg = (
      config.get("text_config", config)
      if isinstance(config, dict) and isinstance(config.get("text_config"), dict) else config
  )
  if (isinstance(config, dict) and not config.get("language_model_only", False)
      and ("text_config" in config or "vision_config" in config)):
    text_base = "model.language_model"
  else:
    text_base = "model"
  hidden_size = tcfg["hidden_size"]
  vocab_size = tcfg["vocab_size"]
  num_hidden_layers = tcfg["num_hidden_layers"]
  num_attention_heads = tcfg["num_attention_heads"]
  num_key_value_heads = tcfg["num_key_value_heads"]
  head_dim = tcfg["head_dim"]
  hc_count = tcfg.get("hc_count", 4)
  hc_lowrank = tcfg.get("hc_lowrank", 320)
  hc_dim = hc_count * hidden_size
  moe_intermediate_size = tcfg["moe_intermediate_size"]
  shared_expert_intermediate_size = tcfg["shared_expert_intermediate_size"]
  num_experts = tcfg["num_experts"]
  linear_num_key_heads = tcfg["linear_num_key_heads"]
  linear_num_value_heads = tcfg["linear_num_value_heads"]
  linear_key_head_dim = tcfg["linear_key_head_dim"]
  linear_value_head_dim = tcfg["linear_value_head_dim"]
  linear_conv_kernel_dim = tcfg.get("linear_conv_kernel_dim", 4)
  indexer_n_heads = tcfg.get("indexer_n_heads", 4)
  indexer_kv_heads = tcfg.get("indexer_kv_heads", 1)
  indexer_head_dim = tcfg.get("indexer_head_dim", 128)
  layer_types = tcfg.get("layer_types") or [
      "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention" for layer_idx in range(num_hidden_layers)
  ]
  q_dim = num_attention_heads * head_dim
  kv_dim = num_key_value_heads * head_dim
  gdn_k_dim = linear_num_key_heads * linear_key_head_dim
  gdn_v_dim = linear_num_value_heads * linear_value_head_dim
  conv_dim = 2 * gdn_k_dim + gdn_v_dim
  mapping = {
      f"{text_base}.embed_tokens.weight": [vocab_size, hidden_size],
      "lm_head.weight": [vocab_size, hidden_size],
      f"{text_base}.hyper_connection_mixer.hc_norm.weight": [hc_dim],
      f"{text_base}.hyper_connection_mixer.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
      f"{text_base}.hyper_connection_mixer.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
  }

  def _layer_shapes(prefix: str, is_qsa: bool) -> dict:
    layer_map = {
        f"{prefix}.mlp.gate.weight": [num_experts, hidden_size],
        f"{prefix}.mlp.experts.gate_up_proj": [num_experts, 2 * moe_intermediate_size, hidden_size],
        f"{prefix}.mlp.experts.down_proj": [num_experts, hidden_size, moe_intermediate_size],
        f"{prefix}.mlp.shared_expert.gate_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{prefix}.mlp.shared_expert.up_proj.weight": [shared_expert_intermediate_size, hidden_size],
        f"{prefix}.mlp.shared_expert.down_proj.weight": [hidden_size, shared_expert_intermediate_size],
        f"{prefix}.mlp.shared_expert_gate.weight": [1, hidden_size],
    }
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
      layer_map.update({
          f"{prefix}.{hc}.hc_norm.weight": [hc_dim],
          f"{prefix}.{hc}.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
          f"{prefix}.{hc}.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
          f"{prefix}.{hc}.block_inject_weight.weight": [hc_count, hc_dim],
      })
    if is_qsa:
      layer_map.update({
          f"{prefix}.self_attn.q_proj.weight": [2 * q_dim, hidden_size],
          f"{prefix}.self_attn.k_proj.weight": [kv_dim, hidden_size],
          f"{prefix}.self_attn.v_proj.weight": [kv_dim, hidden_size],
          f"{prefix}.self_attn.o_proj.weight": [hidden_size, q_dim],
          f"{prefix}.self_attn.q_norm.weight": [head_dim],
          f"{prefix}.self_attn.k_norm.weight": [head_dim],
          f"{prefix}.self_attn.indexer.index_qk_proj.weight": [
              (indexer_n_heads + indexer_kv_heads) * indexer_head_dim, hidden_size
          ],
          f"{prefix}.self_attn.indexer.q_layernorm.weight": [indexer_head_dim],
          f"{prefix}.self_attn.indexer.k_layernorm.weight": [indexer_head_dim],
      })
    else:
      layer_map.update({
          f"{prefix}.linear_attn.A_log": [linear_num_value_heads],
          f"{prefix}.linear_attn.dt_bias": [linear_num_value_heads],
          f"{prefix}.linear_attn.conv1d.weight": [conv_dim, 1, linear_conv_kernel_dim],
          f"{prefix}.linear_attn.in_proj_qkv.weight": [conv_dim, hidden_size],
          f"{prefix}.linear_attn.in_proj_z.weight": [gdn_v_dim, hidden_size],
          f"{prefix}.linear_attn.in_proj_b.weight": [linear_num_value_heads, hidden_size],
          f"{prefix}.linear_attn.in_proj_a.weight": [linear_num_value_heads, hidden_size],
          f"{prefix}.linear_attn.norm.weight": [linear_value_head_dim],
          f"{prefix}.linear_attn.out_proj.weight": [hidden_size, gdn_v_dim],
      })
    return layer_map

  for layer_idx in range(num_hidden_layers):
    mapping.update(_layer_shapes(f"{text_base}.layers.{layer_idx}", is_qsa=layer_types[layer_idx] == "full_attention"))
  ple_layer_ids = tcfg.get("ple_layer_ids", [2])
  if tcfg.get("use_ple_layer", bool(ple_layer_ids)):
    default_pid = (int(ple_layer_ids[0]) - 1) if ple_layer_ids else 1
    ple_layer_id = tcfg.get("ple_layer_id", default_pid)
    ple_pfx = f"{text_base}.layers.{ple_layer_id}.ple"
    ple_embed_dim = tcfg.get("ple_embed_dim", hidden_size)
    ple_conv_kernel_size = tcfg.get("ple_conv_kernel_size", 4)
    ngram_size = tcfg.get("ngram_size", 3)
    heads_per_ngram = tcfg.get("heads_per_ngram", 8)
    ngram_vocab_size_base = tcfg.get("ngram_vocab_size_base", 20000000)
    split_ngram_parts = max(1, int(tcfg.get("split_ngram_parts", 128)))
    divisible_by = tcfg.get("make_ngram_vocab_size_divisible_by", 128)
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    head_dim_per_ngram = ple_embed_dim // ngram_heads
    total_vocab = sum(_qwen3_8_flash_next_nth_prime_after(ngram_vocab_size_base - 1, head + 1)
                      for head in range(ngram_heads))
    padded_vocab_size = math.ceil(total_vocab / divisible_by) * divisible_by
    shard_rows = padded_vocab_size // split_ngram_parts
    mapping.update({
        f"{ple_pfx}.conv1d.weight": [hc_dim, 1, ple_conv_kernel_size],
        f"{ple_pfx}.key_proj.weight": [hc_dim, ple_embed_dim],
        f"{ple_pfx}.value_proj.weight": [hidden_size, ple_embed_dim],
        f"{ple_pfx}.norm_key.weight": [hc_dim],
        f"{ple_pfx}.norm_query.weight": [hc_dim],
        f"{ple_pfx}.norm_conv.weight": [hc_dim],
        f"{ple_pfx}.ple_embedding.ngram_embedding.weight": [padded_vocab_size, head_dim_per_ngram],
    })
    shard_keys = tuple(f"{ple_pfx}.ple_embedding.ngram_embedding.shard_{shard}.weight"
                       for shard in range(split_ngram_parts))
    for shard_key in shard_keys:
      mapping[shard_key] = [shard_rows, head_dim_per_ngram]
    mapping[shard_keys] = tuple([shard_rows, head_dim_per_ngram] for _ in range(split_ngram_parts))
  mtp_num_layers = tcfg.get("mtp_num_hidden_layers", tcfg.get("mtp_num_layers", 1))
  if mtp_num_layers and mtp_num_layers > 0:
    mapping.update({
        "mtp.pre_fc_norm_embedding.weight": [hidden_size],
        "mtp.pre_fc_norm_hidden.weight": [hc_dim],
        "mtp.fc_embedding.weight": [hidden_size, hidden_size],
        "mtp.fc_hidden.weight": [hidden_size, hidden_size],
        "mtp.hyper_connection_mixer.hc_norm.weight": [hc_dim],
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [hc_lowrank, hc_dim],
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [hc_dim, hc_lowrank],
    })
    for mtp_idx in range(mtp_num_layers):
      mapping.update(_layer_shapes(f"mtp.layers.{mtp_idx}", is_qsa=True))
  return mapping
