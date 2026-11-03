"""Hugging Face DeepSeek V4 weight shapes for checkpoint conversion."""


def DEEPSEEKV4_HF_WEIGHTS_TO_SHAPE(config):
  hidden_size = config["hidden_size"]
  vocab_size = config["vocab_size"]
  num_hidden_layers = config["num_hidden_layers"]
  q_lora_rank = config.get("q_lora_rank", 1024)
  o_lora_rank = config.get("o_lora_rank", 1024) * config.get("o_groups", 8)
  num_attention_heads = config["num_attention_heads"]
  head_dim = config["head_dim"]
  moe_intermediate_size = config["moe_intermediate_size"]
  n_routed_experts = config["n_routed_experts"]
  hc_mult = config.get("hc_mult", config.get("mhc_expansion_rate", 4))
  hc_dim = (2 * hc_mult) + (hc_mult**2)
  num_hash_layers = config.get("num_hash_layers", config.get("first_num_hash_layers", 3))
  index_n_heads = config.get("index_n_heads", config.get("indexer_n_heads", 64))
  index_head_dim = config.get("index_head_dim", config.get("indexer_head_dim", 128))

  mapping = {
      "embed.weight": [vocab_size, hidden_size],
      "norm.weight": [hidden_size],
      "head.weight": [vocab_size, hidden_size],
      "hc_head_fn": [hc_mult, hidden_size * hc_mult],
      "hc_head_base": [hc_mult],
      "hc_head_scale": [1],
  }

  def _compressor_shapes(comp_prefix, comp_dim, norm_dim, ratio):
    return {
        f"{comp_prefix}.wgate.weight": [comp_dim, hidden_size],
        f"{comp_prefix}.wkv.weight": [comp_dim, hidden_size],
        f"{comp_prefix}.norm.weight": [norm_dim],
        f"{comp_prefix}.ape": [ratio, comp_dim],
    }

  def _layer_shapes(layer_prefix, is_hash, ratio):
    layer_mapping = {
        f"{layer_prefix}.attn_norm.weight": [hidden_size],
        f"{layer_prefix}.ffn_norm.weight": [hidden_size],
        f"{layer_prefix}.attn.wq_a.weight": [q_lora_rank, hidden_size],
        f"{layer_prefix}.attn.q_norm.weight": [q_lora_rank],
        f"{layer_prefix}.attn.wq_b.weight": [num_attention_heads * head_dim, q_lora_rank],
        f"{layer_prefix}.attn.wkv.weight": [
            config.get("num_key_value_heads", 1) * config.get("head_dim", 64), hidden_size
        ],
        f"{layer_prefix}.attn.kv_norm.weight": [head_dim],
        f"{layer_prefix}.attn.attn_sink": [num_attention_heads],
        f"{layer_prefix}.attn.wo_a.weight": [
            o_lora_rank, (num_attention_heads * head_dim) // config.get("o_groups", 8)
        ],
        f"{layer_prefix}.attn.wo_b.weight": [hidden_size, o_lora_rank],
        f"{layer_prefix}.hc_attn_fn": [hc_dim, hidden_size * hc_mult],
        f"{layer_prefix}.hc_attn_base": [hc_dim],
        f"{layer_prefix}.hc_attn_scale": [3],
        f"{layer_prefix}.hc_ffn_fn": [hc_dim, hidden_size * hc_mult],
        f"{layer_prefix}.hc_ffn_base": [hc_dim],
        f"{layer_prefix}.hc_ffn_scale": [3],
        f"{layer_prefix}.ffn.gate.weight": [n_routed_experts, hidden_size],
        f"{layer_prefix}.ffn.gate.bias": [n_routed_experts],
    }
    if is_hash:
      layer_mapping[f"{layer_prefix}.ffn.gate.tid2eid"] = [vocab_size, config.get("num_experts_per_tok", 2)]
    if ratio > 0:
      comp_dim = 2 * head_dim if ratio == 4 else head_dim
      layer_mapping.update(_compressor_shapes(f"{layer_prefix}.attn.compressor", comp_dim, head_dim, ratio))
      if ratio == 4:
        layer_mapping.update(
            _compressor_shapes(f"{layer_prefix}.attn.indexer.compressor", 2 * index_head_dim, index_head_dim, ratio)
        )
        layer_mapping[f"{layer_prefix}.attn.indexer.weights_proj.weight"] = [index_n_heads, hidden_size]
        layer_mapping[f"{layer_prefix}.attn.indexer.wq_b.weight"] = [index_n_heads * index_head_dim, q_lora_rank]
    for exp_idx in range(n_routed_experts):
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w1.weight"] = [moe_intermediate_size, hidden_size]
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w2.weight"] = [hidden_size, moe_intermediate_size]
      layer_mapping[f"{layer_prefix}.ffn.experts.{exp_idx}.w3.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w1.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w3.weight"] = [moe_intermediate_size, hidden_size]
    layer_mapping[f"{layer_prefix}.ffn.shared_experts.w2.weight"] = [hidden_size, moe_intermediate_size]
    return layer_mapping

  compress_ratios = config["compress_ratios"]
  for layer_idx in range(num_hidden_layers):
    mapping.update(_layer_shapes(f"layers.{layer_idx}", is_hash=layer_idx < num_hash_layers,
                                 ratio=compress_ratios[layer_idx]))

  mtp_num_layers = config.get("num_nextn_predict_layers", config.get("num_mtp_layers", config.get("mtp_num_layers", 0)))
  for mtp_idx in range(mtp_num_layers):
    mtp_prefix = f"mtp.{mtp_idx}"
    mapping.update(_layer_shapes(mtp_prefix, is_hash=False, ratio=0))
    mapping.update({
        f"{mtp_prefix}.enorm.weight": [hidden_size],
        f"{mtp_prefix}.hnorm.weight": [hidden_size],
        f"{mtp_prefix}.norm.weight": [hidden_size],
        f"{mtp_prefix}.e_proj.weight": [hidden_size, hidden_size],
        f"{mtp_prefix}.h_proj.weight": [hidden_size, hidden_size],
        (f"{mtp_prefix}.e_proj.weight", f"{mtp_prefix}.h_proj.weight"): (
            [hidden_size, hidden_size], [hidden_size, hidden_size]
        ),
        f"{mtp_prefix}.hc_head_fn": [hc_mult, hidden_size * hc_mult],
        f"{mtp_prefix}.hc_head_base": [hc_mult],
        f"{mtp_prefix}.hc_head_scale": [1],
    })
  return mapping
