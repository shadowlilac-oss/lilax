"""Ming parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


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
