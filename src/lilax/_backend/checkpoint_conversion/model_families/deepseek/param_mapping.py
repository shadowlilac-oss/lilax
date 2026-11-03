"""DeepSeek V2 / V3 / V3.2 parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


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
