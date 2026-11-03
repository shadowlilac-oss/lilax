"""Mixtral parameter mappings and transformation hooks for checkpoint conversion."""

import numpy as np


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
