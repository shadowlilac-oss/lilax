import gc
import logging
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import psutil
import torch
from safetensors import safe_open
from tqdm import tqdm

from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from lilax._backend.observability import max_logging


MODEL_PARAMS_DICT = {
    "deepseek2-16b": {
        "num_layers": 27, "first_num_dense_layers": 1, "base_num_query_heads": 16, "base_emb_dim": 2048,
        "num_experts": 64, "q_lora_rank": 0, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "has_mtp": False,
    },
    "deepseek3-671b": {
        "num_layers": 61, "first_num_dense_layers": 3, "base_num_query_heads": 128, "base_emb_dim": 7168,
        "num_experts": 256, "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "has_mtp": True,
    },
    "kimi-k2-1t": {
        "num_layers": 61, "first_num_dense_layers": 1, "base_num_query_heads": 64, "base_emb_dim": 7168,
        "num_experts": 384, "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "has_mtp": False,
    },
}
MODEL_PARAMS_DICT["kimi-k2-thinking"] = {**MODEL_PARAMS_DICT["kimi-k2-1t"], "compressed_int4": True}
MODEL_PARAMS_DICT["kimi-k2.5-text"] = {
    **MODEL_PARAMS_DICT["kimi-k2-thinking"], "hf_key_prefix": "language_model.",
}
MODEL_PARAMS_DICT["kimi-k2.6-text"] = dict(MODEL_PARAMS_DICT["kimi-k2.5-text"])
WEIGHT_DTYPES = {"float16": torch.float16, "float32": torch.float32}
INT4_GROUP_SIZE = 32


def _to_numpy(tensor: torch.Tensor, weight_dtype: str) -> np.ndarray:
  if weight_dtype != "keep":
    tensor = tensor.to(WEIGHT_DTYPES[weight_dtype])
  elif tensor.dtype == torch.bfloat16:
    tensor = tensor.to(torch.float16)
  return tensor.numpy()


def dequantize_pack_quantized_int4(packed, scale, out_shape):
  out_features, in_features = int(out_shape[0]), int(out_shape[1])
  if in_features % INT4_GROUP_SIZE != 0:
    raise ValueError(f"in_features={in_features} not divisible by group_size={INT4_GROUP_SIZE}")

  shifts = torch.arange(8, dtype=torch.int32) * 4
  nibbles = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
  w_int = (nibbles - 8).reshape(out_features, -1)[:, :in_features].to(torch.float32)

  s = scale.to(torch.float32).unsqueeze(-1)
  w = (w_int.reshape(out_features, in_features // INT4_GROUP_SIZE, INT4_GROUP_SIZE) * s).reshape(
      out_features, in_features
  )
  return w.to(torch.bfloat16)


MTP_KEYS_TO_SKIP = [
    "model.layers.61.embed_tokens.weight",
    "model.layers.61.shared_head.head.weight",
]


def is_key_allowed(key, banned_keys) -> bool:
  """
  Checks if a key is NOT in a list of banned keys.
  """
  return key not in banned_keys


def hf_to_maxtext_mapping(layer_idx, num_experts, first_num_dense_layers, num_main_layers, has_mtp=False) -> dict:
  """
  Generates a mapping between Hugging Face (HF) and MaxText model weight names.
  HF MLP is using self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)).


  Args:
      layer_idx: The index of the current layer.
      num_experts: The number of experts in MoE layers.
      first_num_dense_layers: The number of initial dense layers.
      num_main_layers: The total number of main layers in the model.
      has_mtp: Whether the model has an MTP head.

  Returns:
      A dictionary mapping HF weight names to MaxText weight names.
  """
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "lm_head.weight": "logits_dense.kernel",
      "model.norm.weight": "decoder_norm.scale",
  }

  if layer_idx < first_num_dense_layers:
    mapping.update(
        {
            f"model.layers.{layer_idx}.input_layernorm.weight": f"dense_layers.{layer_idx}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"dense_layers.{layer_idx}.post_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"dense_layers.{layer_idx}.self_attention.query.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"dense_layers.{layer_idx}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"dense_layers.{layer_idx}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.up_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_1.kernel",
            f"model.layers.{layer_idx}.mlp.down_proj.weight": f"dense_layers.{layer_idx}.mlp.wo.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.kv_norm.scale",
        }
    )
  elif layer_idx < num_main_layers:
    moe_layer_idx_in_maxtext = layer_idx - first_num_dense_layers
    for expert_idx in range(num_experts):
      mapping.update(
          {
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wo",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_0",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_1",
          }
      )
    mapping.update(
        {
            f"model.layers.{layer_idx}.mlp.gate.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel",
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias",
            f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wo.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.kv_norm.scale",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.query.kernel",
            f"model.layers.{layer_idx}.input_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.post_self_attention_layer_norm.scale",
        }
    )
  elif has_mtp and layer_idx == num_main_layers:
    mapping.update(
        {
            f"model.layers.{layer_idx}.enorm.weight": "mtp_block.mtp_layer_1.mtp_1_embedding_norm.scale",
            f"model.layers.{layer_idx}.hnorm.weight": "mtp_block.mtp_layer_1.mtp_1_hidden_state_norm.scale",
            f"model.layers.{layer_idx}.eh_proj.weight": "mtp_block.mtp_layer_1.mtp_1_projection.kernel",
            f"model.layers.{layer_idx}.shared_head.norm.weight": "mtp_block.mtp_layer_1.mtp_1_final_norm.scale",
        }
    )
    for expert_idx in range(num_experts):
      mapping.update(
          {
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wo",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_0",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_1",
          }
      )
    mapping.update(
        {
            f"model.layers.{layer_idx}.mlp.gate.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel",
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias",
            f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wo.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.kv_norm.scale",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.out.kernel",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.query.kernel",
            f"model.layers.{layer_idx}.input_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.post_self_attention_layer_norm.scale",
        }
    )
  return mapping


def _convert_huggingface_to_jax_weights(
    base_model_path, model_params, mem_info, enable_mtp=False, weight_dtype="float16"
) -> dict:
  """Convert Huggingface Checkpoint to Jax, storing weights as `weight_dtype` (see `_to_numpy`)."""
  base_num_decoder_layers = model_params["num_layers"]
  first_num_dense_layers = model_params["first_num_dense_layers"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_emb_dim = model_params["base_emb_dim"]
  num_experts = model_params["num_experts"]
  q_lora_rank = model_params["q_lora_rank"]
  kv_lora_rank = model_params["kv_lora_rank"]
  qk_nope_head_dim = model_params["qk_nope_head_dim"]
  qk_rope_head_dim = model_params["qk_rope_head_dim"]
  v_head_dim = model_params["v_head_dim"]
  has_mtp = model_params.get("has_mtp", False) and enable_mtp

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  is_compressed = bool(model_params.get("compressed_int4", False))
  hf_key_prefix = model_params.get("hf_key_prefix", "")

  def _normalize(raw_key):
    """Strip multimodal prefix; return None to drop keys outside the text branch."""
    if not hf_key_prefix:
      return raw_key
    if raw_key.startswith(hf_key_prefix):
      return raw_key[len(hf_key_prefix) :]
    return None

  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for raw_key in f.keys():
        key = _normalize(raw_key)
        if key is None:
          continue
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if is_compressed and key.endswith((".weight_scale", ".weight_shape")):
          continue

        is_packed = is_compressed and key.endswith(".weight_packed")
        hf_key = key[: -len(".weight_packed")] + ".weight" if is_packed else key

        if not is_key_allowed(hf_key, MTP_KEYS_TO_SKIP):
          continue
        layer = int(hf_key.split(".")[2]) if "layers" in hf_key else 0
        mapped_key = hf_to_maxtext_mapping(
            layer, num_experts, first_num_dense_layers, base_num_decoder_layers, has_mtp
        ).get(hf_key)
        if not mapped_key:
          continue

        if is_packed:
          raw_base = raw_key[: -len(".weight_packed")]
          chkpt_vars[mapped_key] = dequantize_pack_quantized_int4(
              f.get_tensor(raw_key),
              f.get_tensor(raw_base + ".weight_scale"),
              f.get_tensor(raw_base + ".weight_shape").tolist(),
          )
        else:
          chkpt_vars[mapped_key] = f.get_tensor(raw_key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = {
      "decoder": {
          "dense_layers": {
              "mlp": {
                  "wi_0": {"kernel": None},
                  "wi_1": {"kernel": None},
                  "wo": {"kernel": None},
              },
              "self_attention": {
                  "kv_norm": {"scale": None},
                  "wkv_a": {"kernel": None},
                  "wkv_b": {"kernel": None},
                  "out": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
          "moe_layers": {
              "DeepSeekMoeBlock_0": {
                  "MoeBlock_0": {
                      "wi_0": None,
                      "wi_1": None,
                      "wo": None,
                      "gate": {"kernel": None},
                  },
                  "shared_experts": {
                      "wi_0": {"kernel": None},
                      "wi_1": {"kernel": None},
                      "wo": {"kernel": None},
                  },
              },
              "self_attention": {
                  "kv_norm": {"scale": None},
                  "wkv_a": {"kernel": None},
                  "wkv_b": {"kernel": None},
                  "out": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  max_logging.log("Processing decoder norm scale")
  jax_weights["decoder"]["decoder_norm"]["scale"] = _to_numpy(chkpt_vars["decoder_norm.scale"], weight_dtype)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log("Processing logits dense")
  jax_weights["decoder"]["logits_dense"]["kernel"] = _to_numpy(
      chkpt_vars["logits_dense.kernel"], weight_dtype
  ).transpose()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log("Processing token embeddings")
  jax_weights["token_embedder"]["embedding"] = _to_numpy(chkpt_vars["token_embedder.embedding"], weight_dtype)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  layers = {
      "dense_layers": first_num_dense_layers,
      "moe_layers": base_num_decoder_layers - first_num_dense_layers,
  }
  max_logging.log("Processing self attention and normalization in dense layer")
  for layer_key, layer_value in layers.items():
    self_attention = jax_weights["decoder"][f"{layer_key}"]["self_attention"]
    pre_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"]
    post_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"]

    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
      pre_self_attention = _to_numpy(
          chkpt_vars[f"{layer_key}.{layer_idx}.pre_self_attention_layer_norm.scale"], weight_dtype
      )
      post_self_attention = _to_numpy(
          chkpt_vars[f"{layer_key}.{layer_idx}.post_self_attention_layer_norm.scale"], weight_dtype
      )
      kv_norm = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.kv_norm.scale"], weight_dtype).transpose()
      wkv_a = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_a.kernel"], weight_dtype).transpose()
      wkv_b = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_b.kernel"], weight_dtype).transpose()
      out = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.out.kernel"], weight_dtype).transpose()
      if q_lora_rank != 0:
        q_norm = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.q_norm.scale"], weight_dtype)
        wq_a = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_a.kernel"], weight_dtype).transpose()
        wq_b = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_b.kernel"], weight_dtype).transpose()
      else:
        query = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"], weight_dtype).transpose()

      wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
      out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
      if q_lora_rank != 0:
        wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
      else:
        query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

      if self_attention["kv_norm"]["scale"] is None:
        stack_shape = (layer_value,)
        self_attention["kv_norm"]["scale"] = np.zeros(stack_shape + kv_norm.shape, dtype=kv_norm.dtype)
        self_attention["wkv_a"]["kernel"] = np.zeros(stack_shape + wkv_a.shape, dtype=wkv_a.dtype)
        self_attention["wkv_b"]["kernel"] = np.zeros(stack_shape + wkv_b.shape, dtype=wkv_b.dtype)
        self_attention["out"]["kernel"] = np.zeros(stack_shape + out.shape, dtype=out.dtype)
        pre_self_attention_layer_norm["scale"] = np.zeros(
            stack_shape + pre_self_attention.shape, dtype=pre_self_attention.dtype
        )
        post_self_attention_layer_norm["scale"] = np.zeros(
            stack_shape + post_self_attention.shape, dtype=post_self_attention.dtype
        )
        if q_lora_rank != 0:
          self_attention.update(
              {
                  "q_norm": {"scale": None},
                  "wq_a": {"kernel": None},
                  "wq_b": {"kernel": None},
              }
          )
          self_attention["q_norm"]["scale"] = np.zeros(stack_shape + q_norm.shape, dtype=q_norm.dtype)
          self_attention["wq_a"]["kernel"] = np.zeros(stack_shape + wq_a.shape, dtype=wq_a.dtype)
          self_attention["wq_b"]["kernel"] = np.zeros(stack_shape + wq_b.shape, dtype=wq_b.dtype)
        else:
          self_attention.update({"query": {"kernel": None}})
          self_attention["query"]["kernel"] = np.zeros(stack_shape + query.shape, dtype=query.dtype)

      self_attention["kv_norm"]["scale"][layer_idx, ...] = kv_norm
      self_attention["wkv_a"]["kernel"][layer_idx, ...] = wkv_a
      self_attention["wkv_b"]["kernel"][layer_idx, ...] = wkv_b
      self_attention["out"]["kernel"][layer_idx, ...] = out
      pre_self_attention_layer_norm["scale"][layer_idx, ...] = pre_self_attention
      post_self_attention_layer_norm["scale"][layer_idx, ...] = post_self_attention
      if q_lora_rank != 0:
        self_attention["q_norm"]["scale"][layer_idx, ...] = q_norm
        self_attention["wq_a"]["kernel"][layer_idx, ...] = wq_a
        self_attention["wq_b"]["kernel"][layer_idx, ...] = wq_b
      else:
        self_attention["query"]["kernel"][layer_idx, ...] = query

    self_attention["kv_norm"]["scale"] = np.transpose(self_attention["kv_norm"]["scale"], axes=(1, 0))
    self_attention["wkv_a"]["kernel"] = np.transpose(self_attention["wkv_a"]["kernel"], axes=(1, 0, 2))
    self_attention["wkv_b"]["kernel"] = np.transpose(self_attention["wkv_b"]["kernel"], axes=(1, 0, 2, 3))
    self_attention["out"]["kernel"] = np.transpose(self_attention["out"]["kernel"], axes=(1, 0, 2, 3))
    pre_self_attention_layer_norm["scale"] = np.transpose(pre_self_attention_layer_norm["scale"], axes=(1, 0))
    post_self_attention_layer_norm["scale"] = np.transpose(post_self_attention_layer_norm["scale"], axes=(1, 0))
    if q_lora_rank != 0:
      self_attention["q_norm"]["scale"] = np.transpose(self_attention["q_norm"]["scale"], axes=(1, 0))
      self_attention["wq_a"]["kernel"] = np.transpose(self_attention["wq_a"]["kernel"], axes=(1, 0, 2))
      self_attention["wq_b"]["kernel"] = np.transpose(self_attention["wq_b"]["kernel"], axes=(1, 0, 2, 3))
    else:
      self_attention["query"]["kernel"] = np.transpose(self_attention["query"]["kernel"], axes=(1, 0, 2, 3))

    jax_weights["decoder"][f"{layer_key}"]["self_attention"] = self_attention
    jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
    jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"] = post_self_attention_layer_norm
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log("Processing layer weights")
  for layer_key, layer_value in layers.items():
    if layer_key == "dense_layers":
      mlp = jax_weights["decoder"][f"{layer_key}"]["mlp"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        wi_0 = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"], weight_dtype).transpose()
        wi_1 = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"], weight_dtype).transpose()
        wo = _to_numpy(chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"], weight_dtype).transpose()
        if mlp["wo"]["kernel"] is None:
          stack_shape = (layer_value,)
          mlp["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=wi_0.dtype)
          mlp["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=wi_1.dtype)
          mlp["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=wo.dtype)
        mlp["wi_0"]["kernel"][layer_idx, ...] = wi_0
        mlp["wi_1"]["kernel"][layer_idx, ...] = wi_1
        mlp["wo"]["kernel"][layer_idx, ...] = wo

      mlp["wi_0"]["kernel"] = np.transpose(mlp["wi_0"]["kernel"], axes=(1, 0, 2))
      mlp["wi_1"]["kernel"] = np.transpose(mlp["wi_1"]["kernel"], axes=(1, 0, 2))
      mlp["wo"]["kernel"] = np.transpose(mlp["wo"]["kernel"], axes=(1, 0, 2))
      jax_weights["decoder"][f"{layer_key}"]["mlp"] = mlp
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    else:
      moe = jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        if q_lora_rank != 0:
          gate_bias = _to_numpy(
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"], weight_dtype
          ).transpose()
        gate = _to_numpy(
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"], weight_dtype
        ).transpose()
        shared_wi_0 = _to_numpy(
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"], weight_dtype
        ).transpose()
        shared_wi_1 = _to_numpy(
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"], weight_dtype
        ).transpose()
        shared_wo = _to_numpy(
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wo.kernel"], weight_dtype
        ).transpose()

        if moe["MoeBlock_0"]["gate"]["kernel"] is None:
          stack_shape = (layer_value,)
          if q_lora_rank != 0:
            moe["MoeBlock_0"]["gate"].update({"bias": None})
            moe["MoeBlock_0"]["gate"]["bias"] = np.zeros(stack_shape + gate_bias.shape, dtype=gate_bias.dtype)
          moe["MoeBlock_0"]["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=gate.dtype)
          moe["shared_experts"]["wi_0"]["kernel"] = np.zeros(stack_shape + shared_wi_0.shape, dtype=shared_wi_0.dtype)
          moe["shared_experts"]["wi_1"]["kernel"] = np.zeros(stack_shape + shared_wi_1.shape, dtype=shared_wi_1.dtype)
          moe["shared_experts"]["wo"]["kernel"] = np.zeros(stack_shape + shared_wo.shape, dtype=shared_wo.dtype)

        if q_lora_rank != 0:
          moe["MoeBlock_0"]["gate"]["bias"][layer_idx, ...] = gate_bias
        moe["MoeBlock_0"]["gate"]["kernel"][layer_idx, ...] = gate
        moe["shared_experts"]["wi_0"]["kernel"][layer_idx, ...] = shared_wi_0
        moe["shared_experts"]["wi_1"]["kernel"][layer_idx, ...] = shared_wi_1
        moe["shared_experts"]["wo"]["kernel"][layer_idx, ...] = shared_wo

      moe["MoeBlock_0"]["gate"]["kernel"] = np.transpose(moe["MoeBlock_0"]["gate"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_0"]["kernel"] = np.transpose(moe["shared_experts"]["wi_0"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_1"]["kernel"] = np.transpose(moe["shared_experts"]["wi_1"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wo"]["kernel"] = np.transpose(moe["shared_experts"]["wo"]["kernel"], axes=(1, 0, 2))

      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        for k in tqdm(range(num_experts), desc="experts", leave=False):
          wi_0 = _to_numpy(
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"], weight_dtype
          ).transpose()
          wi_1 = _to_numpy(
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"], weight_dtype
          ).transpose()
          wo = _to_numpy(
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"], weight_dtype
          ).transpose()

          if moe["MoeBlock_0"]["wi_0"] is None:
            stack_shape = (num_experts, layer_value)
            moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=wi_0.dtype)
            moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=wi_1.dtype)
            moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=wo.dtype)
          moe["MoeBlock_0"]["wi_0"][k, layer_idx, ...] = wi_0
          moe["MoeBlock_0"]["wi_1"][k, layer_idx, ...] = wi_1
          moe["MoeBlock_0"]["wo"][k, layer_idx, ...] = wo

      jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"] = moe
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  if has_mtp:
    max_logging.log("Processing MTP Layer")

    jax_weights["mtp_block"] = {
        "mtp_layer_1": {
            "mtp_1_embedding_norm": {"scale": None},
            "mtp_1_hidden_state_norm": {"scale": None},
            "mtp_1_projection": {"kernel": None},
            "mtp_1_final_norm": {"scale": None},
            "mtp_1_transformer_layer": {
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "kv_norm": {"scale": None},
                    "wkv_a": {"kernel": None},
                    "wkv_b": {"kernel": None},
                    "out": {"kernel": None},
                },
                "DeepSeekMoeBlock_0": {
                    "MoeBlock_0": {
                        "wi_0": None,
                        "wi_1": None,
                        "wo": None,
                        "gate": {"kernel": None},
                    },
                    "shared_experts": {
                        "wi_0": {"kernel": None},
                        "wi_1": {"kernel": None},
                        "wo": {"kernel": None},
                    },
                },
            },
        }
    }

    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_embedding_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_embedding_norm.scale"], weight_dtype
    )
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_hidden_state_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_hidden_state_norm.scale"], weight_dtype
    )
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_projection"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_projection.kernel"], weight_dtype
    ).transpose()
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_final_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_final_norm.scale"], weight_dtype
    )

    mtp_transformer_layer = jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_transformer_layer"]
    mtp_transformer_layer["pre_self_attention_layer_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.pre_self_attention_layer_norm.scale"], weight_dtype
    )
    mtp_transformer_layer["post_self_attention_layer_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.post_self_attention_layer_norm.scale"], weight_dtype
    )

    mtp_attn_block = mtp_transformer_layer["self_attention"]
    mtp_attn_block["kv_norm"]["scale"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.kv_norm.scale"], weight_dtype
    ).transpose()
    mtp_attn_block["wkv_a"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_a.kernel"], weight_dtype
    ).transpose()
    wkv_b = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_b.kernel"], weight_dtype
    ).transpose()
    out = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.out.kernel"], weight_dtype
    ).transpose()
    mtp_attn_block["wkv_b"]["kernel"] = np.reshape(
        wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)]
    )
    mtp_attn_block["out"]["kernel"] = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
    if q_lora_rank != 0:
      mtp_attn_block.update(
          {
              "q_norm": {"scale": None},
              "wq_a": {"kernel": None},
              "wq_b": {"kernel": None},
          }
      )
    else:
      mtp_attn_block.update({"query": {"kernel": None}})
    if q_lora_rank != 0:
      mtp_attn_block["q_norm"]["scale"] = _to_numpy(
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.q_norm.scale"], weight_dtype
      )
      mtp_attn_block["wq_a"]["kernel"] = _to_numpy(
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_a.kernel"], weight_dtype
      ).transpose()
      wq_b = _to_numpy(
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_b.kernel"], weight_dtype
      ).transpose()
      mtp_attn_block["wq_b"]["kernel"] = np.reshape(
          wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)]
      )
    else:
      query = _to_numpy(
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.query.kernel"], weight_dtype
      ).transpose()
      mtp_attn_block["query"]["kernel"] = np.reshape(
          query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)]
      )

    moe = mtp_transformer_layer["DeepSeekMoeBlock_0"]
    moe["MoeBlock_0"]["gate"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"],
        weight_dtype,
    ).transpose()
    moe["shared_experts"]["wi_0"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"],
        weight_dtype,
    ).transpose()
    moe["shared_experts"]["wi_1"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"],
        weight_dtype,
    ).transpose()
    moe["shared_experts"]["wo"]["kernel"] = _to_numpy(
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wo.kernel"],
        weight_dtype,
    ).transpose()
    if q_lora_rank != 0:
      moe["MoeBlock_0"]["gate"].update({"bias": None})
      moe["MoeBlock_0"]["gate"]["bias"] = _to_numpy(
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"],
          weight_dtype,
      ).transpose()

    experts_wi0, experts_wi1, experts_wo = [], [], []
    for k in range(num_experts):
      experts_wi0.append(
          _to_numpy(
              chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"],
              weight_dtype,
          ).transpose()
      )
      experts_wi1.append(
          _to_numpy(
              chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"],
              weight_dtype,
          ).transpose()
      )
      experts_wo.append(
          _to_numpy(
              chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"],
              weight_dtype,
          ).transpose()
      )

    moe["MoeBlock_0"]["wi_0"] = np.stack(experts_wi0)
    moe["MoeBlock_0"]["wi_1"] = np.stack(experts_wi1)
    moe["MoeBlock_0"]["wo"] = np.stack(experts_wo)

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None,
                           enable_mtp: bool = False, weight_dtype: str = "float16", scanned: bool = True) -> dict:
  params = MODEL_PARAMS_DICT[model_size] if model_params is None else model_params
  if scanned:
    max_logging.log(f"Loading the base model from {base_model_path}")
    return _convert_huggingface_to_jax_weights(base_model_path, params, psutil.Process(), enable_mtp, weight_dtype)
  tensors = {}
  key_prefix = params.get("hf_key_prefix", "")
  compressed = params.get("compressed_int4", False)
  has_mtp = scanned and enable_mtp and params.get("has_mtp", False)
  common_keys = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
  attention_keys = {
      "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
      "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
      "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
      "self_attn.kv_a_layernorm.weight",
  }
  dense_keys = {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
  moe_keys = {
      "mlp.gate.weight", "mlp.gate.e_score_correction_bias", "mlp.shared_experts.down_proj.weight",
      "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
  }
  for expert in range(params["num_experts"]):
    for projection in ("down_proj", "gate_proj", "up_proj"):
      moe_keys.add(f"mlp.experts.{expert}.{projection}.weight")
  mtp_keys = {"enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"}
  for path in sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as source:
      for raw_key in source.keys():
        if not raw_key.startswith(key_prefix):
          continue
        key = raw_key[len(key_prefix):]
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if compressed and key.endswith((".weight_scale", ".weight_shape")):
          continue
        is_packed = compressed and key.endswith(".weight_packed")
        logical_key = key[:-len(".weight_packed")] + ".weight" if is_packed else key
        if logical_key in ("model.layers.61.embed_tokens.weight", "model.layers.61.shared_head.head.weight"):
          continue
        layer_index = int(logical_key.split(".")[2]) if "layers" in logical_key else 0
        if logical_key not in common_keys:
          prefix = f"model.layers.{layer_index}."
          suffix = logical_key[len(prefix):] if logical_key.startswith(prefix) else ""
          if layer_index < params["first_num_dense_layers"]:
            allowed = attention_keys | dense_keys
          elif layer_index < params["num_layers"]:
            allowed = attention_keys | moe_keys
          elif has_mtp and layer_index == params["num_layers"]:
            allowed = attention_keys | moe_keys | mtp_keys
          else:
            allowed = set()
          if suffix not in allowed:
            continue
        if is_packed:
          base = raw_key[:-len(".weight_packed")]
          tensors[key[:-len(".weight_packed")] + ".weight"] = dequantize_pack_quantized_int4(
              source.get_tensor(raw_key), source.get_tensor(base + ".weight_scale"),
              source.get_tensor(base + ".weight_shape").tolist(),
          )
        else:
          tensors[key] = source.get_tensor(raw_key)

  def value(key, transpose=False):
    tensor = tensors[key]
    if weight_dtype != "keep":
      tensor = tensor.to(WEIGHT_DTYPES[weight_dtype])
    elif tensor.dtype == torch.bfloat16:
      tensor = tensor.to(torch.float16)
    array = tensor.numpy()
    return array.transpose() if transpose else array

  heads, emb = params["base_num_query_heads"], params["base_emb_dim"]
  q_rank, kv_rank = params["q_lora_rank"], params["kv_lora_rank"]
  qk_dim = params["qk_nope_head_dim"] + params["qk_rope_head_dim"]
  kv_dim = params["qk_nope_head_dim"] + params["v_head_dim"]
  mtp_expert_values = {name: [] for name in ("wi_0", "wi_1", "wo")}

  def layer(index, dense, include_experts=True, part=None):
    prefix = f"model.layers.{index}."
    def get(name, transpose=False):
      return value(prefix + name, transpose)

    if part == "mlp":
      result = {}
    else:
      pre = get("input_layernorm.weight")
      post = get("post_attention_layernorm.weight")
      kv_norm = get("self_attn.kv_a_layernorm.weight", True)
      wkv_a = get("self_attn.kv_a_proj_with_mqa.weight", True)
      wkv_b = get("self_attn.kv_b_proj.weight", True)
      out = get("self_attn.o_proj.weight", True)
      is_mtp_layer = scanned and index == params["num_layers"]
      if q_rank and not is_mtp_layer:
        q_norm = get("self_attn.q_a_layernorm.weight")
        wq_a = get("self_attn.q_a_proj.weight", True)
        wq_b = get("self_attn.q_b_proj.weight", True)
      elif not is_mtp_layer:
        query = get("self_attn.q_proj.weight", True)
      attention = {
        "kv_norm": {"scale": kv_norm},
        "wkv_a": {"kernel": wkv_a},
        "wkv_b": {"kernel": np.reshape(wkv_b, [kv_rank, heads, kv_dim])},
        "out": {"kernel": np.reshape(out, [heads, params["v_head_dim"], emb])},
      }
      if q_rank:
        if is_mtp_layer:
          q_norm = get("self_attn.q_a_layernorm.weight")
          wq_a = get("self_attn.q_a_proj.weight", True)
          wq_b = get("self_attn.q_b_proj.weight", True)
        attention.update({
          "q_norm": {"scale": q_norm},
          "wq_a": {"kernel": wq_a},
          "wq_b": {"kernel": np.reshape(wq_b, [q_rank, heads, qk_dim])},
        })
      else:
        if is_mtp_layer:
          query = get("self_attn.q_proj.weight", True)
        attention["query"] = {"kernel": np.reshape(query, [emb, heads, qk_dim])}
      result = {
          "self_attention": attention,
          "pre_self_attention_layer_norm": {"scale": pre},
          "post_self_attention_layer_norm": {"scale": post},
      }
      if part == "attention":
        return result
    projections = (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj"))
    if dense:
      result["mlp"] = {name: {"kernel": get(f"mlp.{source}.weight", True)} for name, source in projections}
    else:
      if q_rank and part == "mlp":
        bias = get("mlp.gate.e_score_correction_bias", True)
      gate_kernel = get("mlp.gate.weight", True)
      shared = {
          name: {"kernel": get(f"mlp.shared_experts.{source}.weight", True)} for name, source in projections
      }
      gate = {}
      if q_rank and part == "mlp":
        gate["bias"] = bias
      gate["kernel"] = gate_kernel
      if q_rank and part != "mlp":
        gate["bias"] = get("mlp.gate.e_score_correction_bias", True)
      routed = {"gate": gate}
      if include_experts:
        expert_values = mtp_expert_values if scanned and index == params["num_layers"] else {
            name: [] for name, _ in projections
        }
        for expert in range(params["num_experts"]):
          for name, source in projections:
            expert_values[name].append(get(f"mlp.experts.{expert}.{source}.weight", True))
        for name, _ in projections:
          routed[name] = np.stack(expert_values[name])
      result["DeepSeekMoeBlock_0"] = {
          "MoeBlock_0": routed,
          "shared_experts": shared,
      }
    return result

  weights = {"decoder": {
      "decoder_norm": {"scale": value("model.norm.weight")},
      "logits_dense": {"kernel": value("lm_head.weight", True)},
  }}
  weights["token_embedder"] = {"embedding": value("model.embed_tokens.weight")}
  decoder = weights["decoder"]
  dense_count = params["first_num_dense_layers"]
  groups = (("dense_layers", 0, dense_count), ("moe_layers", dense_count, params["num_layers"]))
  for kind, start, end in groups:
    for index in range(start, end):
      decoder[f"{kind}_{index - start}"] = layer(index, kind == "dense_layers")
  del tensors
  gc.collect()
  return weights


def convert(base_model_path: str, maxtext_model_path: str, model_size: str, simulated_cpu_devices_count: int = 16,
            use_ocdbt: bool = True, use_zarr3: bool = True, enable_mtp: bool = False,
            weight_dtype: str = "float16", model_params: dict | None = None) -> None:
  if model_params is None and model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{model_size}' is not supported.")
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, simulated_cpu_devices_count,
               use_ocdbt, use_zarr3, enable_mtp, weight_dtype, model_params)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.standalone_scripts "
          "import convert_deepseek_family_ckpt as converter; "
          "converter._run_cpu_worker(result_fd)"
      )],
      input=pickle.dumps(arguments), stdout=subprocess.PIPE, env=environment, check=True,
  )
  error = pickle.loads(child.stdout)
  if error is not None:
    raise error


def _run_cpu_worker(result_fd):
  try:
    _convert_on_cpu(*pickle.load(sys.stdin.buffer))
    error = None
  except Exception as exc:
    error = exc
  with os.fdopen(result_fd, "wb") as result_file:
    pickle.dump(error, result_file)


def _convert_on_cpu(base_model_path, maxtext_model_path, model_size, simulated_cpu_devices_count,
                    use_ocdbt, use_zarr3, enable_mtp, weight_dtype, model_params):
  weights = convert_to_jax_weights(base_model_path, model_size, model_params, enable_mtp, weight_dtype)
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
