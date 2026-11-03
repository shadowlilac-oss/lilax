import gc
import logging
import pathlib

import ml_dtypes
import numpy as np
import psutil
import torch
from safetensors import safe_open
from tqdm import tqdm

from lilax._backend.checkpoint_conversion.checkpoint_cpu_bootstrap import run_on_cpu
from lilax._backend.checkpoint_conversion.standalone_scripts.convert_qwen3_next_scanned import MODEL_PARAMS_DICT
from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from lilax._backend.utils import max_logging


CAST_DTYPE = ml_dtypes.bfloat16


def _pt_to_np(pt_weight, cast_dtype=None, transpose=False):
  if cast_dtype:
    np_weight = pt_weight.to(torch.float32).numpy().astype(cast_dtype)
  else:
    np_weight = pt_weight.to(torch.float32).numpy()
  if transpose:
    np_weight = np_weight.transpose()
  return np_weight


def create_unscanned_layer_pytree(layer_idx):
  layer = {
      "input_layernorm": {"scale": None},
      "post_attention_layernorm": {"scale": None},
      "mlp": {
          "shared_expert": {"wi_0": {"kernel": None}, "wi_1": {"kernel": None}, "wo": {"kernel": None}},
          "shared_expert_gate": {"kernel": None},
          "routed_experts": {"gate": {"kernel": None}, "wi_0": None, "wi_1": None, "wo": None},
      },
  }
  if layer_idx % 4 == 3:
    layer["attention"] = {"attention": {
        "query": {"kernel": None}, "key": {"kernel": None}, "value": {"kernel": None},
        "out": {"kernel": None}, "query_norm": {"scale": None}, "key_norm": {"scale": None},
    }}
  else:
    layer["attention"] = {
        "A_log": None, "conv1d": {"kernel": None}, "dt_bias": None, "in_proj_ba": {"kernel": None},
        "in_proj_qkvz": {"kernel": None}, "norm": {"rms_norm": {"scale": None}}, "out_proj": {"kernel": None},
    }
  return layer


def convert_hf_to_maxtext(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  num_layers = model_params["num_hidden_layers"]
  hidden_size = model_params["hidden_size"]
  num_experts = model_params["num_experts"]
  num_q_heads = model_params["ga_num_q_heads"]
  num_kv_heads = model_params["ga_num_kv_heads"]
  head_dim = model_params["head_dim"]
  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  chkpt_vars = {}
  for i, path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
      for key in checkpoint.keys():
        if key.startswith("model.") or key.startswith("lm_head."):
          chkpt_vars[key] = checkpoint.get_tensor(key)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  def weight(key):
    return _pt_to_np(chkpt_vars[key], cast_dtype=CAST_DTYPE)

  weights = {
      "token_embedder": {"embedding": None},
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
  }
  for layer in range(num_layers):
    weights["decoder"][f"layers_{layer}"] = create_unscanned_layer_pytree(layer)
  max_logging.log("Populating non-layer weights...")
  weights["decoder"]["decoder_norm"]["scale"] = weight("model.norm.weight")
  weights["token_embedder"]["embedding"] = weight("model.embed_tokens.weight")
  weights["decoder"]["logits_dense"]["kernel"] = weight("lm_head.weight").transpose()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log("Processing linear & gated attention layers")
  for layer in tqdm(range(num_layers), desc="layers", leave=False):
    prefix = f"model.layers.{layer}"
    attention = weights["decoder"][f"layers_{layer}"]["attention"]
    if layer % 4 == 3:
      attn = attention["attention"]
      attn["key"]["kernel"] = weight(f"{prefix}.self_attn.k_proj.weight").transpose().reshape(
          hidden_size, num_kv_heads, head_dim
      )
      attn["key_norm"]["scale"] = weight(f"{prefix}.self_attn.k_norm.weight")
      attn["out"]["kernel"] = weight(f"{prefix}.self_attn.o_proj.weight").transpose()
      attn["query"]["kernel"] = weight(f"{prefix}.self_attn.q_proj.weight").transpose().reshape(
          hidden_size, num_q_heads, head_dim * 2
      )
      attn["query_norm"]["scale"] = weight(f"{prefix}.self_attn.q_norm.weight")
      attn["value"]["kernel"] = weight(f"{prefix}.self_attn.v_proj.weight").transpose().reshape(
          hidden_size, num_kv_heads, head_dim
      )
    else:
      attn_prefix = f"{prefix}.linear_attn"
      attention["A_log"] = weight(f"{attn_prefix}.A_log")
      attention["conv1d"]["kernel"] = weight(f"{attn_prefix}.conv1d.weight").transpose(2, 1, 0)
      attention["dt_bias"] = weight(f"{attn_prefix}.dt_bias")
      attention["in_proj_ba"]["kernel"] = weight(f"{attn_prefix}.in_proj_ba.weight").transpose()
      attention["in_proj_qkvz"]["kernel"] = weight(f"{attn_prefix}.in_proj_qkvz.weight").transpose()
      attention["norm"]["rms_norm"]["scale"] = weight(f"{attn_prefix}.norm.weight")
      attention["out_proj"]["kernel"] = weight(f"{attn_prefix}.out_proj.weight").transpose()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log("Processing pre and post self attention norms")
  for layer in tqdm(range(num_layers), desc="layers", leave=False):
    prefix = f"model.layers.{layer}"
    layer_weights = weights["decoder"][f"layers_{layer}"]
    layer_weights["input_layernorm"]["scale"] = weight(f"{prefix}.input_layernorm.weight")
    layer_weights["post_attention_layernorm"]["scale"] = weight(f"{prefix}.post_attention_layernorm.weight")
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log("Processing mlp layer weights")
  for layer in tqdm(range(num_layers), desc="layers", leave=False):
    prefix = f"model.layers.{layer}"
    mlp = weights["decoder"][f"layers_{layer}"]["mlp"]
    for name, projection in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
      mlp["shared_expert"][name]["kernel"] = weight(f"{prefix}.mlp.shared_expert.{projection}.weight").transpose()
    mlp["shared_expert_gate"]["kernel"] = weight(f"{prefix}.mlp.shared_expert_gate.weight").transpose()
    wi_0, wi_1, wo = [], [], []
    mlp["routed_experts"]["gate"]["kernel"] = weight(f"{prefix}.mlp.gate.weight").transpose()
    for i in range(num_experts):
      wi_0.append(weight(f"{prefix}.mlp.experts.{i}.gate_proj.weight").transpose())
      wi_1.append(weight(f"{prefix}.mlp.experts.{i}.up_proj.weight").transpose())
      wo.append(weight(f"{prefix}.mlp.experts.{i}.down_proj.weight").transpose())
    mlp["routed_experts"]["wi_0"] = np.stack(wi_0, axis=0)
    mlp["routed_experts"]["wi_1"] = np.stack(wi_1, axis=0)
    mlp["routed_experts"]["wo"] = np.stack(wo, axis=0)
    gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  del chkpt_vars
  gc.collect()
  return weights


def convert_to_jax_weights(base_model_path: str, model_size: str):
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return convert_hf_to_maxtext(base_model_path, model_size, model_params, mem_info)


def convert(base_model_path: str, maxtext_model_path: str, model_size: str,
            simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True):
  if model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{model_size}' is not supported.")
  return run_on_cpu(
      "convert_qwen3_next_unscanned", simulated_cpu_devices_count=simulated_cpu_devices_count,
      base_model_path=base_model_path, maxtext_model_path=maxtext_model_path, model_size=model_size,
      use_ocdbt=use_ocdbt, use_zarr3=use_zarr3,
  )


def _convert(base_model_path: str, maxtext_model_path: str, model_size: str,
             simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True):
  weights = convert_to_jax_weights(base_model_path, model_size)
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
  max_logging.log(f"Successfully saved base_weights to {maxtext_model_path}.")
