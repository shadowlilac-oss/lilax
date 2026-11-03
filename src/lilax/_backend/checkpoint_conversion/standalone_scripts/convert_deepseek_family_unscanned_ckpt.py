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

from lilax._backend.checkpoint_conversion.standalone_scripts import convert_deepseek_family_ckpt as ds_ckpt
from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint


MODEL_PARAMS_DICT = ds_ckpt.MODEL_PARAMS_DICT


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info) -> dict:
  num_layers = model_params["num_layers"]
  dense_layers = model_params["first_num_dense_layers"]
  num_heads = model_params["base_num_query_heads"]
  emb_dim = model_params["base_emb_dim"]
  num_experts = model_params["num_experts"]
  q_rank = model_params["q_lora_rank"]
  kv_rank = model_params["kv_lora_rank"]
  nope_dim = model_params["qk_nope_head_dim"]
  rope_dim = model_params["qk_rope_head_dim"]
  v_dim = model_params["v_head_dim"]
  is_compressed = bool(model_params.get("compressed_int4", False))
  hf_key_prefix = model_params.get("hf_key_prefix", "")
  chkpt_vars = {}

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  for i, ckpt_path in enumerate(ckpt_paths):
    logging.info("Loading checkpoint %s of %s ...", i + 1, len(ckpt_paths))
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for raw_key in f.keys():
        if hf_key_prefix and not raw_key.startswith(hf_key_prefix):
          continue
        key = raw_key[len(hf_key_prefix) :]
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if is_compressed and key.endswith((".weight_scale", ".weight_shape")):
          continue
        is_packed = is_compressed and key.endswith(".weight_packed")
        hf_key = key[: -len(".weight_packed")] + ".weight" if is_packed else key
        if not ds_ckpt.is_key_allowed(hf_key, ds_ckpt.MTP_KEYS_TO_SKIP):
          continue
        layer = int(hf_key.split(".")[2]) if "layers" in hf_key else 0
        mapped_key = ds_ckpt.hf_to_maxtext_mapping(layer, num_experts, dense_layers, num_layers).get(hf_key)
        if not mapped_key:
          logging.info("Debug: Allowed key '%s' (layer %s) has no mapping in hf_to_maxtext_mapping.", hf_key, layer)
          continue
        if is_packed:
          raw_base = raw_key[: -len(".weight_packed")]
          chkpt_vars[mapped_key] = ds_ckpt.dequantize_pack_quantized_int4(
              f.get_tensor(raw_key), f.get_tensor(raw_base + ".weight_scale"),
              f.get_tensor(raw_base + ".weight_shape").tolist(),
          )
        else:
          chkpt_vars[mapped_key] = f.get_tensor(raw_key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  def array(key, transpose=False):
    value = chkpt_vars[key].to(torch.float16).numpy()
    return value.transpose() if transpose else value

  def transformer(prefix, is_dense):
    attention_prefix = f"{prefix}.self_attention"
    attention = {
        "kv_norm": {"scale": array(f"{attention_prefix}.kv_norm.scale", True)},
        "wkv_a": {"kernel": array(f"{attention_prefix}.wkv_a.kernel", True)},
        "wkv_b": {"kernel": array(f"{attention_prefix}.wkv_b.kernel", True).reshape(
            kv_rank, num_heads, nope_dim + v_dim)},
        "out": {"kernel": array(f"{attention_prefix}.out.kernel", True).reshape(num_heads, v_dim, emb_dim)},
    }
    if q_rank != 0:
      attention.update({
          "q_norm": {"scale": array(f"{attention_prefix}.q_norm.scale")},
          "wq_a": {"kernel": array(f"{attention_prefix}.wq_a.kernel", True)},
          "wq_b": {"kernel": array(f"{attention_prefix}.wq_b.kernel", True).reshape(
              q_rank, num_heads, nope_dim + rope_dim)},
      })
    else:
      attention["query"] = {"kernel": array(f"{attention_prefix}.query.kernel", True).reshape(
          emb_dim, num_heads, nope_dim + rope_dim)}
    result = {
        "self_attention": attention,
        "pre_self_attention_layer_norm": {"scale": array(f"{prefix}.pre_self_attention_layer_norm.scale")},
        "post_self_attention_layer_norm": {"scale": array(f"{prefix}.post_self_attention_layer_norm.scale")},
    }
    if is_dense:
      result["mlp"] = {name: {"kernel": array(f"{prefix}.mlp.{name}.kernel", True)}
                       for name in ("wi_0", "wi_1", "wo")}
    else:
      moe_prefix = f"{prefix}.DeepSeekMoeBlock_0"
      gate = {"kernel": array(f"{moe_prefix}.MoeBlock_0.gate.kernel", True)}
      if q_rank != 0:
        gate["bias"] = array(f"{moe_prefix}.MoeBlock_0.gate.bias", True)
      experts = {"gate": gate, "wi_0": None, "wi_1": None, "wo": None}
      for k in tqdm(range(num_experts), desc="experts", leave=False):
        for name in ("wi_0", "wi_1", "wo"):
          weight = array(f"{moe_prefix}.MoeBlock_0.{k}.{name}", True)
          if experts[name] is None:
            experts[name] = np.zeros((num_experts,) + weight.shape, dtype=np.float16)
          experts[name][k, ...] = weight
      result["DeepSeekMoeBlock_0"] = {
          "MoeBlock_0": experts,
          "shared_experts": {name: {"kernel": array(f"{moe_prefix}.shared_experts.{name}.kernel", True)}
                             for name in ("wi_0", "wi_1", "wo")},
      }
    return result

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": array("decoder_norm.scale")},
          "logits_dense": {"kernel": array("logits_dense.kernel", True)},
      },
      "token_embedder": {"embedding": array("token_embedder.embedding")},
  }
  layers = {"dense_layers": dense_layers, "moe_layers": num_layers - dense_layers}
  for layer_key, layer_count in layers.items():
    for layer_idx in tqdm(range(layer_count), desc=layer_key, leave=False):
      jax_weights["decoder"][f"{layer_key}_{layer_idx}"] = transformer(
          f"{layer_key}.{layer_idx}", layer_key == "dense_layers"
      )

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict:
  params = MODEL_PARAMS_DICT[model_size] if model_params is None else model_params
  return _convert_huggingface_to_jax_weights(base_model_path, params, psutil.Process())


def convert(base_model_path: str, maxtext_model_path: str, model_size: str, simulated_cpu_devices_count: int = 16,
            use_ocdbt: bool = True, use_zarr3: bool = True, model_params: dict | None = None) -> None:
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, simulated_cpu_devices_count,
               use_ocdbt, use_zarr3, model_params)
  subprocess.run(
      [sys.executable, "-c", (
          "import pickle, sys; "
          "from lilax._backend.checkpoint_conversion.standalone_scripts "
          "import convert_deepseek_family_unscanned_ckpt as converter; "
          "converter._convert_on_cpu(*pickle.load(sys.stdin.buffer))"
      )],
      input=pickle.dumps(arguments), env=environment, check=True,
  )


def _convert_on_cpu(base_model_path, maxtext_model_path, model_size, simulated_cpu_devices_count,
                    use_ocdbt, use_zarr3, model_params):
  weights = convert_to_jax_weights(base_model_path, model_size, model_params)
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
