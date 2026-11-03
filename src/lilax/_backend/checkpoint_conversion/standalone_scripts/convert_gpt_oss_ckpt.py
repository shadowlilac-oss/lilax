import gc
import os
import pathlib
import pickle
import sys

import ml_dtypes
import numpy as np
from safetensors import safe_open

from lilax._backend.checkpoint_conversion.standalone_scripts import convert_gpt_oss_unscanned_ckpt
from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint


MODEL_PARAMS_DICT = convert_gpt_oss_unscanned_ckpt.MODEL_PARAMS_DICT


def convert_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict:
  params = MODEL_PARAMS_DICT[model_size] if model_params is None else model_params
  tensors = {}
  for path in sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as source:
      for key in source.keys():
        layer = int(key.split(".")[2]) if "layers" in key else 0
        convert_gpt_oss_unscanned_ckpt._hf_to_maxtext_mapping(layer)[key]
        tensors[key] = source.get_tensor(key)

  value = convert_gpt_oss_unscanned_ckpt._pt_to_np
  weights = {
      "token_embedder": {"embedding": value(tensors["model.embed_tokens.weight"])},
      "decoder": {
          "decoder_norm": {"scale": value(tensors["model.norm.weight"])},
          "logits_dense": {"kernel": value(tensors["lm_head.weight"]).transpose()},
          "layers": {},
      },
  }
  layers = weights["decoder"]["layers"]
  interval = params["inhomogeneous_layer_cycle_interval"]
  count = params["base_num_decoder_layers"]
  emb = params["base_emb_dim"]
  q_heads = params["base_num_query_heads"]
  kv_heads = params["base_num_kv_heads"]
  head_dim = params["head_dim"]
  for block in range(interval):
    layers[f"layers_{block}"] = {"GptOssAttention": {}, "GptOssMlp": {}}

  def store(destination, key, item, index):
    if key not in destination:
      destination[key] = np.zeros((count // interval,) + item.shape, dtype=ml_dtypes.bfloat16)
    destination[key][index] = item

  for index in range(count):
    position, block = divmod(index, interval)
    attention = layers[f"layers_{block}"]["GptOssAttention"]
    prefix = f"model.layers.{index}.self_attn."
    for name, source, heads in (("query", "q_proj", q_heads), ("key", "k_proj", kv_heads),
                                ("value", "v_proj", kv_heads)):
      target = attention.setdefault(name, {})
      kernel = value(tensors[prefix + source + ".weight"]).transpose().reshape(emb, heads, head_dim)
      bias = value(tensors[prefix + source + ".bias"]).reshape(heads, head_dim)
      store(target, "kernel", kernel, position)
      store(target, "bias", bias, position)
    target = attention.setdefault("out", {})
    kernel = value(tensors[prefix + "o_proj.weight"]).transpose().reshape(q_heads, head_dim, emb)
    store(target, "kernel", kernel, position)
    store(target, "bias", value(tensors[prefix + "o_proj.bias"]), position)
    store(attention, "sinks", value(tensors[prefix + "sinks"]), position)

  for index in range(count):
    position, block = divmod(index, interval)
    target = layers[f"layers_{block}"]
    prefix = f"model.layers.{index}."
    for name, source in (("pre_self_attention_layer_norm", "input_layernorm.weight"),
                         ("post_self_attention_layer_norm", "post_attention_layernorm.weight")):
      store(target.setdefault(name, {}), "scale", value(tensors[prefix + source]), position)

  for index in range(count):
    position, block = divmod(index, interval)
    mlp = layers[f"layers_{block}"]["GptOssMlp"]
    prefix = f"model.layers.{index}.mlp."
    gate = mlp.setdefault("gate", {})
    store(gate, "kernel", value(tensors[prefix + "router.weight"]).transpose(), position)
    store(gate, "bias", value(tensors[prefix + "router.bias"]), position)
    gate_up = value(tensors[prefix + "experts.gate_up_proj"])
    gate_up_bias = value(tensors[prefix + "experts.gate_up_proj_bias"])
    store(mlp, "wi_0", gate_up[..., ::2], position)
    store(mlp, "wi_1", gate_up[..., 1::2], position)
    store(mlp, "wi_0_bias", gate_up_bias[..., ::2], position)
    store(mlp, "wi_1_bias", gate_up_bias[..., 1::2], position)
    store(mlp, "wo", value(tensors[prefix + "experts.down_proj"]), position)
    store(mlp, "wo_bias", value(tensors[prefix + "experts.down_proj_bias"]), position)
    gc.collect()

  def reorder(values):
    for key, item in values.items():
      if isinstance(item, dict):
        reorder(item)
      else:
        values[key] = item.swapaxes(0, 1)

  reorder(layers)
  del tensors
  gc.collect()
  return weights


def convert(base_model_path: str, maxtext_model_path: str, model_size: str, simulated_cpu_devices_count: int = 16,
            use_ocdbt: bool = True, use_zarr3: bool = True, model_params: dict | None = None) -> None:
  convert_gpt_oss_unscanned_ckpt._convert_on_cpu(
      "convert_gpt_oss_ckpt", base_model_path, maxtext_model_path, model_size, simulated_cpu_devices_count,
      use_ocdbt, use_zarr3, model_params,
  )


def _run_cpu_worker(result_fd):
  try:
    _convert(*pickle.load(sys.stdin.buffer))
    error = None
  except Exception as exc:
    error = exc
  with os.fdopen(result_fd, "wb") as result_file:
    pickle.dump(error, result_file)


def _convert(base_model_path, maxtext_model_path, model_size, simulated_cpu_devices_count,
             use_ocdbt, use_zarr3, model_params):
  weights = convert_to_jax_weights(base_model_path, model_size, model_params)
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
