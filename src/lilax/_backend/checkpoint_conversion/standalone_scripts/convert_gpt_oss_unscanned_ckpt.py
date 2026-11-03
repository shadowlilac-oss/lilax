import os
import pathlib
import pickle
import subprocess
import sys

import ml_dtypes
import torch
from safetensors import safe_open

from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint


MODEL_PARAMS_DICT = {
    "gpt-oss-20b": {
        "base_emb_dim": 2880, "base_num_query_heads": 64, "base_num_kv_heads": 8,
        "head_dim": 64, "base_num_decoder_layers": 24, "inhomogeneous_layer_cycle_interval": 2,
    },
    "gpt-oss-120b": {
        "base_emb_dim": 2880, "base_num_query_heads": 64, "base_num_kv_heads": 8,
        "head_dim": 64, "base_num_decoder_layers": 36, "inhomogeneous_layer_cycle_interval": 2,
    },
}


def _pt_to_np(weight):
  return weight.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)


def _hf_to_maxtext_mapping(layer_idx: int = -1) -> dict:
  return {
      "model.embed_tokens.weight": "tok_embeddings.weight",
      "model.norm.weight": "norm.weight",
      "lm_head.weight": "output.weight",
      f"model.layers.{layer_idx}.input_layernorm.weight": f"layers.{layer_idx}.attention_norm.weight",
      f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"layers.{layer_idx}.ffn_norm.weight",
      f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"layers.{layer_idx}.attention.wq.weight",
      f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"layers.{layer_idx}.attention.wk.weight",
      f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"layers.{layer_idx}.attention.wv.weight",
      f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"layers.{layer_idx}.attention.wo.weight",
      f"model.layers.{layer_idx}.self_attn.q_proj.bias": f"layers.{layer_idx}.attention.wq.bias",
      f"model.layers.{layer_idx}.self_attn.k_proj.bias": f"layers.{layer_idx}.attention.wk.bias",
      f"model.layers.{layer_idx}.self_attn.v_proj.bias": f"layers.{layer_idx}.attention.wv.bias",
      f"model.layers.{layer_idx}.self_attn.o_proj.bias": f"layers.{layer_idx}.attention.wo.bias",
      f"model.layers.{layer_idx}.self_attn.sinks": f"layers.{layer_idx}.attention.sinks",
      f"model.layers.{layer_idx}.mlp.router.weight": f"layers.{layer_idx}.feed_forward.gate.weight",
      f"model.layers.{layer_idx}.mlp.router.bias": f"layers.{layer_idx}.feed_forward.gate.bias",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj": f"layers.{layer_idx}.feed_forward.experts.gate_up_proj",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias": f"layers.{layer_idx}.feed_forward.experts.gate_up_proj_bias",
      f"model.layers.{layer_idx}.mlp.experts.down_proj": f"layers.{layer_idx}.feed_forward.experts.down_proj",
      f"model.layers.{layer_idx}.mlp.experts.down_proj_bias": f"layers.{layer_idx}.feed_forward.experts.down_proj_bias",
  }


def convert_to_jax_weights(base_model_path: str, model_size: str, model_params: dict | None = None) -> dict:
  params = MODEL_PARAMS_DICT[model_size] if model_params is None else model_params
  tensors = {}
  for path in sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as source:
      for key in source.keys():
        layer = int(key.split(".")[2]) if "layers" in key else 0
        _hf_to_maxtext_mapping(layer)[key]
        tensors[key] = source.get_tensor(key)

  weights = {
      "token_embedder": {"embedding": _pt_to_np(tensors["model.embed_tokens.weight"])},
      "decoder": {
          "decoder_norm": {"scale": _pt_to_np(tensors["model.norm.weight"])},
          "logits_dense": {"kernel": _pt_to_np(tensors["lm_head.weight"]).transpose()},
      },
  }
  emb = params["base_emb_dim"]
  q_heads = params["base_num_query_heads"]
  kv_heads = params["base_num_kv_heads"]
  head_dim = params["head_dim"]
  for index in range(params["base_num_decoder_layers"]):
    prefix = f"model.layers.{index}."
    def value(name):
      return _pt_to_np(tensors[prefix + name])

    attention = {}
    for name, source, heads in (("query", "q_proj", q_heads), ("key", "k_proj", kv_heads),
                                ("value", "v_proj", kv_heads)):
      attention[name] = {
          "kernel": value(f"self_attn.{source}.weight").transpose().reshape(emb, heads, head_dim),
          "bias": value(f"self_attn.{source}.bias").reshape(heads, head_dim),
      }
    attention["out"] = {
        "kernel": value("self_attn.o_proj.weight").transpose().reshape(q_heads, head_dim, emb),
        "bias": value("self_attn.o_proj.bias"),
    }
    attention["sinks"] = value("self_attn.sinks")
    gate_up = value("mlp.experts.gate_up_proj")
    gate_up_bias = value("mlp.experts.gate_up_proj_bias")
    weights["decoder"][f"layers_{index}"] = {
        "pre_self_attention_layer_norm": {"scale": value("input_layernorm.weight")},
        "post_self_attention_layer_norm": {"scale": value("post_attention_layernorm.weight")},
        "GptOssAttention": attention,
        "GptOssMlp": {
            "gate": {"kernel": value("mlp.router.weight").transpose(), "bias": value("mlp.router.bias")},
            "wi_0": gate_up[..., ::2], "wi_1": gate_up[..., 1::2],
            "wi_0_bias": gate_up_bias[..., ::2], "wi_1_bias": gate_up_bias[..., 1::2],
            "wo": value("mlp.experts.down_proj"), "wo_bias": value("mlp.experts.down_proj_bias"),
        },
    }
  return weights


def convert(base_model_path: str, maxtext_model_path: str, model_size: str, simulated_cpu_devices_count: int = 16,
            use_ocdbt: bool = True, use_zarr3: bool = True, model_params: dict | None = None) -> None:
  _convert_on_cpu("convert_gpt_oss_unscanned_ckpt", base_model_path, maxtext_model_path, model_size,
                  simulated_cpu_devices_count, use_ocdbt, use_zarr3, model_params)


def _convert_on_cpu(module, base_model_path, maxtext_model_path, model_size, simulated_cpu_devices_count,
                    use_ocdbt, use_zarr3, model_params):
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, simulated_cpu_devices_count,
               use_ocdbt, use_zarr3, model_params)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          f"from lilax._backend.checkpoint_conversion.standalone_scripts import {module} as converter; "
          "converter._run_cpu_worker(result_fd)"
      )],
      input=pickle.dumps(arguments), stdout=subprocess.PIPE, env=environment, check=True,
  )
  error = pickle.loads(child.stdout)
  if error is not None:
    raise error


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
