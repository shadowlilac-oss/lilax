import gc
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm


MODEL_PARAMS_DICT = {
    "qwen3-235b-a22b": {
        "num_hidden_layers": 94, "num_attention_heads": 64, "num_key_value_heads": 4,
        "hidden_size": 4096, "head_dim": 128, "num_experts": 128, "moe_intermediate_size": 1536,
    },
    "qwen3-30b-a3b": {
        "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
        "hidden_size": 2048, "head_dim": 128, "num_experts": 128, "moe_intermediate_size": 768,
    },
    "qwen3-480b-a35b": {
        "num_hidden_layers": 62, "num_attention_heads": 96, "num_key_value_heads": 8,
        "hidden_size": 6144, "head_dim": 128, "num_experts": 160, "moe_intermediate_size": 2560,
    },
}


def hf_to_maxtext_mapping(layer_idx: int, num_experts: int) -> dict:
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "model.norm.weight": "decoder.decoder_norm.scale",
      "lm_head.weight": "decoder.logits_dense.kernel",
      f"model.layers.{layer_idx}.input_layernorm.weight": f"decoder.layers.{layer_idx}.pre_self_attention_layer_norm.scale",
      f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"decoder.layers.{layer_idx}.post_self_attention_layer_norm.scale",
      f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"decoder.layers.{layer_idx}.self_attention.query.kernel",
      f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"decoder.layers.{layer_idx}.self_attention.key.kernel",
      f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"decoder.layers.{layer_idx}.self_attention.value.kernel",
      f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"decoder.layers.{layer_idx}.self_attention.out.kernel",
      f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"decoder.layers.{layer_idx}.self_attention.query_norm.scale",
      f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"decoder.layers.{layer_idx}.self_attention.key_norm.scale",
      f"model.layers.{layer_idx}.mlp.gate.weight": f"decoder.layers.{layer_idx}.moe_block.gate.kernel",
  }
  for i in range(num_experts):
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.{i}.wi_0"
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.{i}.wi_1"
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.{i}.wo"
  return mapping


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
  num_layers = model_params["num_hidden_layers"]
  num_experts = model_params["num_experts"]
  hidden_size = model_params["hidden_size"]
  num_heads = model_params["num_attention_heads"]
  num_kv_heads = model_params["num_key_value_heads"]
  head_dim = model_params["head_dim"]
  moe_intermediate_size = model_params["moe_intermediate_size"]

  chkpt_vars = {}
  for ckpt_path in sorted(pathlib.Path(base_model_path).glob("*.safetensors")):
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        if "layers" not in key and "embed_tokens" not in key and "norm" not in key and "lm_head" not in key:
          continue
        layer_idx_str = key.split(".")[2] if "layers" in key else "0"
        layer_idx = int(layer_idx_str) if layer_idx_str.isdigit() else 0
        maxtext_key = hf_to_maxtext_mapping(layer_idx, num_experts).get(key)
        if maxtext_key:
          chkpt_vars[maxtext_key] = f.get_tensor(key)

  ln = {
      "pre_self_attention_layer_norm": {"scale": np.zeros((num_layers, hidden_size), dtype=np.float16)},
      "post_self_attention_layer_norm": {"scale": np.zeros((num_layers, hidden_size), dtype=np.float16)},
      "self_attention": {
          "query": {"kernel": np.zeros((num_layers, hidden_size, num_heads, head_dim), dtype=np.float16)},
          "key": {"kernel": np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)},
          "value": {"kernel": np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)},
          "out": {"kernel": np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=np.float16)},
          "query_norm": {"scale": np.zeros((num_layers, head_dim), dtype=np.float16)},
          "key_norm": {"scale": np.zeros((num_layers, head_dim), dtype=np.float16)},
      },
      "moe_block": {
          "gate": {"kernel": np.zeros((num_layers, hidden_size, num_experts), dtype=np.float16)},
          "wi_0": np.zeros((num_experts, num_layers, hidden_size, moe_intermediate_size), dtype=np.float16),
          "wi_1": np.zeros((num_experts, num_layers, hidden_size, moe_intermediate_size), dtype=np.float16),
          "wo": np.zeros((num_experts, num_layers, moe_intermediate_size, hidden_size), dtype=np.float16),
      },
  }
  maxtext_weights = {
      "decoder": {
          "layers": ln,
          "decoder_norm": {"scale": chkpt_vars["decoder.decoder_norm.scale"].to(torch.float16).numpy()},
          "logits_dense": {"kernel": chkpt_vars["decoder.logits_dense.kernel"].to(torch.float16).numpy().transpose()},
      },
      "token_embedder": {"embedding": chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy()},
  }
  s_attn = ln["self_attention"]
  moe = ln["moe_block"]
  for layer in tqdm(range(num_layers), desc="Stacking layer weights"):
    prefix = f"decoder.layers.{layer}"
    for name in ("pre_self_attention_layer_norm", "post_self_attention_layer_norm"):
      ln[name]["scale"][layer, :] = chkpt_vars[f"{prefix}.{name}.scale"].to(torch.float16).numpy()
    for name, shape in (
        ("query", (hidden_size, num_heads, head_dim)),
        ("key", (hidden_size, num_kv_heads, head_dim)),
        ("value", (hidden_size, num_kv_heads, head_dim)),
        ("out", (num_heads, head_dim, hidden_size)),
    ):
      s_attn[name]["kernel"][layer, ...] = (
          chkpt_vars[f"{prefix}.self_attention.{name}.kernel"].to(torch.float16).numpy().transpose().reshape(shape)
      )
    for name in ("query_norm", "key_norm"):
      s_attn[name]["scale"][layer, ...] = chkpt_vars[f"{prefix}.self_attention.{name}.scale"].to(torch.float16).numpy()
    moe["gate"]["kernel"][layer, ...] = chkpt_vars[f"{prefix}.moe_block.gate.kernel"].to(torch.float16).numpy().transpose()
    for expert in range(num_experts):
      for name in ("wi_0", "wi_1", "wo"):
        moe[name][expert, layer, ...] = (
            chkpt_vars[f"{prefix}.moe_block.{expert}.{name}"].to(torch.float16).numpy().transpose()
        )

  for name in ("pre_self_attention_layer_norm", "post_self_attention_layer_norm"):
    ln[name]["scale"] = np.transpose(ln[name]["scale"], axes=(1, 0))
  for name in ("query_norm", "key_norm"):
    s_attn[name]["scale"] = np.transpose(s_attn[name]["scale"], axes=(1, 0))
  for name in ("query", "key", "value", "out"):
    s_attn[name]["kernel"] = np.transpose(s_attn[name]["kernel"], axes=(1, 0, 2, 3))
  moe["gate"]["kernel"] = np.transpose(moe["gate"]["kernel"], axes=(1, 0, 2))
  gc.collect()
  return maxtext_weights


def convert(base_model_path: str, maxtext_model_path: str, model_size: str,
            simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True):
  if model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{model_size}' not found in MODEL_PARAMS_DICT.")
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, simulated_cpu_devices_count,
               use_ocdbt, use_zarr3)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.standalone_scripts.qwen3_moe_cpu_worker import run; "
          "run(result_fd)"
      )],
      input=pickle.dumps(arguments), stdout=subprocess.PIPE, env=environment, check=True,
  )
  error = pickle.loads(child.stdout)
  if error is not None:
    raise error
