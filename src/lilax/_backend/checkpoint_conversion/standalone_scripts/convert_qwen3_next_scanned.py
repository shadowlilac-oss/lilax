import argparse
import gc
import os
import pathlib
import pickle
import subprocess
import sys
from functools import partial

import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from lilax._backend.utils import max_logging


MODEL_PARAMS_DICT = {
    "qwen3-next-80b-a3b": {
        "num_hidden_layers": 48, "hidden_size": 2048, "num_experts": 512, "moe_intermediate_size": 512,
        "head_dim": 256, "ga_num_q_heads": 16, "ga_num_kv_heads": 2, "ga_o_proj_input_dim": 4096,
        "gdn_num_value_heads": 32, "gdn_num_key_heads": 16, "inhomogeneous_layer_cycle_interval": 4,
        "gdn_conv_kernel_dim": 4, "gdn_key_head_dim": 128, "gdn_value_head_dim": 128,
        "gdn_a_log_dim": 32, "gdn_conv_features": 8192, "gdn_norm_dim": 128, "gdn_out_proj_input_dim": 4096,
    },
}


CAST_DTYPE = ml_dtypes.bfloat16


def to_np_bfloat16(tensor):
  return tensor.to(torch.float32).numpy().astype(CAST_DTYPE)


def hf_to_maxtext_mapping(layer_idx: int, num_experts: int, inhomogeneous_layer_cycle_interval: int) -> dict:
  block_idx = layer_idx % inhomogeneous_layer_cycle_interval
  hf_prefix = f"model.layers.{layer_idx}"
  mt_prefix = f"decoder.layers.layer_{block_idx}"
  attn = f"{mt_prefix}.attention"
  mlp = f"{mt_prefix}.mlp"
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "model.norm.weight": "decoder.decoder_norm.scale",
      "lm_head.weight": "decoder.logits_dense.kernel",
      f"{hf_prefix}.input_layernorm.weight": f"{mt_prefix}.input_layernorm.scale",
      f"{hf_prefix}.post_attention_layernorm.weight": f"{mt_prefix}.post_attention_layernorm.scale",
  }
  if (layer_idx + 1) % inhomogeneous_layer_cycle_interval == 0:
    for hf_name, mt_name in (("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value"), ("o_proj", "out")):
      mapping[f"{hf_prefix}.self_attn.{hf_name}.weight"] = f"{attn}.attention.{mt_name}.kernel"
    for hf_name, mt_name in (("q_norm", "query_norm"), ("k_norm", "key_norm")):
      mapping[f"{hf_prefix}.self_attn.{hf_name}.weight"] = f"{attn}.attention.{mt_name}.scale"
  else:
    for name in ("in_proj_qkvz", "in_proj_ba", "conv1d"):
      mapping[f"{hf_prefix}.linear_attn.{name}.weight"] = f"{attn}.{name}.kernel"
    for name in ("A_log", "dt_bias"):
      mapping[f"{hf_prefix}.linear_attn.{name}"] = f"{attn}.{name}"
    mapping[f"{hf_prefix}.linear_attn.norm.weight"] = f"{attn}.norm.rms_norm.scale"
    mapping[f"{hf_prefix}.linear_attn.out_proj.weight"] = f"{attn}.out_proj.kernel"
  mapping[f"{hf_prefix}.mlp.gate.weight"] = f"{mlp}.routed_experts.gate.kernel"
  for mt_name, hf_name in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
    mapping[f"{hf_prefix}.mlp.shared_expert.{hf_name}.weight"] = f"{mlp}.shared_expert.{mt_name}.kernel"
  mapping[f"{hf_prefix}.mlp.shared_expert_gate.weight"] = f"{mlp}.shared_expert_gate.kernel"
  for i in range(num_experts):
    for mt_name, hf_name in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
      mapping[f"{hf_prefix}.mlp.experts.{i}.{hf_name}.weight"] = f"{mlp}.routed_experts.{i}.{mt_name}"
  return mapping


def _get_hf_tensor(maxtext_key_suffix, hf_map, l, chkpt_vars):
  for hf_key, mt_key in hf_map.items():
    if mt_key.endswith(maxtext_key_suffix):
      if hf_key in chkpt_vars:
        return chkpt_vars[hf_key]
      raise ValueError(f"HF Key {hf_key} not found in chkpt_vars for MaxText suffix: {maxtext_key_suffix} in layer {l}")
  raise ValueError(f"Could not find HF key for MaxText suffix: {maxtext_key_suffix} in layer {l}")


def init_maxtext_weights(model_params, num_layers_to_convert, num_experts_to_convert):
  emb_dim = model_params["emb_dim"]
  num_q_heads = model_params["num_q_heads"]
  num_kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["head_dim"]
  intermediate = model_params["moe_intermediate_size"]
  cycle = model_params["inhomogeneous_layer_cycle_interval"]
  stacked = num_layers_to_convert // cycle
  value_heads = model_params["gdn_num_value_heads"]
  key_dim = model_params["gdn_num_key_heads"] * model_params["gdn_key_head_dim"]
  value_dim = value_heads * model_params["gdn_value_head_dim"]
  conv_dim = key_dim * 2 + value_dim
  zeros = lambda shape: np.zeros(shape, dtype=CAST_DTYPE)
  weights = {"decoder": {"layers": {}, "decoder_norm": {"scale": None}, "logits_dense": {"kernel": None}},
             "token_embedder": {"embedding": None}}
  for i in range(cycle):
    layer = {
        "input_layernorm": {"scale": zeros((emb_dim, stacked))},
        "post_attention_layernorm": {"scale": zeros((emb_dim, stacked))},
        "attention": {},
        "mlp": {
            "routed_experts": {
                "gate": {"kernel": zeros((emb_dim, stacked, num_experts_to_convert))},
                "wi_0": zeros((num_experts_to_convert, stacked, emb_dim, intermediate)),
                "wi_1": zeros((num_experts_to_convert, stacked, emb_dim, intermediate)),
                "wo": zeros((num_experts_to_convert, stacked, intermediate, emb_dim)),
            },
            "shared_expert": {
                "wi_0": {"kernel": zeros((emb_dim, stacked, intermediate))},
                "wi_1": {"kernel": zeros((emb_dim, stacked, intermediate))},
                "wo": {"kernel": zeros((intermediate, stacked, emb_dim))},
            },
            "shared_expert_gate": {"kernel": zeros((emb_dim, stacked, 1))},
        },
    }
    if (i + 1) % cycle == 0:
      layer["attention"] = {"attention": {
          "query": {"kernel": zeros((emb_dim, stacked, num_q_heads, head_dim * 2))},
          "key": {"kernel": zeros((emb_dim, stacked, num_kv_heads, head_dim))},
          "value": {"kernel": zeros((emb_dim, stacked, num_kv_heads, head_dim))},
          "out": {"kernel": zeros((num_q_heads * head_dim, stacked, emb_dim))},
          "query_norm": {"scale": zeros((head_dim, stacked))}, "key_norm": {"scale": zeros((head_dim, stacked))},
      }}
    else:
      layer["attention"] = {
          "in_proj_qkvz": {"kernel": zeros((emb_dim, stacked, key_dim * 2 + value_dim * 2))},
          "in_proj_ba": {"kernel": zeros((emb_dim, stacked, value_heads * 2))},
          "conv1d": {"kernel": zeros((model_params["gdn_conv_kernel_dim"], stacked, 1, conv_dim))},
          "A_log": zeros((value_heads, stacked)), "dt_bias": zeros((value_heads, stacked)),
          "norm": {"rms_norm": {"scale": zeros((model_params["gdn_value_head_dim"], stacked))}},
          "out_proj": {"kernel": zeros((value_dim, stacked, emb_dim))},
      }
    weights["decoder"]["layers"][f"layer_{i}"] = layer
  return weights


def convert_hf_to_maxtext(base_model_path: str, model_params: dict, args) -> dict:
  num_layers = model_params["num_layers"]
  num_experts = model_params["num_experts"]
  emb_dim = model_params["emb_dim"]
  num_q_heads = model_params["num_q_heads"]
  num_kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["head_dim"]
  cycle = model_params["inhomogeneous_layer_cycle_interval"]
  num_layers_to_convert = args.num_layers_to_convert if args.num_layers_to_convert > 0 else num_layers
  num_experts_to_convert = args.num_experts_to_convert if args.num_experts_to_convert > 0 else num_experts
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("model-*-of-*.safetensors"))
  chkpt_vars = {}
  max_logging.log(f"Loading {len(ckpt_paths)} checkpoint files...")
  for path in tqdm(ckpt_paths, desc="Loading HF Checkpoints"):
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
      for key in checkpoint.keys():
        chkpt_vars[key] = checkpoint.get_tensor(key)
  gc.collect()
  max_logging.log("HF weights loaded.")

  weights = init_maxtext_weights(model_params, num_layers_to_convert, num_experts_to_convert)
  max_logging.log("Populating non-layer weights...")
  if "model.embed_tokens.weight" in chkpt_vars:
    weights["token_embedder"]["embedding"] = to_np_bfloat16(chkpt_vars["model.embed_tokens.weight"])
  if "model.norm.weight" in chkpt_vars:
    weights["decoder"]["decoder_norm"]["scale"] = to_np_bfloat16(chkpt_vars["model.norm.weight"])
  if "lm_head.weight" in chkpt_vars:
    weights["decoder"]["logits_dense"]["kernel"] = to_np_bfloat16(chkpt_vars["lm_head.weight"]).transpose()

  max_logging.log(f"Populating layer weights for {num_layers_to_convert} layers...")
  for layer in tqdm(range(num_layers_to_convert), desc="Processing Layers"):
    block = layer % cycle
    stack_idx = layer // cycle
    hf_map = hf_to_maxtext_mapping(layer, model_params["num_experts"], cycle)
    get_hf_tensor = partial(_get_hf_tensor, hf_map=hf_map, l=layer, chkpt_vars=chkpt_vars)
    ln = weights["decoder"]["layers"][f"layer_{block}"]
    ln["input_layernorm"]["scale"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".input_layernorm.scale"))
    ln["post_attention_layernorm"]["scale"][:, stack_idx] = to_np_bfloat16(
        get_hf_tensor(".post_attention_layernorm.scale")
    )

    attn = ln["attention"]
    if (layer + 1) % cycle == 0:
      attn = attn["attention"]
      q_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.query.kernel")).transpose()
      attn["query"]["kernel"][:, stack_idx, :, :] = q_kernel.reshape(emb_dim, num_q_heads, head_dim * 2)
      k_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.key.kernel")).transpose()
      attn["key"]["kernel"][:, stack_idx, :, :] = k_kernel.reshape(emb_dim, num_kv_heads, head_dim)
      v_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.value.kernel")).transpose()
      attn["value"]["kernel"][:, stack_idx, :, :] = v_kernel.reshape(emb_dim, num_kv_heads, head_dim)
      attn["out"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.attention.out.kernel")
      ).transpose()
      attn["query_norm"]["scale"][:, stack_idx] = to_np_bfloat16(
          get_hf_tensor(".attention.attention.query_norm.scale")
      )
      attn["key_norm"]["scale"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".attention.attention.key_norm.scale"))
    else:
      attn["in_proj_qkvz"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.in_proj_qkvz.kernel")
      ).transpose()
      attn["in_proj_ba"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.in_proj_ba.kernel")
      ).transpose()
      conv1d_kernel = to_np_bfloat16(get_hf_tensor(".attention.conv1d.kernel"))
      attn["conv1d"]["kernel"][:, stack_idx, :, :] = conv1d_kernel.transpose(2, 1, 0)
      attn["A_log"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".attention.A_log"))
      attn["dt_bias"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".attention.dt_bias"))
      attn["norm"]["rms_norm"]["scale"][:, stack_idx] = to_np_bfloat16(
          get_hf_tensor(".attention.norm.rms_norm.scale")
      )
      attn["out_proj"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.out_proj.kernel")
      ).transpose()

    mlp = ln["mlp"]
    mlp["routed_experts"]["gate"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.routed_experts.gate.kernel")
    ).transpose()
    for name in ("wi_0", "wi_1", "wo"):
      mlp["shared_expert"][name]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(f".mlp.shared_expert.{name}.kernel")
      ).transpose()
    mlp["shared_expert_gate"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.shared_expert_gate.kernel")
    ).transpose()
    for i in range(num_experts_to_convert):
      for name in ("wi_0", "wi_1", "wo"):
        mlp["routed_experts"][name][i, stack_idx, :, :] = to_np_bfloat16(
            get_hf_tensor(f".mlp.routed_experts.{i}.{name}")
        ).transpose()
  gc.collect()
  return weights


def convert(base_model_path: str, maxtext_model_path: str, model_size: str,
            simulated_cpu_devices_count: int = 16, use_ocdbt: bool = True, use_zarr3: bool = True,
            num_layers_to_convert: int = 0, num_experts_to_convert: int = 0):
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, simulated_cpu_devices_count,
               use_ocdbt, use_zarr3, num_layers_to_convert, num_experts_to_convert)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.standalone_scripts.qwen3_next_cpu_worker import run; "
          "run(result_fd)"
      )],
      input=pickle.dumps(arguments), stdout=subprocess.PIPE, env=environment, check=True,
  )
  error = pickle.loads(child.stdout)
  if error is not None:
    raise error


def _convert(base_model_path: str, maxtext_model_path: str, model_size: str,
             simulated_cpu_devices_count: int, use_ocdbt: bool, use_zarr3: bool,
             num_layers_to_convert: int, num_experts_to_convert: int):
  if model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{model_size}' not found in MODEL_PARAMS_DICT.")
  args = argparse.Namespace(num_layers_to_convert=num_layers_to_convert, num_experts_to_convert=num_experts_to_convert)
  max_logging.log(f"Starting conversion for Qwen3-Next model size: {model_size}")
  weights = convert_hf_to_maxtext(
      base_model_path, MODEL_PARAMS_DICT[model_size], args
  )
  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {maxtext_model_path}")
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
  max_logging.log("Checkpoint saved successfully.")


def main(args):
  if args.model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")
  model_params = MODEL_PARAMS_DICT[args.model_size]
  for key in ("num_layers", "num_experts", "emb_dim", "num_q_heads", "num_kv_heads", "head_dim",
              "inhomogeneous_layer_cycle_interval"):
    model_params[key]
  return convert(
      args.base_model_path, args.maxtext_model_path, args.model_size, args.simulated_cpu_devices_count,
      args.use_ocdbt, args.use_zarr3, args.num_layers_to_convert, args.num_experts_to_convert,
  )
