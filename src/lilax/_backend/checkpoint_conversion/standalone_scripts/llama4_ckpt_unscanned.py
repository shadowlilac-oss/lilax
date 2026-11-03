import gc
import logging
import os
import pathlib
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import psutil
import torch
from safetensors import safe_open
from tqdm import tqdm

from lilax._backend.checkpoint_conversion.standalone_scripts.llama_or_mistral_ckpt import MODEL_PARAMS_DICT
from lilax._backend.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from lilax._backend.utils import max_logging


CAST_DTYPE = ml_dtypes.bfloat16
SIMULATED_CPU_DEVICES_COUNT = 16


def _pytorch_to_maxtext_mapping(layer_idx: int = -1, expert_idx: int = -1, model_size="") -> dict:
  mapping = {
      "tok_embeddings.weight": "model.embed_tokens.weight",
      "norm.weight": "model.norm.weight",
      "output.weight": "lm_head.weight",
      f"layers.{layer_idx}.attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
      f"layers.{layer_idx}.ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      f"layers.{layer_idx}.attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
      f"layers.{layer_idx}.attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
      f"layers.{layer_idx}.attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
      f"layers.{layer_idx}.attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
      f"layers.{layer_idx}.feed_forward.gate.weight": f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
      f"layers.{layer_idx}.feed_forward.w_in_shared_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_out_shared_DF.weight": f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_swiglu_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
  }
  if model_size.startswith("llama4"):
    mapping[f"layers.{layer_idx}.attention.wkq.layer_norm_weight"] = mapping.pop(
        f"layers.{layer_idx}.attention_norm.weight"
    )
    mapping[f"layers.{layer_idx}feed_forward.norm.weight"] = mapping.pop(f"layers.{layer_idx}.ffn_norm.weight")
    for original, expert in (("moe_w_in_eD_F", "w1"), ("moe_w_out_eF_D", "w2"), ("moe_w_swiglu_eD_F", "w3")):
      mapping[f"layers.{layer_idx}.feed_forward.experts.{original}"] = mapping.pop(
          f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.{expert}.weight"
      )
  return mapping


def _hf_to_maxtext_mapping(layer_idx: int = -1) -> dict:
  source = f"language_model.model.layers.{layer_idx}."
  target = f"layers.{layer_idx}."
  return {
      "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
      "language_model.model.norm.weight": "norm.weight",
      "language_model.lm_head.weight": "output.weight",
      source + "input_layernorm.weight": target + "attention_norm.weight",
      source + "post_attention_layernorm.weight": target + "ffn_norm.weight",
      source + "self_attn.q_proj.weight": target + "attention.wq.weight",
      source + "self_attn.k_proj.weight": target + "attention.wk.weight",
      source + "self_attn.v_proj.weight": target + "attention.wv.weight",
      source + "self_attn.o_proj.weight": target + "attention.wo.weight",
      source + "feed_forward.router.weight": target + "feed_forward.gate.weight",
      source + "feed_forward.experts.down_proj": target + "feed_forward.experts.down_proj",
      source + "feed_forward.experts.gate_up_proj": target + "feed_forward.experts.gate_up_proj",
      source + "feed_forward.shared_expert.gate_proj.weight": target + "feed_forward.shared_experts.gate_proj.weight",
      source + "feed_forward.shared_expert.down_proj.weight": target + "feed_forward.shared_experts.down_proj.weight",
      source + "feed_forward.shared_expert.up_proj.weight": target + "feed_forward.shared_experts.up_proj.weight",
      source + "feed_forward.up_proj.weight": target + "feed_forward.w1.weight",
      source + "feed_forward.gate_proj.weight": target + "feed_forward.w3.weight",
      source + "feed_forward.down_proj.weight": target + "feed_forward.w2.weight",
  }


@dataclass
class _NamespaceMapper:
  collection: dict
  model_size: str = ""
  delimiter: str = "."

  def __getitem__(self, key):
    if key in self.collection:
      return self.collection[key]
    fields = [int(field) for field in key.split(self.delimiter) if re.match(r"[0-9]+", field) is not None]
    mapping = _pytorch_to_maxtext_mapping(*fields, model_size=self.model_size)
    if key not in mapping:
      raise ValueError(f"Key `{key}` is missing from the original collection and from the mapping.")
    new_key = mapping[key]
    if new_key not in self.collection:
      raise ValueError(f"New key `{new_key}` mapped from `{key}` is missing from the collection.")
    return self.collection[new_key]


def _pt_to_np(pt_weight, cast_dtype=None, transpose=False):
  weight = pt_weight.to(torch.float32).numpy()
  if cast_dtype:
    weight = weight.astype(cast_dtype)
  return weight.transpose() if transpose else weight


def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict,
                                        mem_info: psutil.Process):
  assert model_params.get("rope_type") == "llama3.1", "NOTE: only `rope_type` of `llama3.1` is supported for now!"
  max_logging.log(f"Loading the base model from {base_model_path}")
  paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  tensors = {}
  for i, path in enumerate(paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(paths)} ...")
    with safe_open(path, framework="pt", device="cpu") as source:
      for key in source.keys():
        parts = key.split(".")
        layer = int(parts[3]) if "layers" in key else 0
        mapped = key if "vision" in key or "multi_modal_projector" in key else _hf_to_maxtext_mapping(layer)[key]
        tensors[mapped] = source.get_tensor(key)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  def value(key, transpose=False):
    return _pt_to_np(tensors[key], cast_dtype=CAST_DTYPE, transpose=transpose)

  vision = {
      "Llama4VisionEncoder_0": {},
      "class_embedding": value("vision_model.class_embedding"),
      "positional_embedding_vlm": value("vision_model.positional_embedding_vlm"),
      "Llama4UnfoldConvolution_0": {
          "vit_unfold_linear": {"kernel": value("vision_model.patch_embedding.linear.weight", True)}
      },
      "layernorm_pre": {"scale": value("vision_model.layernorm_pre.weight"),
                        "bias": value("vision_model.layernorm_pre.bias")},
      "layernorm_post": {"scale": value("vision_model.layernorm_post.weight"),
                         "bias": value("vision_model.layernorm_post.bias")},
      "Llama4VisionPixelShuffleMLP_0": {"pixel_shuffle_mlp": {
          "vit_pixel_shuffle_mlp_fc1": {"kernel": value("vision_model.vision_adapter.mlp.fc1.weight", True)},
          "vit_pixel_shuffle_mlp_fc2": {"kernel": value("vision_model.vision_adapter.mlp.fc2.weight", True)},
      }},
  }
  heads_vit = model_params.get("num_att_head_vit", 0)
  hidden_vit = model_params.get("hidden_size_vit", 0)
  dim_vit = hidden_vit // heads_vit if heads_vit > 0 else 0
  for index in tqdm(range(model_params.get("num_layers_vit", 0)), desc="layers", leave=False):
    prefix = f"vision_model.model.layers.{index}."
    attention = {}
    for name, projection in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
      attention[name] = {
          "kernel": value(prefix + f"self_attn.{projection}.weight", True).reshape(hidden_vit, heads_vit, dim_vit),
          "bias": value(prefix + f"self_attn.{projection}.bias").reshape(heads_vit, dim_vit),
      }
    attention["out"] = {
        "kernel": value(prefix + "self_attn.o_proj.weight", True).reshape(heads_vit, dim_vit, hidden_vit),
        "bias": value(prefix + "self_attn.o_proj.bias"),
    }
    vision["Llama4VisionEncoder_0"][f"layers_{index}"] = {
        "self_attention_vision": attention,
        "Llama4VisionMLP_0": {
            f"vit_encoder_layer_mlp_{fc}": {"kernel": value(prefix + f"mlp.{fc}.weight", True),
                                           "bias": value(prefix + f"mlp.{fc}.bias")}
            for fc in ("fc1", "fc2")
        },
        "input_layer_norm": {"scale": value(prefix + "input_layernorm.weight"),
                             "bias": value(prefix + "input_layernorm.bias")},
        "post_attention_layer_norm": {"scale": value(prefix + "post_attention_layernorm.weight"),
                                      "bias": value(prefix + "post_attention_layernorm.bias")},
    }
  weights = {
      "decoder": {"decoder_norm": {"scale": value("norm.weight")},
                  "logits_dense": {"kernel": value("output.weight", True)[:, :model_params["vocab"]]}},
      "token_embedder": {"embedding": value("tok_embeddings.weight")},
      "vision_encoder": {
          "Llama4VisionModel_0": vision,
          "Llama4MultiModalProjector_0": {
              "vit_multi_modal_projector": {"kernel": value("multi_modal_projector.linear_1.weight", True)}
          },
      },
  }
  if model_size[:6] not in ("llama3", "llama4"):
    weights["token_embedder"]["embedding"] = weights["token_embedder"]["embedding"][:model_params["vocab"], :]
  q_heads, kv_heads, dim = model_params["num_heads"], model_params["num_kv_heads"], model_params["dims_per_head"]
  for index in tqdm(range(model_params["num_layers"]), desc="layers", leave=False):
    prefix = f"layers.{index}."
    attention = {}
    for name, projection, heads in (("query", "wq", q_heads), ("key", "wk", kv_heads), ("value", "wv", kv_heads)):
      kernel = value(prefix + f"attention.{projection}.weight", True).reshape(q_heads * dim, heads, dim)
      if name == "query" and model_params.get("scale_query", True):
        kernel = kernel / np.sqrt(dim).astype(CAST_DTYPE)
      attention[name] = {"kernel": kernel}
    attention["out"] = {"kernel": value(prefix + "attention.wo.weight", True).reshape(q_heads, dim, q_heads * dim)}
    layer = {
        "self_attention": attention,
        "pre_self_attention_layer_norm": {"scale": value(prefix + "attention_norm.weight")},
        "post_self_attention_layer_norm": {"scale": value(prefix + "ffn_norm.weight")},
    }
    if (index + 1) % model_params["interleave_moe_layer_step"] != 0:
      layer["mlp"] = {
          name: {"kernel": value(prefix + f"feed_forward.{source}.weight", True)}
          for name, source in (("wi_0", "w3"), ("wi_1", "w1"), ("wo", "w2"))
      }
    else:
      wi_0, wi_1 = np.split(value(prefix + "feed_forward.experts.gate_up_proj"), 2, axis=-1)
      layer["Llama4MoEBlock_0"] = {
          "MoeBlock_0": {"wi_0": wi_0, "wi_1": wi_1, "wo": value(prefix + "feed_forward.experts.down_proj"),
                         "gate": {"kernel": value(prefix + "feed_forward.gate.weight", True)}},
          "shared_experts": {
              name: {"kernel": value(prefix + f"feed_forward.shared_experts.{projection}.weight", True)}
              for name, projection in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj"))
          },
      }
    weights["decoder"][f"layers_{index}"] = layer
    gc.collect()
  return weights


def _convert_pytorch_to_jax_weights(base_model_path: str, model_size: str, model_params: dict,
                                    mem_info: psutil.Process):
  assert model_params.get("rope_type") == "llama3.1", "NOTE: only `rope_type` of `llama3.1` is supported for now!"
  paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
  shards = {}
  for i, path in enumerate(paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(paths)} ...")
    shards[int(path.name.split(".", maxsplit=2)[1])] = torch.load(path, map_location="cpu", weights_only=True)
  shards = [_NamespaceMapper(shards[i], model_size=model_size) for i in sorted(shards)]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  llama4 = model_size[:6] == "llama4"
  vocab = model_params["vocab"]
  weights = {
      "decoder": {
          "decoder_norm": {"scale": _pt_to_np(shards[0]["norm.weight"], cast_dtype=CAST_DTYPE)},
          "logits_dense": {"kernel": np.concatenate(
              [_pt_to_np(shard["output.weight"], cast_dtype=CAST_DTYPE) for shard in shards], axis=0
          ).transpose()[:, :vocab]},
      },
      "token_embedder": {"embedding": np.concatenate(
          [_pt_to_np(shard["tok_embeddings.weight"], cast_dtype=CAST_DTYPE) for shard in shards],
          axis=0 if model_size[:6] in ("llama3", "llama4") else 1,
      )},
  }
  if model_size[:6] not in ("llama3", "llama4"):
    weights["token_embedder"]["embedding"] = weights["token_embedder"]["embedding"][:vocab, :]
  q_heads, kv_heads, dim = model_params["num_heads"], model_params["num_kv_heads"], model_params["dims_per_head"]
  for index in tqdm(range(model_params["num_layers"]), desc="layers", leave=False):
    prefix = f"layers.{index}."
    dense = (index + 1) % model_params["interleave_moe_layer_step"] != 0
    if llama4:
      hidden = model_params["base_emb_dim"]
      kv_hidden = hidden // (q_heads // kv_heads)
      wq = np.zeros((hidden, hidden), dtype=ml_dtypes.bfloat16)
      wk = np.zeros((kv_hidden, hidden), dtype=ml_dtypes.bfloat16)
      wv = np.zeros((kv_hidden, hidden), dtype=ml_dtypes.bfloat16)
      for i, shard in enumerate(shards):
        qkv = _pt_to_np(shard[prefix + "attention.wqkv.weight"])
        local_d = dim // len(shards)
        local_hidden = hidden // len(shards)
        local_kv = kv_hidden // len(shards)
        wq[i * local_d * q_heads:(i + 1) * local_d * q_heads] = qkv[:local_hidden, :]
        wk[i * local_d * kv_heads:(i + 1) * local_d * kv_heads] = qkv[local_hidden:local_hidden + local_kv, :]
        wv[i * local_d * kv_heads:(i + 1) * local_d * kv_heads] = qkv[local_hidden + local_kv:, :]
    else:
      step = 2 if model_size == "llama3.1-405b" else 1
      wq = np.concatenate([_pt_to_np(shard[prefix + "attention.wq.weight"], cast_dtype=CAST_DTYPE)
                           for shard in shards], axis=0)
      wk = np.concatenate([_pt_to_np(shard[prefix + "attention.wk.weight"], cast_dtype=CAST_DTYPE)
                           for shard in shards[::step]], axis=0)
      wv = np.concatenate([_pt_to_np(shard[prefix + "attention.wv.weight"], cast_dtype=CAST_DTYPE)
                           for shard in shards[::step]], axis=0)
    wq = wq.transpose().reshape(q_heads * dim, q_heads, dim)
    wk = wk.transpose().reshape(q_heads * dim, kv_heads, dim)
    wv = wv.transpose().reshape(q_heads * dim, kv_heads, dim)
    wo = np.concatenate([_pt_to_np(shard[prefix + "attention.wo.weight"], cast_dtype=CAST_DTYPE)
                         for shard in shards], axis=1).transpose().reshape(q_heads, dim, q_heads * dim)
    if model_params.get("scale_query", True):
      wq = wq / np.sqrt(dim).astype(CAST_DTYPE)
    pre_norm = "attention.wqkv.layer_norm_weight" if llama4 else "attention_norm.weight"
    if llama4:
      post_norm = "feed_forward.mlp.layer_norm_weight" if dense else "feed_forward.norm.weight"
    else:
      post_norm = "ffn_norm.weight"
    layer = {
        "self_attention": {"query": {"kernel": wq}, "key": {"kernel": wk},
                           "value": {"kernel": wv}, "out": {"kernel": wo}},
        "pre_self_attention_layer_norm": {"scale": _pt_to_np(shards[0][prefix + pre_norm], cast_dtype=CAST_DTYPE)},
        "post_self_attention_layer_norm": {"scale": _pt_to_np(shards[0][prefix + post_norm], cast_dtype=CAST_DTYPE)},
    }
    if dense:
      wi_0, wi_1 = [], []
      for shard in shards:
        first, second = np.split(_pt_to_np(shard[prefix + "feed_forward.mlp.fc1_weight"]), 2, axis=0)
        wi_0.append(first)
        wi_1.append(second)
      layer["mlp"] = {
          "wi_0": {"kernel": np.concatenate(wi_0, axis=0).transpose()},
          "wi_1": {"kernel": np.concatenate(wi_1, axis=0).transpose()},
          "wo": {"kernel": np.concatenate(
              [_pt_to_np(shard[prefix + "feed_forward.mlp.fc2_weight"], cast_dtype=CAST_DTYPE)
               for shard in shards], axis=1).transpose()},
      }
    else:
      experts = model_params["num_experts"]
      emb = model_params["base_emb_dim"]
      routed = {"gate": {"kernel": _pt_to_np(shards[0][prefix + "feed_forward.router_DE"], cast_dtype=CAST_DTYPE)}}
      for name, source in (("wi_0", "moe_w_in_eD_F"), ("wi_1", "moe_w_swiglu_eD_F")):
        routed[name] = np.concatenate([
            _pt_to_np(shard[prefix + "feed_forward.experts." + source], cast_dtype=CAST_DTYPE).reshape(experts, emb, -1)
            for shard in shards
        ], axis=2)
      routed["wo"] = np.concatenate([
          _pt_to_np(shard[prefix + "feed_forward.experts.moe_w_out_eF_D"], cast_dtype=CAST_DTYPE).reshape(experts, -1, emb)
          for shard in shards
      ], axis=1)
      shared = {}
      for name, source, axis in (("wi_0", "w_in_shared_FD", 0), ("wi_1", "w_swiglu_FD", 0),
                                 ("wo", "w_out_shared_DF", 1)):
        shared[name] = {"kernel": np.concatenate([
            _pt_to_np(shard[prefix + f"feed_forward.{source}.weight"], cast_dtype=CAST_DTYPE) for shard in shards
        ], axis=axis).transpose()}
      layer["Llama4MoEBlock_0"] = {"MoeBlock_0": routed, "shared_experts": shared}
    weights["decoder"][f"layers_{index}"] = layer
    gc.collect()
  del shards
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return weights


def convert_to_jax_weights(base_model_path: str, model_size: str, huggingface_ckpt: bool):
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  if huggingface_ckpt:
    return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)
  return _convert_pytorch_to_jax_weights(base_model_path, model_size, model_params, mem_info)


def convert(base_model_path: str, maxtext_model_path: str, model_size: str, huggingface_checkpoint: bool = False,
            use_ocdbt: bool = True, use_zarr3: bool = True,
            simulated_cpu_devices_count: int = SIMULATED_CPU_DEVICES_COUNT) -> None:
  if model_size not in MODEL_PARAMS_DICT or not model_size.startswith("llama4"):
    raise NotImplementedError("Currently, we only support llama4 models but got " + model_size)
  environment = os.environ.copy()
  environment["JAX_PLATFORMS"] = "cpu"
  environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={simulated_cpu_devices_count}"
  arguments = (os.fspath(base_model_path), os.fspath(maxtext_model_path), model_size, huggingface_checkpoint,
               use_ocdbt, use_zarr3, simulated_cpu_devices_count)
  child = subprocess.run(
      [sys.executable, "-c", (
          "import os; result_fd = os.dup(1); os.dup2(2, 1); "
          "from lilax._backend.checkpoint_conversion.standalone_scripts "
          "import llama4_ckpt_unscanned as converter; converter._run_cpu_worker(result_fd)"
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


def _convert(base_model_path, maxtext_model_path, model_size, huggingface_checkpoint, use_ocdbt, use_zarr3,
             simulated_cpu_devices_count):
  weights = convert_to_jax_weights(base_model_path, model_size, huggingface_checkpoint)
  save_weights_to_checkpoint(maxtext_model_path, weights, simulated_cpu_devices_count, use_ocdbt, use_zarr3)
  max_logging.log(f"Successfully saved base_weights to {maxtext_model_path}.")
