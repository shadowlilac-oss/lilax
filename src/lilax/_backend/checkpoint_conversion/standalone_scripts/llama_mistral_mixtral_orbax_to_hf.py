import numpy as np
import torch
from jax.sharding import Mesh
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM

from lilax._backend.checkpoint_conversion.standalone_scripts.llama_ckpt_conversion_inference_only import MODEL_PARAMS_DICT
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.utils import device_mesh
from lilax._backend.utils.generate_param_only_checkpoint import _read_train_checkpoint
from lilax._backend.utils.max_utils import unpermute_from_match_maxtext_rope


def reverse_scale(arr, scale):
  return arr * np.sqrt(scale)


def load_hf_model(model_size):
  if model_size == "llama2-7b":
    return LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
  if model_size == "mistral-7b":
    return MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
  if model_size == "mixtral-8x7b":
    return AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map="auto")
  if model_size == "llama3.1-8b":
    return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B"))
  raise NotImplementedError


def load_model_state(config):
  mesh = Mesh(device_mesh.create_device_mesh(config), config.mesh_axes)
  manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir, config.enable_checkpointing, config.async_checkpointing, config.checkpoint_period
  )
  training_state, _ = _read_train_checkpoint(config, manager, mesh)
  return training_state


def convert_state_to_hf(training_state, model_size):
  model_params = {
      **MODEL_PARAMS_DICT,
      "mistral-7b": {"num_layers": 32, "num_heads": 32, "num_kv_heads": 8, "dims_per_head": 128},
      "mixtral-8x7b": {"num_layers": 32, "num_heads": 32, "num_kv_heads": 8, "dims_per_head": 128, "num_experts": 8},
      "mixtral-8x22b": {"num_layers": 56, "num_heads": 48, "num_kv_heads": 8, "dims_per_head": 128, "num_experts": 8},
      "llama4-17b-16e": {"num_layers": 48, "num_heads": 40, "num_kv_heads": 8, "dims_per_head": 128, "num_experts": 16},
      "llama4-17b-128e": {"num_layers": 48, "num_heads": 40, "num_kv_heads": 8, "dims_per_head": 128, "num_experts": 128},
  }[model_size]
  heads = model_params["num_heads"]
  kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["dims_per_head"]
  params = training_state.params["params"]
  decoder = params["decoder"]
  layers = decoder["layers"]
  result = {"model.embed_tokens.weight": torch.tensor(np.asarray(params["token_embedder"]["embedding"]), dtype=torch.float16)}
  for layer in tqdm(range(model_params["num_layers"]), desc="Porting parameters layerwise"):
    prefix = f"model.layers.{layer}"
    attention = layers["self_attention"]
    query = reverse_scale(attention["query"]["kernel"][:, layer, :, :], head_dim)
    key = attention["key"]["kernel"][:, layer, :, :]
    query = unpermute_from_match_maxtext_rope(query, model_size).reshape(heads * head_dim, heads * head_dim).T
    key = unpermute_from_match_maxtext_rope(key, model_size).reshape(heads * head_dim, kv_heads * head_dim).T
    value = attention["value"]["kernel"][:, layer, :, :].reshape(heads * head_dim, kv_heads * head_dim).T
    out = attention["out"]["kernel"][:, layer, :, :].reshape(heads * head_dim, heads * head_dim).T
    for name, array in (("q_proj", query), ("k_proj", key), ("v_proj", value), ("o_proj", out)):
      result[f"{prefix}.self_attn.{name}.weight"] = torch.tensor(np.asarray(array), dtype=torch.float16)
    if "num_experts" not in model_params:
      for name, projection in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
        array = layers["mlp"][name]["kernel"][:, layer, :].T
        result[f"{prefix}.mlp.{projection}.weight"] = torch.tensor(np.asarray(array), dtype=torch.float16)
    else:
      moe = layers["MoeBlock_0"]
      result[f"{prefix}.block_sparse_moe.gate.weight"] = torch.tensor(
          np.asarray(moe["gate"]["kernel"][:, layer, :].T), dtype=torch.float16
      )
      for expert in range(model_params["num_experts"]):
        for name, projection in (("wi_0", "w1"), ("wo", "w2"), ("wi_1", "w3")):
          array = moe[name][expert, layer, :, :].T
          result[f"{prefix}.block_sparse_moe.experts.{expert}.{projection}.weight"] = torch.tensor(
              np.asarray(array), dtype=torch.float16
          )
    for name, output in (("pre_self_attention_layer_norm", "input_layernorm"),
                         ("post_self_attention_layer_norm", "post_attention_layernorm")):
      array = layers[name]["scale"][:, layer].reshape(heads * head_dim)
      result[f"{prefix}.{output}.weight"] = torch.tensor(np.asarray(array), dtype=torch.float16)
  result["lm_head.weight"] = torch.tensor(np.asarray(decoder["logits_dense"]["kernel"].T), dtype=torch.float16)
  result["model.norm.weight"] = torch.tensor(
      np.asarray(decoder["decoder_norm"]["scale"].reshape(heads * head_dim)), dtype=torch.float16
  )
  return result


def convert_orbax_hf(hf_model_path, config):
  model = load_hf_model(config.model_name)
  state = load_model_state(config)
  model.save_pretrained(hf_model_path, state_dict=convert_state_to_hf(state, config.model_name))
