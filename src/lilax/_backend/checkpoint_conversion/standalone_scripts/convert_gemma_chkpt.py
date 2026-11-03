from typing import Any
import copy
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import orbax

from lilax._backend.utils import max_logging
from lilax._backend.common import checkpointing
from lilax._backend.checkpoint_conversion.standalone_scripts._gemma_conversion_process import run_conversion

Params = dict[str, Any]


def nest_params(params: Params) -> Params:
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split("/")
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params


def main(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  if model_size not in ("2b", "7b", "9b"):
    raise NotImplementedError(model_size)

  _convert(base_model_path, maxtext_model_path, model_size)


def _convert(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  run_conversion("gemma", base_model_path, maxtext_model_path, model_size)


def _convert_in_worker(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  print("Loading checkpoint")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = nest_params(checkpointer.restore(base_model_path))
  num_layers = max((int(k.split("_")[1]) for k in params["transformer"] if "layer_" in k)) + 1
  hidden_dim, embed_dim = params["transformer"]["layer_0"]["mlp"]["linear"]["w"].shape
  num_heads, head_dim, _ = params["transformer"]["layer_0"]["attn"]["attn_vec_einsum"]["w"].shape
  print("Model configurations from checkpoint")
  print(f"num_layers: {num_layers}")
  print(f"hidden_dim: {hidden_dim}")
  print(f"embed_dim: {embed_dim}")
  print(f"num_heads: {num_heads}")
  print(f"head_dim: {head_dim}")

  jax_weights = {
      "decoder": {"decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1}},
      "token_embedder": {"embedding": params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
  }
  self_attention = {name: {"kernel": []} for name in ("query", "key", "value", "out")}
  layer_weight = {
      "mlp": {name: {"kernel": []} for name in ("wi_0", "wi_1", "wo")},
      "pre_self_attention_norm": {"scale": []},
      "pre_ffw_norm": {"scale": []},
  }
  for layer_idx in range(num_layers):
    layer = params["transformer"]["layer_" + str(layer_idx)]
    if model_size in ("2b", "9b"):
      query = layer["attn"]["q_einsum"]["w"]
      key = layer["attn"]["kv_einsum"]["w"][0]
      value = layer["attn"]["kv_einsum"]["w"][1]
    else:
      query = layer["attn"]["qkv_einsum"]["w"][0]
      key = layer["attn"]["qkv_einsum"]["w"][1]
      value = layer["attn"]["qkv_einsum"]["w"][2]
    self_attention["query"]["kernel"].append(query.transpose((1, 0, 2)) * head_dim**-0.5)
    self_attention["key"]["kernel"].append(key.transpose((1, 0, 2)))
    self_attention["value"]["kernel"].append(value.transpose((1, 0, 2)))
    self_attention["out"]["kernel"].append(layer["attn"]["attn_vec_einsum"]["w"])
    layer_weight["mlp"]["wi_0"]["kernel"].append(layer["mlp"]["gating_einsum"]["w"][0])
    layer_weight["mlp"]["wi_1"]["kernel"].append(layer["mlp"]["gating_einsum"]["w"][1])
    layer_weight["mlp"]["wo"]["kernel"].append(layer["mlp"]["linear"]["w"])
    layer_weight["pre_self_attention_norm"]["scale"].append(layer["pre_attention_norm"]["scale"] + 1)
    layer_weight["pre_ffw_norm"]["scale"].append(layer["pre_ffw_norm"]["scale"] + 1)

  for weights in self_attention.values():
    weights["kernel"] = np.array(weights["kernel"]).transpose((1, 0, 2, 3))
  for weights in layer_weight["mlp"].values():
    weights["kernel"] = np.array(weights["kernel"]).transpose((1, 0, 2))
  for name in ("pre_self_attention_norm", "pre_ffw_norm"):
    layer_weight[name]["scale"] = np.array(layer_weight[name]["scale"]).transpose((1, 0))
  layer_weight["self_attention"] = copy.deepcopy(self_attention)
  jax_weights["decoder"]["layers"] = copy.deepcopy(layer_weight)
  jax_weights = jax.tree_util.tree_map(jnp.array, jax_weights)
  jax_weights = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16) if isinstance(x, jnp.ndarray) else x, jax_weights)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(maxtext_model_path, True, False, 1)
  state_new = train_state.TrainState(step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={})
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, state_new):
      max_logging.log("saved a checkpoint at step 0")
    if checkpointing.reached_preemption(checkpoint_manager, 0):
      checkpointing.wait_until_finished(checkpoint_manager)
      sys.exit()
