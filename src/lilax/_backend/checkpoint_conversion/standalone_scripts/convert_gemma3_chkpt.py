from typing import Any
import copy
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import orbax

from lilax._backend.utils import max_logging
from lilax._backend.common.checkpoints import checkpointing
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


def rename_nested_keys(data, old_key, new_key):
  new_data = {}
  for key, value in data.items():
    new_k = new_key if key == old_key else key
    if isinstance(value, dict):
      new_data[new_k] = rename_nested_keys(value, old_key, new_key)
    elif isinstance(value, list):
      new_data[new_k] = [rename_nested_keys(item, old_key, new_key) if isinstance(item, dict) else item for item in value]
    else:
      new_data[new_k] = value
  return new_data


def main(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  if model_size not in ("4b", "12b", "27b"):
    raise NotImplementedError("only implemented for gemma 3 classes")

  _convert(base_model_path, maxtext_model_path, model_size)


def _convert(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  run_conversion("gemma3", base_model_path, maxtext_model_path, model_size)


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
  vision = params["SigLiPFromPatches_0"]["siglip_encoder"]
  embedder = params["transformer"]["embedder"]
  jax_weights = {
      "decoder": {"decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1}},
      "token_embedder": {"embedding": embedder["input_embedding"] * jnp.sqrt(embed_dim)},
      "vision_encoder": {
          "Gemma3VisionEncoderLayer_0": {
              "embedding": {"bias": vision["embedding"]["bias"], "kernel": vision["embedding"]["kernel"]},
              "pos_embedding": vision["pos_embedding"],
              "Transformer": vision["Transformer"],
          },
          "VisionEmbedder_0": {
              "mm_input_projection": embedder["mm_input_projection"],
              "mm_soft_embedding_norm": {"scale": embedder["mm_soft_embedding_norm"]["scale"] + 1},
          },
      },
  }
  jax_weights["vision_encoder"] = rename_nested_keys(jax_weights["vision_encoder"], "MlpBlock_0", "MlpBlockViT_0")
  self_attention = {name: {"kernel": []} for name in ("query", "key", "value", "out")}
  self_attention.update({name: {"scale": []} for name in ("query_norm", "key_norm")})
  layer_weight = {"mlp": {name: {"kernel": []} for name in ("wi_0", "wi_1", "wo")}}
  for name in ("pre_self_attention_norm", "pre_ffw_norm", "post_self_attention_norm", "post_ffw_norm"):
    layer_weight[name] = {"scale": []}

  for layer_idx in range(num_layers):
    layer = params["transformer"]["layer_" + str(layer_idx)]
    self_attention["query"]["kernel"].append(layer["attn"]["q_einsum"]["w"].transpose((1, 0, 2)))
    self_attention["key"]["kernel"].append(layer["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2)))
    self_attention["value"]["kernel"].append(layer["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2)))
    self_attention["out"]["kernel"].append(layer["attn"]["attn_vec_einsum"]["w"])
    self_attention["key_norm"]["scale"].append(layer["attn"]["_key_norm"]["scale"] + 1)
    self_attention["query_norm"]["scale"].append(layer["attn"]["_query_norm"]["scale"] + 1)
    for index in (0, 1):
      layer_weight["mlp"]["wi_" + str(index)]["kernel"].append(np.transpose(layer["mlp"]["gating_einsum"]["w"][index]))
    layer_weight["mlp"]["wo"]["kernel"].append(layer["mlp"]["linear"]["w"])
    for output, source in (("pre_self_attention_norm", "pre_attention_norm"), ("pre_ffw_norm", "pre_ffw_norm"),
                           ("post_self_attention_norm", "post_attention_norm"), ("post_ffw_norm", "post_ffw_norm")):
      layer_weight[output]["scale"].append(layer[source]["scale"] + 1)

  for name in ("query", "key", "value", "out"):
    self_attention[name]["kernel"] = np.array(self_attention[name]["kernel"]).transpose((1, 0, 2, 3))
  for name in ("query_norm", "key_norm"):
    self_attention[name]["scale"] = np.array(self_attention[name]["scale"]).transpose((1, 0))
  for weights in layer_weight["mlp"].values():
    weights["kernel"] = np.array(weights["kernel"]).transpose((1, 0, 2))
  for name in ("pre_self_attention_norm", "pre_ffw_norm", "post_self_attention_norm", "post_ffw_norm"):
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
      max_logging.log(f"Checkpoint saved to: {maxtext_model_path}")
    if checkpointing.reached_preemption(checkpoint_manager, 0):
      checkpointing.wait_until_finished(checkpoint_manager)
      sys.exit()
