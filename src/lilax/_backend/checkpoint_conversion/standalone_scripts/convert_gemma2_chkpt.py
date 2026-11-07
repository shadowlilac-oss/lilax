from typing import Any
import copy
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import orbax

from lilax._backend.observability import max_logging
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


def main(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  if model_size not in ("2b", "9b", "27b"):
    raise NotImplementedError("only implemented for gemma 2 classes")

  _convert(base_model_path, maxtext_model_path, model_size)


def _convert(base_model_path: str, maxtext_model_path: str, model_size: str) -> None:
  run_conversion("gemma2", base_model_path, maxtext_model_path, model_size)


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
  query_pre_attn_scalar = head_dim**-0.5 if model_size in ("2b", "9b") else (embed_dim // num_heads) ** -0.5
  transpose_gating_einsum = model_size != "2b"
  jax_weights = {
      "decoder": {"decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1}},
      "token_embedder": {"embedding": params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
  }
  attention = {kind: {name: {"kernel": []} for name in ("query", "key", "value", "out")}
               for kind in ("local", "global")}
  layer_weight = {}
  for kind in ("local", "global"):
    layer_weight["mlp_" + kind] = {name: {"kernel": []} for name in ("wi_0", "wi_1", "wo")}
    for name in ("pre_self_attention_norm", "pre_ffw_norm", "post_self_attention_norm", "post_ffw_norm"):
      layer_weight[name + "_" + kind] = {"scale": []}

  for layer_idx in range(0, num_layers, 2):
    for offset, kind in enumerate(("local", "global")):
      layer = params["transformer"]["layer_" + str(layer_idx + offset)]
      attn = attention[kind]
      attn["query"]["kernel"].append(layer["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * query_pre_attn_scalar)
      attn["key"]["kernel"].append(layer["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2)))
      attn["value"]["kernel"].append(layer["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2)))
      attn["out"]["kernel"].append(layer["attn"]["attn_vec_einsum"]["w"])
      mlp = layer_weight["mlp_" + kind]
      for index in (0, 1):
        weight = layer["mlp"]["gating_einsum"]["w"][index]
        mlp["wi_" + str(index)]["kernel"].append(np.transpose(weight) if transpose_gating_einsum else weight)
      mlp["wo"]["kernel"].append(layer["mlp"]["linear"]["w"])
      for output, source in (("pre_self_attention_norm", "pre_attention_norm"), ("pre_ffw_norm", "pre_ffw_norm"),
                             ("post_self_attention_norm", "post_attention_norm"), ("post_ffw_norm", "post_ffw_norm")):
        layer_weight[output + "_" + kind]["scale"].append(layer[source]["scale"] + 1)

  for kind in ("local", "global"):
    for weights in attention[kind].values():
      weights["kernel"] = np.array(weights["kernel"]).transpose((1, 0, 2, 3))
    for weights in layer_weight["mlp_" + kind].values():
      weights["kernel"] = np.array(weights["kernel"]).transpose((1, 0, 2))
    for name in ("pre_self_attention_norm", "pre_ffw_norm", "post_self_attention_norm", "post_ffw_norm"):
      weights = layer_weight[name + "_" + kind]
      weights["scale"] = np.array(weights["scale"]).transpose((1, 0))
    layer_weight["self_attention_" + kind] = copy.deepcopy(attention[kind])
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
