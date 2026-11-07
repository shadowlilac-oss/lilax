"""Utilities for Muon optimizer integration and dimension number generation."""

import collections.abc
import sys
from typing import Optional, Tuple

from flax import nnx
import jax
from optax.contrib._muon import MuonDimensionNumbers as mdn

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.distributed import device_mesh
from lilax._backend.models.creation import nnx_abstract_model
from lilax._backend.configs.inert import MaxTextConfig


def _is_path_contain_any(tuples, path):
  return any(x in path for x in tuples)


EXCLUDED_SUBSTRINGS = (
    "scale", "embedding", "logits_dense", "post_beta", "pre_beta", "res_beta", "hc_base", "sinks",
    "tid2eid", "A_log", "dt_bias", "conv1d", "shared_expert_gate",
)
EXCLUDED_EXACT_SEGMENTS = {"bias"}
ATTENTION_BLOCK_NAMES = (
    "self_attention", "full_attention", "attention", "self_attn", "attn", "attention_mla", "GptOssAttention",
)
ATTENTION_QKV_NAMES = ("query", "key", "value", "wq_b", "wkv_b", "wkv")
ATTENTION_OUT_NAMES = ("out",)
MOE_BLOCK_NAMES = ("MoeBlock_0", "moe_block", "routed_experts", "GptOssMlp", "routed_moe")


def transform_logic(
    path: Tuple[str, ...], shape: Optional[Tuple[int, ...]] = None, include_routers: bool = True,
) -> Optional[mdn]:
  if shape is not None and len(shape) < 2:
    return None
  if any(
      segment in EXCLUDED_EXACT_SEGMENTS
      or (segment.endswith("bias") and segment != "position_bias")
      or any(x in segment for x in EXCLUDED_SUBSTRINGS)
      for segment in path
  ):
    return None
  if _is_path_contain_any(MOE_BLOCK_NAMES, path) and not _is_path_contain_any(("shared_experts", "shared_expert"), path):
    if _is_path_contain_any(("wi", "wi_0", "wi_1", "wo", "gate_up_proj"), path):
      return mdn((-2,), (-1,))
    if _is_path_contain_any(("gate", "router"), path):
      return mdn((0,), (-1,)) if include_routers else None
  if "o_a_proj" in path:
    return mdn((-2,), (-1,))
  if _is_path_contain_any(ATTENTION_BLOCK_NAMES, path) and (shape is None or len(shape) > 2):
    if _is_path_contain_any(ATTENTION_QKV_NAMES, path):
      return mdn((0,), (-2, -1))
    if _is_path_contain_any(ATTENTION_OUT_NAMES, path):
      if _is_path_contain_any(("self_attention", "GptOssAttention"), path) or (shape is not None and len(shape) == 4):
        return mdn((0, -2), (-1,))
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=(), include_routers: bool = True):
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: get_transform_tree(v, path=path + (k,), include_routers=include_routers) for k, v in tree.items()}
  else:
    val = getattr(tree, "value", tree)
    val_shape = getattr(val, "shape", None)
    return transform_logic(path, shape=val_shape, include_routers=include_routers)


def get_muon_weight_dimension_numbers(model, config=None, verbose=False):
  include_routers = getattr(config, "muon_include_routers", True) if config is not None else True
  _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

  def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf):
    path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
    val = leaf.get_value() if hasattr(leaf, "get_value") else leaf
    val_shape = getattr(val, "shape", None)
    return transform_logic(path_strings, shape=val_shape, include_routers=include_routers)

  muon_weight_dimension_numbers = jax.tree_util.tree_map_with_path(apply_transform_nnx, nnx.to_pure_dict(abstract_param))
  muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)
  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  def get_leaf_info(leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct):
      return {"shape": leaf.shape}
    return {"shape": "N/A"}

  info_tree = jax.tree_util.tree_map(get_leaf_info, abstract_param, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
  print(f"\n=== Model Structure ===\n{info_tree}")
  print(f"\n=== Muon Dimension Numbers ===\n{muon_weight_dimension_numbers}")
  print("\nIs this reasonable?")


def get_model_mdn(model_name, scan_layers=True, verbose=False, include_routers=True):
  config = resolve_runtime_config(MaxTextConfig(
      model_name=model_name, scan_layers=scan_layers, attention="dot_product",
      muon_include_routers=include_routers, skip_jax_distributed_system=True,
  ))
  devices_array = device_mesh.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  _, model = nnx_abstract_model.create_nnx_abstract_model(config, mesh)
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
  return {"params": nnx.to_pure_dict(muon_weight_dimension_numbers)}


if __name__ == "__main__":
  if len(sys.argv) not in (3, 4):
    print("Usage: python3 -m lilax._backend.optimizers.muon_config.dimensions <model_name> <scan_layers:True/False> [include_routers:True/False]")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  include_routers_arg = sys.argv[3].lower() == "true" if len(sys.argv) == 4 else True
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True, include_routers=include_routers_arg)
