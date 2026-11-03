"""Build Orbax restore requests and unwrap restored parameters for Linen and NNX checkpoint layouts."""

from flax import nnx
from flax.core.meta import Partitioned
import jax
from orbax import checkpoint as ocp

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.m3.core import checkpoint as m3_checkpoint
from lilax._backend.utils.model_creation.checkpoint_loading.moe_fusion import adjust_target
from lilax._backend.utils.model_creation.checkpoint_loading.restore_args import _fix_restore_args_for_shape_mismatch


def _intersect_with_metadata(target, meta_tree):
  """Returns the sub-tree of target whose paths exist in checkpoint metadata."""
  if not hasattr(target, "items"):
    return target
  out = {}
  for key, value in target.items():
    if hasattr(meta_tree, "items") and key in meta_tree:
      sub_tree = _intersect_with_metadata(value, meta_tree[key])
      if not hasattr(sub_tree, "items") or sub_tree:
        out[key] = sub_tree
  return out


def _build_linen_restore_request(param_state, metadata_tree, config, mesh):
  target = jax.tree.map(lambda value: value[...], param_state, is_leaf=lambda node: isinstance(node, nnx.Variable))
  if config.use_m3_model:
    target = m3_checkpoint.match_checkpoint_layer_names(target, metadata_tree["params"]["params"])
  target = adjust_target(target, metadata_tree["params"]["params"], False)
  collections = metadata_tree["params"]
  item = {"params": {"params": target}}
  for collection, collection_metadata in collections.items():
    if collection != "params":
      collection_target = _intersect_with_metadata(target, collection_metadata)
      if collection_target:
        item["params"][collection] = collection_target
  restore_args = {"params": {
      collection: _fix_restore_args_for_shape_mismatch(
          ocp.checkpoint_utils.construct_restore_args(collection_target), collections[collection], mesh
      ) for collection, collection_target in item["params"].items()
  }}
  return item, restore_args


def _build_nnx_restore_request(layout, param_state, metadata_tree, config, mesh):
  param_metadata = metadata_tree.get("base", metadata_tree)

  def build_target(value):
    inner = value.get_value() if hasattr(value, "get_value") else value[...]
    if hasattr(inner, "shape"):
      return {"value": value[...]} if layout.use_value_wrapper else value[...]
    inner = jax.tree.map(
        lambda leaf: leaf.value if isinstance(leaf, Partitioned) else leaf,
        inner, is_leaf=lambda leaf: isinstance(leaf, Partitioned),
    )
    return {"value": inner} if layout.use_value_wrapper else inner

  target = jax.tree.map(build_target, param_state, is_leaf=lambda node: isinstance(node, nnx.Variable))
  if config.use_m3_model:
    target = m3_checkpoint.match_checkpoint_layer_names(target, param_metadata)
  target = adjust_target(target, param_metadata, True, layout.use_value_wrapper)
  item = {"base": target} if layout.has_base_key else target
  restore_args = _fix_restore_args_for_shape_mismatch(
      ocp.checkpoint_utils.construct_restore_args(target), param_metadata, mesh
  )
  if layout.has_base_key:
    restore_args = {"base": restore_args}
  return item, restore_args


def build_restore_request(layout, param_state, metadata_tree, config, mesh):
  """Return the Orbax restore item and restore arguments matching the checkpoint layout."""
  if not layout.is_nnx:
    return _build_linen_restore_request(param_state, metadata_tree, config, mesh)
  return _build_nnx_restore_request(layout, param_state, metadata_tree, config, mesh)


def extract_restored_params(layout, restored):
  """Return restored parameters without the checkpoint layout's nesting."""
  if layout.is_nnx:
    restored_root = restored["base"] if layout.has_base_key else restored
    return jax.tree.map(
        lambda value: value["value"], restored_root,
        is_leaf=lambda value: isinstance(value, dict) and "value" in value and not isinstance(value.get("value"), dict),
    ) if layout.use_value_wrapper else restored_root
  checkpoint = restored["params"]["params"]
  for collection, collection_tree in restored["params"].items():
    if collection != "params":
      checkpoint = checkpointing._deep_merge_dicts(checkpoint, collection_tree)
  return checkpoint
