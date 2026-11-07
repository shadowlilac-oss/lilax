"""Restore an Orbax parameter checkpoint into a sharded NNX model."""

from etils import epath
from flax import nnx
import jax
from orbax import checkpoint as ocp

from lilax._backend.checkpoint_conversion.projection_layout import projection_layout_adapter
from lilax._backend.m3.core import checkpoint as m3_checkpoint
from lilax._backend.models.creation.checkpoint_loading.alignment import align_checkpoint_tree
from lilax._backend.models.creation.checkpoint_loading.metadata import detect_checkpoint_layout
from lilax._backend.models.creation.checkpoint_loading.moe_fusion import _fuse_moe_weights
from lilax._backend.models.creation.checkpoint_loading.projection_restore import fuse_restored_projections
from lilax._backend.models.creation.checkpoint_loading.restore_layout import (
    build_restore_request, extract_restored_params,
)


def _free_device_memory(sharded_state):
  """Delete the model's parameter buffers so the restore can reuse their device memory."""
  def free_device_memory(path, node):
    is_custom = any("custom_linear" in str(getattr(part, "key", part)) for part in path)
    if (isinstance(node, nnx.Variable)
        and not isinstance(node, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
        and not is_custom):
      inner = node.get_value() if hasattr(node, "get_value") else node[...]
      for leaf in jax.tree_util.tree_leaves(inner):
        if isinstance(leaf, jax.Array) and not leaf.is_deleted():
          leaf.delete()
    elif isinstance(node, jax.Array) and not node.is_deleted() and not is_custom:
      node.delete()
    return node

  jax.tree_util.tree_map_with_path(free_device_memory, sharded_state, is_leaf=lambda node: isinstance(node, nnx.Variable))


def to_dict(tree):
  return {key: to_dict(value) for key, value in tree.items()} if hasattr(tree, "items") else tree


def _verify_all_variables_restored(model, config):
  unrestored = [
      jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
      if isinstance(leaf, jax.Array) and leaf.is_deleted()
  ]
  if unrestored:
    raise ValueError(
        f"Checkpoint '{config.load_parameters_path}' has no value for {len(unrestored)} model "
        f"variables: {unrestored[:5]}"
    )


def restore_orbax_params(model, sharded_state, specs, config, mesh):
  """Restore checkpoint parameters into the model, aligned to its shapes and sharding."""
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
  )
  metadata = ckptr.metadata(config.load_parameters_path)
  if metadata is None or metadata.item_metadata is None:
    raise ValueError(f"No valid Orbax checkpoint found at '{config.load_parameters_path}'. "
                     "Please check your load_parameters_path.")

  param_state = sharded_state.filter(
      lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
  )
  metadata_tree = metadata.item_metadata.tree
  layout = detect_checkpoint_layout(metadata_tree, param_state)
  adapter = projection_layout_adapter(config)
  item, restore_args = build_restore_request(layout, param_state, metadata_tree, config, mesh, adapter)

  _free_device_memory(sharded_state)
  restored = ckptr.restore(epath.Path(config.load_parameters_path), item=item, transforms={}, restore_args=restore_args)
  checkpoint = extract_restored_params(layout, restored)

  if not checkpoint:
    raise ValueError(
        f"Checkpoint restore from '{config.load_parameters_path}' yielded no parameters. "
        "This usually means the checkpoint format is incompatible with the model configuration "
        "(e.g. a scanned checkpoint loaded with scan_layers=False, or vice versa). "
        "Please ensure the checkpoint format matches the scan_layers setting."
    )
  if config.use_m3_model:
    checkpoint = m3_checkpoint.from_legacy_layer_names(checkpoint, sharded_state)
  model_arrays = jax.tree.map(
      lambda value: value.get_value() if hasattr(value, "get_value") else value[...],
      sharded_state, is_leaf=lambda node: isinstance(node, nnx.Variable),
  )
  logical_axes = jax.tree.map(
      lambda value: value.get_value(), specs, is_leaf=lambda node: isinstance(node, nnx.Variable)
  )

  model_arrays, checkpoint, logical_axes = to_dict(model_arrays), to_dict(checkpoint), to_dict(logical_axes)
  checkpoint = _fuse_moe_weights(checkpoint, model_arrays)
  checkpoint = fuse_restored_projections(checkpoint, model_arrays, config, adapter)
  del restored

  checkpoint = align_checkpoint_tree(checkpoint, model_arrays, logical_axes)
  nnx.update(model, checkpoint)
  _verify_all_variables_restored(model, config)
