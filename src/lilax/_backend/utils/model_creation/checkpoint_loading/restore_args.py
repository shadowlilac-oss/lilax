"""Adjust Orbax restore arguments for checkpoint arrays stored at different shapes than the model."""

import dataclasses

import jax
from orbax import checkpoint as ocp

from lilax._backend.integration.vllm.convert_utils import _partition_size
from lilax._backend.utils import max_logging
from lilax._backend.utils.model_creation.checkpoint_loading.metadata import _is_orbax_array_metadata


def _stored_shape_evenly_shardable(restore_arg, stored_shape):
  """Return whether the stored shape is evenly partitioned by the restore sharding."""
  restore_sharding = restore_arg.sharding
  if not isinstance(restore_sharding, jax.sharding.NamedSharding):
    return False
  spec = restore_sharding.spec
  for axis_idx, dim in enumerate(stored_shape):
    partition = spec[axis_idx] if axis_idx < len(spec) else None
    if dim % _partition_size(partition, restore_sharding.mesh) != 0:
      return False
  return True


def _fix_restore_args_for_shape_mismatch(restore_args, stored_metadata_tree, mesh):
  """Adjust restore arguments for checkpoint arrays with different stored shapes."""
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def _key_str(key):
    if hasattr(key, "key"):
      return str(key.key)
    if hasattr(key, "attr"):
      return str(key.attr)
    if hasattr(key, "name"):
      return str(key.name)
    return str(key)

  def _lookup_stored_meta(path):
    node = stored_metadata_tree
    for key in path:
      if isinstance(key, jax.tree_util.SequenceKey):
        if isinstance(node, (list, tuple)) and 0 <= key.idx < len(node):
          node = node[key.idx]
          continue
        return None
      if not isinstance(node, dict):
        return None
      name = _key_str(key)
      if name in node:
        node = node[name]
        continue
      raw = str(key)
      if raw in node:
        node = node[raw]
        continue
      return None
    return node

  mismatched_paths_sharded = []
  mismatched_paths_replicated = []
  rank_mismatched_paths = []
  missing_paths = []
  found_array_count = [0]

  def _fix_one(path, restore_arg):
    if not isinstance(restore_arg, ocp.ArrayRestoreArgs):
      return restore_arg
    stored_meta = _lookup_stored_meta(path)
    if stored_meta is None:
      missing_paths.append(f"  {'.'.join(_key_str(key) for key in path)}")
      return restore_arg
    if _is_orbax_array_metadata(stored_meta):
      stored_shape = tuple(stored_meta.shape)
      if restore_arg.global_shape is not None and restore_arg.global_shape != stored_shape:
        if len(stored_shape) != len(restore_arg.global_shape):
          rank_mismatched_paths.append(
              f"  {'.'.join(_key_str(key) for key in path)}: "
              f"checkpoint shape {stored_shape} (rank {len(stored_shape)}) "
              f"vs model shape {restore_arg.global_shape} (rank {len(restore_arg.global_shape)})"
          )
        else:
          found_array_count[0] += 1
          path_str = f"  {'.'.join(_key_str(key) for key in path)}: stored={stored_shape} -> model={restore_arg.global_shape}"
          if _stored_shape_evenly_shardable(restore_arg, stored_shape):
            mismatched_paths_sharded.append(path_str)
            return dataclasses.replace(restore_arg, global_shape=stored_shape, shape=stored_shape)
          mismatched_paths_replicated.append(path_str)
          return dataclasses.replace(
              restore_arg, global_shape=None, shape=None, sharding=replicated, mesh=None, mesh_axes=None
          )
      else:
        found_array_count[0] += 1
    return restore_arg

  fixed = jax.tree_util.tree_map_with_path(_fix_one, restore_args, is_leaf=lambda x: isinstance(x, ocp.ArrayRestoreArgs))
  if rank_mismatched_paths:
    sample = "\n".join(rank_mismatched_paths[:5])
    more = f"\n  ... and {len(rank_mismatched_paths) - 5} more" if len(rank_mismatched_paths) > 5 else ""
    raise ValueError(f"Checkpoint rank mismatches detected ({len(rank_mismatched_paths)} arrays):\n{sample}{more}")

  total_arrays = found_array_count[0] + len(rank_mismatched_paths) + len(missing_paths)
  if total_arrays > 0 and len(missing_paths) / total_arrays > 0.8:
    sample = "\n".join(missing_paths[:5])
    more = f"\n  ... and {len(missing_paths) - 5} more" if len(missing_paths) > 5 else ""
    raise ValueError(
        f"Checkpoint structure mismatch: {len(missing_paths)} of {total_arrays}"
        " model parameter paths were not found in the checkpoint.\nExample"
        f" missing paths:\n{sample}{more}"
    )

  if mismatched_paths_sharded:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_sharded)} arrays): loading sharded at "
        "stored shape and padding each local shard after restore.\n" + "\n".join(mismatched_paths_sharded)
    )
  if mismatched_paths_replicated:
    max_logging.log(
        f"Checkpoint shape mismatches ({len(mismatched_paths_replicated)} arrays): loading with replicated "
        "sharding (stored shape not evenly partitionable across mesh) and expanding after restore.\n"
        + "\n".join(mismatched_paths_replicated)
    )
  return fixed
