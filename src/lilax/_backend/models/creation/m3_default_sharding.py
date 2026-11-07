"""Default named sharding for M3 variables that carry no partition annotations."""

import math
from typing import Any

from flax import nnx
import jax

from lilax._backend.observability import max_logging


def _apply_m3_default_named_sharding(abs_var_state: Any, named_sharding_state: Any, mesh: Any) -> Any:
  """Assign 1D named sharding to unpartitioned M3 variables."""
  shard_axes = tuple(axis for axis in ("fsdp", "fsdp_transpose") if axis in mesh.axis_names and int(mesh.shape[axis]) > 1)
  if not shard_axes:
    fallback_axes = [
        (axis, int(mesh.shape[axis]))
        for axis in mesh.axis_names if axis not in ("data", "stage") and int(mesh.shape[axis]) > 1
    ]
    if not fallback_axes:
      max_logging.log(
          "WARNING: M3 default sharding found no fsdp/fsdp_transpose or"
          " non-data axis with size > 1; keeping weights replicated."
      )
      return named_sharding_state
    shard_axes = (max(fallback_axes, key=lambda item: item[1])[0],)
  axis_size = math.prod(int(mesh.shape[axis]) for axis in shard_axes)
  spec_axis = shard_axes[0] if len(shard_axes) == 1 else shard_axes

  def _shard_leaf(path, abstract_var, sharding_var):
    current_sharding = sharding_var.get_value() if hasattr(sharding_var, "get_value") else sharding_var
    is_unsharded = (
        not hasattr(current_sharding, "spec")
        or current_sharding.spec == jax.sharding.PartitionSpec()
        or all(axis is None for axis in current_sharding.spec)
    )
    if not is_unsharded:
      return sharding_var
    inner_abstract = abstract_var.get_value() if hasattr(abstract_var, "get_value") else abstract_var
    if not hasattr(inner_abstract, "shape"):
      return sharding_var
    shape = inner_abstract.shape
    path_strs = tuple(str(getattr(key, "key", getattr(key, "attr", key))) for key in path)
    start_dim = 1 if "cycles" in path_strs else 0
    for dim in range(start_dim, len(shape)):
      if shape[dim] % axis_size == 0:
        new_spec = jax.sharding.PartitionSpec(*(spec_axis if index == dim else None for index in range(len(shape))))
        new_sharding = jax.sharding.NamedSharding(mesh, new_spec)
        return sharding_var.replace(new_sharding) if hasattr(sharding_var, "replace") else new_sharding
    return sharding_var

  return jax.tree_util.tree_map_with_path(
      _shard_leaf, abs_var_state, named_sharding_state, is_leaf=lambda node: isinstance(node, nnx.Variable)
  )
