"""Inspect Orbax checkpoint metadata trees."""

import dataclasses
from typing import Any

from flax import nnx
import jax
from orbax.checkpoint.metadata import ArrayMetadata


def _is_orbax_array_metadata(node):
  return isinstance(node, ArrayMetadata)


def _is_array_meta_or_qtensor(node: Any) -> bool:
  """Returns whether a node is array metadata or a QTensor metadata tree."""
  if _is_orbax_array_metadata(node):
    return True
  if not isinstance(node, dict) or "qvalue" not in node:
    return False
  return all(_is_orbax_array_metadata(item) for item in jax.tree.leaves(node, is_leaf=_is_orbax_array_metadata))


def _checkpoint_uses_value_wrapper(meta_tree: Any, param_state: Any) -> bool:
  """Returns whether checkpoint metadata wraps NNX variable leaves under 'value'."""
  if not isinstance(meta_tree, dict) or not meta_tree:
    return False

  if param_state is not None and hasattr(param_state, "flat_state"):
    for path, var in param_state.flat_state():
      if not isinstance(var, nnx.Param):
        continue
      node = meta_tree
      resolved = True
      for key in path:
        if not isinstance(node, dict):
          resolved = False
          break
        if key in node:
          node = node[key]
        elif str(key) in node:
          node = node[str(key)]
        else:
          resolved = False
          break
      if not resolved:
        continue
      if _is_orbax_array_metadata(node):
        return False
      if isinstance(node, dict) and "value" in node:
        return True

  def _has_wrapped_value(node: Any) -> bool:
    if not isinstance(node, dict):
      return False
    if "value" in node and _is_array_meta_or_qtensor(node["value"]):
      return True
    return any(_has_wrapped_value(value) for value in node.values())

  return _has_wrapped_value(meta_tree)


@dataclasses.dataclass(frozen=True)
class CheckpointLayout:
  """How parameters are nested inside a checkpoint."""

  is_nnx: bool
  has_base_key: bool = False
  use_value_wrapper: bool = False


def detect_checkpoint_layout(metadata_tree, param_state) -> CheckpointLayout:
  """Detect whether a checkpoint stores Linen or NNX parameters and how NNX leaves are nested."""
  is_nnx_checkpoint = not ("params" in metadata_tree and "params" in metadata_tree.get("params", {}))
  if not is_nnx_checkpoint:
    return CheckpointLayout(is_nnx=False)
  param_metadata = metadata_tree.get("base", metadata_tree)
  return CheckpointLayout(
      is_nnx=True, has_base_key="base" in metadata_tree,
      use_value_wrapper=_checkpoint_uses_value_wrapper(param_metadata, param_state),
  )
