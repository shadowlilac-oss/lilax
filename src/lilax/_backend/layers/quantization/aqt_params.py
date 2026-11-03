"""Matching AQT-quantized variables to the unquantized params they replace."""

from aqt.jax.v2 import aqt_tensor

import jax
from jax.tree_util import tree_flatten_with_path, tree_unflatten


def match_aqt_and_unquantized_param(aqt_params, params):
  """match aqt and unquantized params"""
  aqt_param_flat, aqt_tree_def = jax.tree_util.tree_flatten_with_path(
      aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor)
  )
  param_tree_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  aqt_paths = []
  # Original path of quantized AQT param path.
  param_paths = []

  for aqt_k, _ in aqt_param_flat:
    index = None
    for index, (k, _) in enumerate(param_tree_flat):
      path_depth = len(k)
      # every quantized parameter has AQT.. as the leaf node
      # AqtDotGeneral and AqtEinsum replace leaf node.
      # Therefore, leaf node should be ignored for path matching
      # Note: Aqt only operates on kernels so don't pop bias parameters.
      # Ref: https://github.com/AI-Hypercomputer/maxtext/compare/main...quantize_r1
      if k[: path_depth - 1] == aqt_k[: path_depth - 1] and k[-1].key != "bias":
        aqt_paths.append(aqt_k)
        param_paths.append(k)
        break
    assert index is not None
    # since the parameter is already added, we can delete it.
    param_tree_flat.pop(index)
  return jax.tree_util.tree_unflatten(aqt_tree_def, param_paths)


def _get_aqt_key_paths(aqt_vars, params):
  """Generate a list of paths which have aqt state"""
  aqt_to_unquantized_key_path = match_aqt_and_unquantized_param(aqt_vars, params)
  aqt_key_paths, _ = jax.tree_util.tree_flatten(aqt_to_unquantized_key_path, is_leaf=lambda x: isinstance(x, tuple))
  return list(aqt_key_paths)


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  quantized_param_paths = _get_aqt_key_paths(aqt_vars, params)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in quantized_param_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)
