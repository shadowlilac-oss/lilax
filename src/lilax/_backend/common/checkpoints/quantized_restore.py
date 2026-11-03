"""Build restore targets and dequantize checkpoint weights with scales."""

from typing import Any

import jax
import jax.numpy as jnp
from lilax._backend.layers.quantization.weights import dequantize_weight
from lilax._backend.distributed import sharding as maxtext_sharding


def _scale_sharding(weight_sharding, scale_shape):
  mesh = getattr(weight_sharding, "mesh", None)
  if mesh is None:
    return None
  spec = getattr(weight_sharding, "spec", None)
  scale_shape = tuple(scale_shape)
  if spec is None or len(spec) != len(scale_shape):
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  try:
    return jax.sharding.NamedSharding(mesh, maxtext_sharding.adjust_pspec_for_indivisible_shapes(spec, scale_shape, mesh))
  except Exception:
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _augment_target_with_scales(want_node: Any, meta_node: Any) -> Any:
  if not isinstance(want_node, dict) or not isinstance(meta_node, dict):
    return want_node
  augmented = {}
  for key, value in want_node.items():
    scale_key = f"{key}_scale"
    if isinstance(value, dict):
      augmented[key] = _augment_target_with_scales(value, meta_node.get(key))
    elif scale_key not in want_node and scale_key in meta_node:
      meta_scale = meta_node[scale_key]
      meta_param = meta_node.get(key)
      param_dtype = getattr(meta_param, "dtype", getattr(value, "dtype", jnp.bfloat16))
      weight_sharding = getattr(value, "sharding", None)
      augmented[key] = jax.ShapeDtypeStruct(shape=value.shape, dtype=param_dtype, sharding=weight_sharding)
      augmented[scale_key] = jax.ShapeDtypeStruct(
          shape=meta_scale.shape, dtype=meta_scale.dtype, sharding=_scale_sharding(weight_sharding, meta_scale.shape)
      )
    else:
      augmented[key] = value
  return augmented


def _augment_want_with_scales(want: Any, stored: Any, is_nnx: bool, restore_key: str) -> Any:
  if not isinstance(stored, dict) or not isinstance(want, dict):
    return want
  meta_weights = stored.get(restore_key)
  if restore_key == "params" and is_nnx and isinstance(meta_weights, dict):
    meta_weights = meta_weights.get("params")
  if not is_nnx and "params" in want and isinstance(meta_weights, dict) and "params" not in meta_weights:
    return {"params": _augment_target_with_scales(want["params"], meta_weights)}
  return _augment_target_with_scales(want, meta_weights)


def maybe_dequantize_restored_params(restored_weights: Any, expected_param_pytree: Any) -> Any:
  if not isinstance(restored_weights, dict):
    return restored_weights
  expected_dict = expected_param_pytree if isinstance(expected_param_pytree, dict) else {}
  scales_to_drop = {
      f"{key}_scale" for key in restored_weights if f"{key}_scale" in restored_weights and f"{key}_scale" not in expected_dict
  }
  out = {}
  for key, value in restored_weights.items():
    if key in scales_to_drop:
      continue
    scale_key = f"{key}_scale"
    if scale_key in scales_to_drop:
      target_param = expected_dict.get(key)
      target_dtype = getattr(target_param, "dtype", jnp.bfloat16)
      out[key] = dequantize_weight(value, restored_weights[scale_key], compute_dtype=target_dtype)
    elif isinstance(value, dict):
      out[key] = maybe_dequantize_restored_params(value, expected_dict.get(key))
    else:
      out[key] = value
  return out
