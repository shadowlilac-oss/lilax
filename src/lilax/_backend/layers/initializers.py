"""Initializers."""

from typing import Callable

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

from lilax._backend.common.common_types import Array, DType, Shape, PRNGKey, is_fp8_dtype

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = int | tuple[int, ...]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]


def _default_embed_init(key, shape, dtype=jnp.float32):
  target_dtype = dtype
  sample_dtype = jnp.float32 if is_fp8_dtype(dtype) else dtype
  fn = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
  return fn(key, shape, sample_dtype).astype(target_dtype)


default_embed_init = _default_embed_init
default_bias_init = jax.nn.initializers.constant(0.0)
default_scalar_init = jax.nn.initializers.constant(0.01)


def nd_dense_init(scale, mode, distribution):
  """Creates a variance-scaling initializer with dynamic in/out axes."""

  def init_fn(key, shape, dtype, in_axis, out_axis):
    target_dtype = dtype
    sample_dtype = jnp.float32 if is_fp8_dtype(dtype) else dtype
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, sample_dtype).astype(target_dtype)

  return init_fn


def variable_to_logically_partitioned(variable: nnx.Variable):
  """Wraps an NNX variable's value in `nn.LogicallyPartitioned`."""
  val = variable.get_value()
  if isinstance(val, aqt_tensor.QTensor):
    return val

  if variable.type.__name__ == "_overwrite_with_gradient":
    return val

  metadata = variable.get_metadata()
  out_sharding = None
  if "out_sharding" in metadata:
    out_sharding = metadata["out_sharding"]
  elif "sharding_names" in metadata:
    out_sharding = metadata["sharding_names"]
  elif "sharding" in metadata:
    out_sharding = metadata["sharding"]

  if out_sharding is not None:
    if nnx.PARTITION_NAME in metadata:
      partition_name = metadata[nnx.PARTITION_NAME]
      scan_axis = metadata.get("param_scan_axis", 0) if variable.type == nnx.Param else 0

      sharding_list = [out_sharding] if isinstance(out_sharding, str) else list(out_sharding)
      if partition_name not in sharding_list:
        sharding_list.insert(scan_axis, partition_name)

      out_sharding = tuple(sharding_list)

    return nn.LogicallyPartitioned(val, out_sharding, mesh=metadata.get("mesh"), rules=metadata.get("rules"))
  else:
    return val
