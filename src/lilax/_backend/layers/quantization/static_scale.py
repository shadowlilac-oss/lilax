"""Static (fixed range) calibration scales and manual per-tensor quantization."""

import qwix.pallas as qpl

import qwix
from qwix._src.core import numerics

import jax
import jax.numpy as jnp


def is_static_calibration(calibration_method: str | None) -> bool:
  """Whether `calibration_method` is a fixed-range (static scale) calibration."""
  return bool(calibration_method) and calibration_method.lower().startswith("fixed")


def _cast_reduced_from(arr, reduced_arr):
  aval = jax.typeof(reduced_arr)
  # In shard map
  if aval.sharding.mesh.axis_types[0] == jax.sharding.AxisType.Manual:
    for axis in aval.mat.reduced:
      arr = jax.lax.pcast(arr, axis, to="reduced")
    return arr
  # Outside shard map
  return jax.reshard(arr, aval.sharding)


def _make_scale_tensor(scale, arr):
  scale_tensor = jnp.full_like(arr, scale, dtype=jnp.bfloat16)
  return _cast_reduced_from(scale_tensor, arr)


def get_static_scale(qtype: jax.typing.DTypeLike, calibration_method: str) -> float:
  """Extracts the static scale.
  Currently, only symmetric fixed range calibration is supported.
  For symmetric calibration, the calibration_method must be in the format 'fixed,-max,max' or 'fixed,max'.

  Args:
    qtype: The dtype to quantize to.
    calibration_method: A string specifying the calibration method.

  Returns:
    The extracted static scale value.
  """
  if calibration_method is None or not calibration_method.lower().startswith("fixed"):
    raise ValueError(f"Only static scale quantization is supported, got {calibration_method}")

  args = [float(a) for a in calibration_method.split(",")[1:]]
  if len(args) == 1:
    args = [-args[0], args[0]]

  if len(args) != 2 or args[0] + args[1] != 0 or args[1] <= 0:
    raise ValueError(f"Expected format: 'fixed,max' or 'fixed,-max,max'. Got: {calibration_method}")

  qmax = numerics.get_symmetric_bound(qtype)
  scale_val = args[1] / qmax

  # Prevent scale from being 0
  tiny_sqrt = jnp.finfo(jnp.float32).tiny ** 0.5
  if scale_val < tiny_sqrt:
    scale_val = 1.0

  return scale_val


def manual_quantize(tensor: jax.Array, dtype: jax.typing.DTypeLike, calibration_method: str) -> qwix.QArray:
  """Manually quantizes a tensor based on per-tensor scaling with symmetric fixed range calibration.

  Args:
    tensor: The tensor to quantize.
    dtype: The logical type of the quantized value, e.g. jnp.float8_e4m3fn
    calibration_method: A string specifying the calibration method. Currently only support
    symmetric fixed range calibration: Expected format is "fixed,{-max_val},{max_val}".

  Returns:
    A qwix.QArray containing the quantized value and the scale.

  Raises:
    ValueError: If calibration_method is None or has an unexpected format.
  """
  scale = get_static_scale(dtype, calibration_method)
  dtype_max = numerics.get_symmetric_bound(dtype)
  dtype_min = -dtype_max
  # scale must be converted to a tensor because grad has reduced axes.
  scale_tensor = _make_scale_tensor(scale, tensor)
  min_bound = _make_scale_tensor(dtype_min, tensor)
  max_bound = _make_scale_tensor(dtype_max, tensor)
  q_tensor = jnp.clip(tensor / scale_tensor, min_bound, max_bound).astype(dtype)

  # get scale for QArray.
  # per-tensor scaling: same scale for each axis
  scale_shape = [1] * tensor.ndim
  # It must stay fully replicated for the backward pass and Pallas.
  scale_tensor_qpl = jnp.full(scale_shape, scale, dtype=tensor.dtype)
  # wrap in QArray
  return qpl.QArray(qvalue=q_tensor, scale=scale_tensor_qpl)
