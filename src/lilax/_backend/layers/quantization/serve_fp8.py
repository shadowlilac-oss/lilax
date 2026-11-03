"""Native FP8 serving compute with pre-quantized FP8 weights and static checkpoint scales."""

import math
from typing import Tuple
from dataclasses import dataclass

import qwix

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from lilax._backend.common.common_types import DType
from lilax._backend.layers.quantization.base import Quantization
from lilax._backend.layers.quantization.static_scale import get_static_scale, is_static_calibration


def infer_scale_granularity(scale_shape: Tuple[int, ...]) -> Tuple[str, int | None]:
  """Infers quantization scheme ('per_tensor', 'per_channel', or 'block_wise') from scale shape."""
  nontrivial_axes = [i for i, s in enumerate(scale_shape) if s > 1]
  if not nontrivial_axes:
    return "per_tensor", None
  if len(nontrivial_axes) == 1:
    return "per_channel", nontrivial_axes[0]
  return "block_wise", None


def native_fp8_dot_general(
    inputs: jnp.ndarray,
    quantized_kernel: jnp.ndarray,
    kernel_scale: jnp.ndarray,
    scale_block_size: int | Tuple[int, ...] | None,
    axis: Tuple[int, ...],
    contract_ind: Tuple[int, ...],
    compute_dtype: DType = jnp.bfloat16,
    precision: jax.lax.PrecisionLike = None,
    out_sharding: NamedSharding | None = None,
    act_calibration_method: str = "absmax",
) -> jnp.ndarray:
  """Computes dot_general with pre-quantized FP8 weights without dequantization."""
  if len(axis) != 1 or len(contract_ind) != 1:
    raise NotImplementedError(
        "native_fp8_dot_general only supports a single contracted axis per "
        f"operand, got axis={axis}, contract_ind={contract_ind}."
    )
  lhs_axis = axis[0]
  rhs_axis = contract_ind[0]

  # Qwix requires scale.ndim == qvalue.ndim. Determine scheme before reshaping minimal-rank scales.
  if kernel_scale.ndim == quantized_kernel.ndim:
    scheme, _ = infer_scale_granularity(kernel_scale.shape)
  elif kernel_scale.size == 1:
    scheme = "per_tensor"
    kernel_scale = kernel_scale.reshape((1,) * quantized_kernel.ndim)
  else:
    # Reshape flattened per-channel scale over output axes.
    scheme = "per_channel"
    out_axes_shape = tuple(d for i, d in enumerate(quantized_kernel.shape) if i != rhs_axis)
    if kernel_scale.size != math.prod(out_axes_shape):
      raise ValueError(
          f"kernel_scale with shape {kernel_scale.shape} does not match the "
          f"kernel's non-contracted output shape {out_axes_shape} (kernel "
          f"shape {quantized_kernel.shape}, contracted axis {rhs_axis})."
      )
    kernel_scale = jnp.expand_dims(kernel_scale.reshape(out_axes_shape), axis=rhs_axis)

  rhs = qwix.QArray(qvalue=quantized_kernel, scale=kernel_scale)
  channelwise_axes = [d for d in range(inputs.ndim) if d != lhs_axis]
  if scheme in ("per_tensor", "per_channel"):
    tiled_axes = {}
  else:
    rhs_block_size = scale_block_size[rhs_axis] if isinstance(scale_block_size, (list, tuple)) else scale_block_size
    tiled_axes = {lhs_axis: rhs_block_size}
  if is_static_calibration(act_calibration_method):
    channelwise_axes = []
    tiled_axes = {}
  lhs = qwix.quantize(
      jnp.asarray(inputs, compute_dtype),
      jnp.float8_e4m3fn,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      calibration_method=act_calibration_method,
  )
  dimension_numbers = (((lhs_axis,), (rhs_axis,)), ((), ()))
  out = qwix.dot_general(lhs, rhs, dimension_numbers, precision=precision, preferred_element_type=jnp.float32)
  out = out.astype(compute_dtype)
  if out_sharding is not None:
    out = jax.lax.with_sharding_constraint(out, out_sharding)
  return out


@dataclass
class ServeFp8WeightQuantization(Quantization):
  """Marks a layer as using native FP8 compute without weight dequantization.

  Weights are FP8 with static scales loaded from the checkpoint. Activations (matmul
  inputs) are quantized to FP8 according to `act_calibration_method`:

    - "absmax" (default): dynamic scale computed from the activation at runtime.
    - "fixed,-A,A" or "fixed,A": static per-tensor scale A / 448, no runtime reduction.

  The setting applies to the dense layers (`native_fp8_dot_general`) and to the routed
  experts on the tpu-inference fused MoE path (`static_act_scale`).
  """

  quant_mode = None
  act_calibration_method: str = "absmax"

  def __post_init__(self):
    if is_static_calibration(self.act_calibration_method):
      # Fail at config time on e.g. an asymmetric range instead of at trace time.
      get_static_scale(jnp.float8_e4m3fn, self.act_calibration_method)

  def static_act_scale(self) -> jax.Array | None:
    """Returns the static per-tensor activation scale as a float32 [1, 1] array.

    Returns None when activations use dynamic scaling.
    """
    if not is_static_calibration(self.act_calibration_method):
      return None
    scale = get_static_scale(jnp.float8_e4m3fn, self.act_calibration_method)
    return jnp.full((1, 1), scale, dtype=jnp.float32)
