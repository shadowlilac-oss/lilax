"""Quantized collectives for the DeepSeek FP8 batch-split schedule."""

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import qwix.pallas as qpl


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(
        1,
        2,
        3,
    ),
)
def quantized_psum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool) -> jax.Array:
  """Forward: Standard BF16 Reduce-Scatter.

  Backward: Quantized FP8 All-Gather (DeepSeek optimization).

  Args:
    x: The input tensor.
    axis_name: The axis name for the psum_scatter/all_gather operation.
    scatter_dimension: The dimension along which to scatter.
    tiled: Whether the scatter/gather is tiled.

  Returns:
    The result of the reduce-scatter operation.
  """
  return _q_psum_scatter_fwd(x, axis_name, scatter_dimension, tiled)[0]


def _q_psum_scatter_fwd(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool) -> tuple[jax.Array, None]:
  out = jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=tiled)
  return out, None


def _q_psum_scatter_bwd(
    axis_name: str,
    scatter_dimension: int,
    tiled: bool,
    res: Any,
    grads: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  """Backward pass for quantized_psum_scatter.

  Performs a quantized All-Gather of the gradients.

  Args:
    axis_name: The axis name for the all_gather operation.
    scatter_dimension: The dimension along which the scatter occurred in the
      forward pass.
    tiled: Whether the gather is tiled.
    res: The residuals from the forward pass (_q_psum_scatter_fwd), containing
      (axis_name, scatter_dimension, tiled).
    grads: The gradients from the next layer, which are in BF16.

  Returns:
    The dequantized and all-gathered gradients.
  """
  del res
  # --- BACKWARD PASS (Dispatch) ---
  # 'grads' is the BF16 gradient arriving from the next layer.
  # We need to broadcast it back to all devices (All-Gather).

  grads_q = qpl.quantize(
      grads,
      jnp.float8_e5m2,
      channelwise_axes=[0],
  )

  gathered_qvals = jax.lax.all_gather(grads_q.qvalue, axis_name=axis_name, tiled=tiled, axis=scatter_dimension)

  return (qpl.dequantize(dataclasses.replace(grads_q, qvalue=gathered_qvals)),)


quantized_psum_scatter.defvjp(_q_psum_scatter_fwd, _q_psum_scatter_bwd)
