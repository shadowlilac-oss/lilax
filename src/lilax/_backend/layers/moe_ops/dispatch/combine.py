"""Combine of expert outputs across expert-parallel shards with an optionally quantized backward all-gather."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import qwix.pallas as qpl


def _all_gather_quantized_payload(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    *,
    axis: int,
    tiled: bool,
    method: str,
    qtype: jnp.dtype = jnp.float8_e5m2,
    num_feature_axes: int = 1,
) -> jax.Array:
  """jax.lax.all_gather with a quantized wire payload; returns dequantized x.dtype.

  Args:
    x: Local shard to gather.
    axis_name: Mesh axis name(s) to gather over.
    axis: Axis of `x` to gather along.
    tiled: Same as `jax.lax.all_gather`.
    method: One of
      'fixed,<b>': static per-tensor scale b/qmax. The scale is identical on all
        shards, so only qvalues are gathered. For E5M2, |x| > ~b saturates and
        |x| < ~1.3e-10*b flushes to zero.
      'rowwise': dynamic per-token absmax scale over the token axes (all but
        the trailing `num_feature_axes`, e.g. (batch, seq) for 3D, (tokens,)
        for 2D), matching the per-row gradient quantization in megablox
        `_bwd_quantize_gradient`. Scales differ per shard, so they are
        gathered alongside qvalues.
    qtype: Wire dtype of the quantized payload.
    num_feature_axes: Number of trailing axes that form one token's features.
      'rowwise' computes one scale per token over these axes, e.g. 2 for the
      (batch, seq, emb // 128, 128) layout of moe_tc_ragged_3d_dispatch.

  Returns:
    The gathered, dequantized array in `x.dtype`.
  """
  static = method.startswith("fixed")
  token_axes = tuple(range(x.ndim - num_feature_axes))
  if not static and method != "rowwise":
    raise ValueError(f"Unsupported method: {method!r}. Supported: 'rowwise', 'fixed,<bound>'.")
  if not static and axis % x.ndim not in token_axes:
    raise ValueError(f"'rowwise' requires gathering along a token axis, got axis={axis}.")

  x_f32 = x.astype(jnp.float32)
  if static:
    x_q = qpl.quantize(
        x_f32,
        qtype=qtype,
        channelwise_axes=(),
        calibration_method=method,
    )
    scale = x_q.scale
  else:  # dynamic rowwise with per-token scale
    x_q = qpl.quantize(
        x_f32,
        qtype=qtype,
        channelwise_axes=token_axes,
        calibration_method="absmax",
    )
    scale = jax.lax.all_gather(x_q.scale, axis_name=axis_name, axis=axis, tiled=tiled)
  gathered_qvals = jax.lax.all_gather(x_q.qvalue, axis_name=axis_name, axis=axis, tiled=tiled)
  return (gathered_qvals.astype(jnp.float32) * scale).astype(x.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _moe_combine_psum_scatter(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    scatter_dimension: int = 0,
    tiled: bool = True,
    bwd_method: str = "",
) -> jax.Array:
  """Evaluates psum_scatter in forward, and quantized All-Gather in backward.

  Shapes:
    Forward:
      x: [global_batch, sequence_length, hidden_dim], where global_batch is
        the full batch across all expert parallelism (EP) shards.
      Output: [local_batch, sequence_length, hidden_dim], scattered along
        scatter_dimension=0 (local_batch = global_batch // num_ep).
    Backward:
      grads: [local_batch, sequence_length, hidden_dim], incoming cotangent.
      Output: [global_batch, sequence_length, hidden_dim], gathered along
        scatter_dimension=0.
    With moe_tc_ragged_3d_dispatch, hidden_dim is split as (hidden_dim // 128, 128).
  """
  return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dimension, tiled=tiled)


def _moe_combine_psum_scatter_fwd(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    scatter_dimension: int = 0,
    tiled: bool = True,
    bwd_method: str = "",
) -> tuple[jax.Array, None]:
  """Custom VJP forward pass for _moe_combine_psum_scatter."""
  return _moe_combine_psum_scatter(x, axis_name, scatter_dimension, tiled, bwd_method), None


def _moe_combine_psum_scatter_bwd(
    axis_name: str | tuple[str, ...],
    scatter_dimension: int,
    tiled: bool,
    bwd_method: str,
    res: Any,
    grads: jax.Array,
) -> tuple[jax.Array]:
  """Custom VJP backward pass for _moe_combine_psum_scatter."""
  if not bwd_method:
    gathered_grads = jax.lax.all_gather(grads, axis_name=axis_name, tiled=tiled, axis=scatter_dimension)
    return (gathered_grads,)

  # The combine tensor is (batch, seq, *features): features are (emb,) or, with moe_tc_ragged_3d_dispatch,
  # (emb // 128, 128). 'rowwise' keeps one scale per token in both layouts.
  gathered_grads = _all_gather_quantized_payload(
      grads, axis_name, axis=scatter_dimension, tiled=tiled, method=bwd_method, num_feature_axes=grads.ndim - 2
  )
  return (gathered_grads,)


_moe_combine_psum_scatter.defvjp(_moe_combine_psum_scatter_fwd, _moe_combine_psum_scatter_bwd)
