"""Pallas operations for quantizing valid rows of ragged tensors."""

import functools
import math
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _ragged_absmax_kernel(x_ref: Any, *, num_tokens_ref: Any, amax_ref: Any) -> None:
  """Accumulates the absmax of the valid rows into amax_ref."""
  x_val = x_ref[...]
  abs_x = jnp.abs(x_val.astype(jnp.float32))
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  abs_x = jnp.where(
      (rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem), abs_x, 0.0,
  )
  if x_val.ndim == 2:
    block_max = jnp.max(abs_x.reshape(-1, 8, x_val.shape[1]), axis=0)
  else:
    block_max = jnp.max(abs_x, axis=0)
  amax_ref[...] = jnp.maximum(amax_ref[...], block_max)


def _ragged_quantize_kernel(
    x_ref: Any, out_ref: Any, *, num_tokens_ref: Any, scale_ref: Any, fmax: float,
) -> None:
  """Quantizes valid rows using scale_ref."""
  x_val = x_ref[...]
  inv_scale = 1.0 / scale_ref[...].astype(jnp.float32)
  if x_val.ndim == 3:
    inv_scale = inv_scale[None]
  q = jnp.clip(x_val.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(out_ref.dtype)
  block_tokens = x_val.shape[0]
  rem = num_tokens_ref[0] - pl.program_id(0) * block_tokens
  q = jnp.where((rem >= block_tokens) | (jax.lax.broadcasted_iota(jnp.int32, x_val.shape, 0) < rem), q, 0)
  out_ref[...] = q


def _ragged_quantize_pipeline(
    num_tokens_ref: Any, x_ref: Any, scale_ref: Any, out_ref: Any, *, block_size_tokens: int, fmax: float,
) -> None:
  """Runs the quantization kernel over valid tokens."""
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    x_spec = pl.BlockSpec((block_size_tokens, *x_ref.shape[1:]), lambda i: (i,) + (0,) * (x_ref.ndim - 1))
    pltpu.emit_pipeline(
        functools.partial(_ragged_quantize_kernel, num_tokens_ref=num_tokens_ref, scale_ref=scale_ref, fmax=fmax),
        grid=(pl.cdiv(num_tokens, block_size_tokens),), in_specs=[x_spec], out_specs=x_spec,
    )(x_ref, out_ref)


def _ragged_absmax_quantize_pipeline(
    num_tokens_ref: Any, x_ref: Any, out_ref: Any, scale_out_ref: Any, amax_ref: Any,
    *, block_size_tokens: int, fmax: float,
) -> None:
  """Computes scale and quantizes the valid rows of x_ref."""
  scale_out_ref[...] = jnp.ones((1, 1), jnp.float32)
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    amax_ref[...] = jnp.zeros_like(amax_ref)
    num_blocks = pl.cdiv(num_tokens, block_size_tokens)
    x_spec = pl.BlockSpec((block_size_tokens, *x_ref.shape[1:]), lambda i: (i,) + (0,) * (x_ref.ndim - 1))
    pltpu.emit_pipeline(
        functools.partial(_ragged_absmax_kernel, num_tokens_ref=num_tokens_ref, amax_ref=amax_ref),
        grid=(num_blocks,), in_specs=[x_spec],
    )(x_ref)
    amax = jnp.max(amax_ref[...], keepdims=True)
    scale_out_ref[...] = jnp.where(amax > 0, amax / fmax, 1.0)
    pltpu.emit_pipeline(
        functools.partial(_ragged_quantize_kernel, num_tokens_ref=num_tokens_ref, scale_ref=scale_out_ref, fmax=fmax),
        grid=(num_blocks,), in_specs=[x_spec], out_specs=x_spec,
    )(x_ref, out_ref)


def ragged_quantize(
    x: jax.Array, scale: jax.Array, qtype: Any, num_tokens: jax.Array | int, *, block_size_tokens: int | None = None,
) -> jax.Array:
  """Quantizes the first num_tokens rows of x to qtype with scale."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)
  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(_ragged_quantize_pipeline, block_size_tokens=block_size_tokens, fmax=fmax),
      out_shape=jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1, in_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))], out_specs=hbm,
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 64 * 1024 * 1024)),
      name="ragged_quantize",
  )(jnp.asarray(num_tokens, jnp.int32)[None], x, scale.astype(jnp.float32).reshape(1, 1))


def ragged_absmax_quantize(
    x: jax.Array, qtype: Any, num_tokens: jax.Array | int, *, block_size_tokens: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Dynamically quantizes the first num_tokens rows of x to qtype."""
  qtype = jnp.dtype(qtype)
  fmax = float(jnp.finfo(qtype).max)
  if block_size_tokens is None:
    default_block = 512 if x.ndim == 2 else 256
    block_size_tokens = math.gcd(default_block, x.shape[0])
  amax_shape = (8, x.shape[1]) if x.ndim == 2 else x.shape[1:]
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      functools.partial(_ragged_absmax_quantize_pipeline, block_size_tokens=block_size_tokens, fmax=fmax),
      out_shape=(
          jax.ShapeDtypeStruct(x.shape, qtype, manual_axis_type=manual_axis_type),
          jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=manual_axis_type),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1, in_specs=[hbm], out_specs=[hbm, pl.BlockSpec((1, 1), lambda *_: (0, 0))],
          scratch_shapes=[pltpu.VMEM(amax_shape, jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 64 * 1024 * 1024)),
      name="ragged_absmax_quantize",
  )(jnp.asarray(num_tokens, jnp.int32)[None], x)
