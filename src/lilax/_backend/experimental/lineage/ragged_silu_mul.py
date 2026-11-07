"""Pallas SiLU-gated multiply over the valid rows of ragged tensors and its backward pass."""

import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from lilax._backend.experimental.lineage import tile_math


def _ragged_silu_mul_kernel(
    x_ref: Any, y_ref: Any, c_ref: Any, out_ref: Any, *, out_scale_ref: Any = None,
) -> None:
  """Kernel body for ragged_silu_mul."""
  act = tile_math.silu_mul_tile(x_ref[...], y_ref[...], c_ref[...], x_ref.dtype)
  if out_scale_ref is not None:
    inv_scale = 1.0 / out_scale_ref[0].astype(jnp.float32)
    out_ref[...] = tile_math.quantize_tile(act, inv_scale, out_ref.dtype)
  else:
    out_ref[...] = act.astype(out_ref.dtype)


def _ragged_silu_mul_x_specs(
    x_offset: jax.Array, x_dtype: jnp.dtype, block_size_tokens: int, block_size_hidden: int, num_hidden_blocks: int,
) -> tuple[pl.BlockSpec, pl.BlockSpec]:
  """Returns block specs of the two gate halves."""
  sublanes = 8 * 4 // jnp.dtype(x_dtype).itemsize
  shape = (pl.Element(block_size_tokens), block_size_hidden)
  row = lambda i: pl.multiple_of(x_offset + i * block_size_tokens, sublanes)
  return (
      pl.BlockSpec(shape, lambda i, j: (row(i), j)),
      pl.BlockSpec(shape, lambda i, j: (row(i), j + num_hidden_blocks)),
  )


def _ragged_silu_mul_fwd_pipeline(
    x_offset_ref: Any, num_tokens_ref: Any, x_ref: Any, c_ref: Any, out_scale_ref: Any, out_ref: Any,
    *, block_size_tokens: int, block_size_hidden: int, num_hidden_blocks: int,
) -> None:
  """Runs the SiLU kernel over valid token blocks."""
  num_tokens = num_tokens_ref[0]

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0], x_ref.dtype, block_size_tokens, block_size_hidden, num_hidden_blocks,
    )
    pltpu.emit_pipeline(
        functools.partial(_ragged_silu_mul_kernel, out_scale_ref=out_scale_ref),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[g0_spec, g1_spec, pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0))],
        out_specs=pl.BlockSpec((block_size_tokens, block_size_hidden), lambda i, j: (i, j)),
    )(x_ref, x_ref, c_ref, out_ref)


def _pallas_ragged_silu_mul(
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"], x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"], num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int, block_size_hidden: int, out_dtype: jt.DTypeLike | None = None,
    out_scale: jt.Float[jax.Array, "1 1"] | None = None,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Pallas call wrapper for ragged_silu_mul forward pass."""
  max_num_tokens = coeffs.shape[0]
  hidden_dim = x.shape[1] // 2
  block_size_hidden = min(block_size_hidden, hidden_dim)
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  target_dtype = jnp.dtype(out_dtype) if out_dtype is not None else x.dtype
  out_scale_smem = out_scale.astype(jnp.float32).reshape(1) if out_scale is not None else None
  out_scale_spec = smem if out_scale_smem is not None else None
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  call_kwargs = {}
  if out_dtype is not None or out_scale is not None:
    call_kwargs["name"] = "ragged_silu_mul_fwd-fp8"
  return pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_fwd_pipeline, block_size_tokens=block_size_tokens,
          block_size_hidden=block_size_hidden, num_hidden_blocks=num_hidden_blocks,
      ),
      out_shape=jax.ShapeDtypeStruct((max_num_tokens, hidden_dim), target_dtype, manual_axis_type=manual_axis_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2, in_specs=[hbm, hbm, out_scale_spec], out_specs=hbm,
      ),
      **call_kwargs,
  )(jnp.asarray(x_offset, jnp.int32)[None], jnp.asarray(num_tokens, jnp.int32)[None], x, coeffs, out_scale_smem)


def _ragged_silu_mul_bwd_kernel(
    g_ref: Any, x_ref: Any, y_ref: Any, c_ref: Any, dx_ref: Any, dy_ref: Any, dc_ref: Any,
    *, g_mask_max: Any = None, num_tokens_ref: Any = None, amax_vmem_ref: Any = None,
) -> None:
  """Kernel body for ragged_silu_mul backward pass."""
  x_val = x_ref[...]
  y_val = y_ref[...]
  g_val = g_ref[...]
  c_val = c_ref[...]
  sig_x = jax.nn.sigmoid(x_val)
  silu_x = x_val * sig_x
  dsilu_x = sig_x * (1.0 + x_val * (1.0 - sig_x))
  if g_mask_max is not None:
    mask_bound = g_mask_max[0] if hasattr(g_mask_max, "ndim") and g_mask_max.ndim > 0 else g_mask_max
    act_bf16 = (silu_x * y_val * c_val).astype(x_val.dtype)
    g_val = jnp.where(jnp.abs(act_bf16) <= mask_bound, g_val, 0).astype(g_val.dtype)
  gc_val = g_val * c_val
  dx = (gc_val * y_val * dsilu_x).astype(dx_ref.dtype)
  dy = (gc_val * silu_x).astype(dy_ref.dtype)
  dx_ref[...] = dx
  dy_ref[...] = dy
  if amax_vmem_ref is not None and num_tokens_ref is not None:
    row_idx = pl.program_id(0) * dx.shape[0] + jax.lax.broadcasted_iota(jnp.int32, (dx.shape[0], 1), 0)
    valid = row_idx < num_tokens_ref[0]
    tile_abs = jnp.where(valid, jnp.maximum(jnp.abs(dx), jnp.abs(dy)), 0)
    tile_max = jnp.max(tile_abs, axis=0).astype(jnp.float32)
    amax_vmem_ref[0] = jnp.maximum(amax_vmem_ref[0], tile_max)

  @pl.when(pl.program_id(1) == 0)
  def _init():
    dc_ref[...] = jnp.zeros_like(dc_ref)

  f32 = jnp.float32
  dc_ref[...] += jnp.sum(
      g_val.astype(f32) * silu_x.astype(f32) * y_val.astype(f32), axis=-1, keepdims=True,
  ).astype(dc_ref.dtype)


def _ragged_silu_mul_bwd_pipeline(
    x_offset_ref: Any, num_tokens_ref: Any, g_ref: Any, x_ref: Any, c_ref: Any, g_mask_max_ref: Any,
    dxy_ref: Any, dc_ref: Any, amax_smem_ref: Any = None, amax_vmem_ref: Any = None,
    *, block_size_tokens: int, block_size_hidden: int, num_hidden_blocks: int, static_g_mask_max: float | None = None,
) -> None:
  """Runs the SiLU backward kernel over valid token blocks."""
  num_tokens = num_tokens_ref[0]
  mask_max = g_mask_max_ref if g_mask_max_ref is not None else static_g_mask_max
  if amax_vmem_ref is not None:
    amax_vmem_ref[...] = jnp.zeros_like(amax_vmem_ref)

  @pl.when(num_tokens > 0)
  def _():
    g0_spec, g1_spec = _ragged_silu_mul_x_specs(
        x_offset_ref[0], x_ref.dtype, block_size_tokens, block_size_hidden, num_hidden_blocks,
    )
    hidden_shape = (block_size_tokens, block_size_hidden)
    hidden_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j))
    dy_spec = pl.BlockSpec(hidden_shape, lambda i, j: (i, j + num_hidden_blocks))
    coeffs_spec = pl.BlockSpec((block_size_tokens, 1), lambda i, j: (i, 0))
    pltpu.emit_pipeline(
        functools.partial(
            _ragged_silu_mul_bwd_kernel, g_mask_max=mask_max, num_tokens_ref=num_tokens_ref,
            amax_vmem_ref=amax_vmem_ref,
        ),
        grid=(pl.cdiv(num_tokens, block_size_tokens), num_hidden_blocks),
        in_specs=[hidden_spec, g0_spec, g1_spec, coeffs_spec], out_specs=[hidden_spec, dy_spec, coeffs_spec],
    )(g_ref, x_ref, x_ref, c_ref, dxy_ref, dxy_ref, dc_ref)
  if amax_smem_ref is not None and amax_vmem_ref is not None:
    amax_smem_ref[0, 0] = jnp.max(amax_vmem_ref[0])


def _pallas_ragged_silu_mul_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"], x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int, coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int, block_size_tokens: int, block_size_hidden: int,
    g_mask_max: jt.Float[jax.Array, ""] | float | None = None, with_absmax: bool = False,
) -> Any:
  """Computes backward pass gradients with Pallas."""
  max_num_tokens, hidden_dim = g.shape
  if (g_mask_max is not None or with_absmax) and block_size_tokens * block_size_hidden > 512 * 1024:
    block_size_tokens = min(block_size_tokens, 512)
  block_size_hidden = min(block_size_hidden, hidden_dim)
  num_hidden_blocks = (hidden_dim + block_size_hidden - 1) // block_size_hidden
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  smem = pl.BlockSpec(memory_space=pltpu.SMEM)
  static_g_mask_max = None
  g_mask_max_smem = None
  if isinstance(g_mask_max, (int, float)):
    static_g_mask_max = float(g_mask_max)
  elif g_mask_max is not None:
    g_mask_max_smem = jnp.asarray(g_mask_max, jnp.float32).reshape(1)
  g_mask_spec = smem if g_mask_max_smem is not None else None
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  dxy_struct = jax.ShapeDtypeStruct((max_num_tokens, 2 * hidden_dim), x.dtype, manual_axis_type=manual_axis_type)
  dc_struct = jax.ShapeDtypeStruct(coeffs.shape, jnp.float32, manual_axis_type=manual_axis_type)
  if with_absmax:
    amax_struct = jax.ShapeDtypeStruct((1, 1), jnp.float32, manual_axis_type=manual_axis_type)
    out_shape = (dxy_struct, dc_struct, amax_struct)
    out_specs = [hbm, hbm, smem]
    scratch_shapes = [pltpu.VMEM((1, block_size_hidden), jnp.float32)]
  else:
    out_shape = (dxy_struct, dc_struct)
    out_specs = [hbm, hbm]
    scratch_shapes = []
  res = pl.pallas_call(
      functools.partial(
          _ragged_silu_mul_bwd_pipeline, block_size_tokens=block_size_tokens, block_size_hidden=block_size_hidden,
          num_hidden_blocks=num_hidden_blocks, static_g_mask_max=static_g_mask_max,
      ),
      out_shape=out_shape,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2, in_specs=[hbm, hbm, hbm, g_mask_spec], out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      name="ragged_silu_mul_bwd-absmax" if with_absmax else None,
  )(jnp.asarray(x_offset, jnp.int32)[None], jnp.asarray(num_tokens, jnp.int32)[None], g, x, coeffs, g_mask_max_smem)
  if with_absmax:
    dxy, dc, amax = res
    return dxy, dc.astype(coeffs.dtype), amax
  dxy, dc = res
  return dxy, dc.astype(coeffs.dtype)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_silu_mul_fwd(
    x: jt.Num[jax.Array, "x_rows double_hidden_dim"], x_offset: jt.Int[jax.Array, ""] | int,
    coeffs: jt.Num[jax.Array, "max_num_tokens 1"], num_tokens: jt.Num[jax.Array, ""] | int,
    *, block_size_tokens: int = 128, block_size_hidden: int = 128, out_dtype: jt.DTypeLike | None = None,
    out_scale: jt.Float[jax.Array, "1 1"] | None = None,
) -> jt.Num[jax.Array, "max_num_tokens hidden_dim"]:
  """Computes ragged silu(g0) * g1 * coeffs using Pallas."""
  return _pallas_ragged_silu_mul(
      x, x_offset, coeffs, num_tokens, block_size_tokens, block_size_hidden, out_dtype=out_dtype, out_scale=out_scale,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_silu_mul_bwd(
    g: jt.Num[jax.Array, "max_num_tokens hidden_dim"], x: jt.Num[jax.Array, "x_rows double_hidden_dim"],
    x_offset: jt.Int[jax.Array, ""] | int, coeffs: jt.Num[jax.Array, "max_num_tokens 1"],
    num_tokens: jt.Num[jax.Array, ""] | int, *, block_size_tokens: int = 128, block_size_hidden: int = 128,
    g_mask_max: jt.Float[jax.Array, ""] | float | None = None, with_absmax: bool = False,
) -> Any:
  """Computes the backward pass for ragged_silu_mul_fwd."""
  return _pallas_ragged_silu_mul_bwd(
      g, x, x_offset, coeffs, num_tokens, block_size_tokens, block_size_hidden,
      g_mask_max=g_mask_max, with_absmax=with_absmax,
  )
