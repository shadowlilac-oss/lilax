"""Ragged tensor layout operations."""

import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import jaxtyping as jt
import typeguard


def _ragged_flatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for flattening [X, D0, D1] -> [X, D]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, (x_val.shape[0], -1))


def _pallas_ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"],
    num_tokens: jt.Num[jax.Array, ""] | int,
    block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Pallas call wrapper for flattening [X, D0, D1] -> [X, D]."""
  max_num_tokens, d0, d1 = x.shape
  d = d0 * d1
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [pl.BlockSpec(block_shape=(block_size_tokens, d0, d1), index_map=lambda i: (i, 0, 0))]
  out_spec = pl.BlockSpec(block_shape=(block_size_tokens, d), index_map=lambda i: (i, 0))
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_flatten_kernel,
      out_shape=jax.ShapeDtypeStruct((max_num_tokens, d), x.dtype, manual_axis_type=manual_axis_type),
      in_specs=in_specs, out_specs=out_spec, grid=grid,
  )(x)


def _ragged_unflatten_kernel(x_ref: Any, out_ref: Any) -> None:
  """Kernel body for unflattening [X, D] -> [X, D0, D1]."""
  x_val = x_ref[...]
  out_ref[...] = jnp.reshape(x_val, out_ref.shape)


def _pallas_ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"], shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int, block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Pallas call wrapper for unflattening [X, D] -> [X, D0, D1]."""
  max_num_tokens, d = x.shape
  d0, d1 = shape
  assert d == d0 * d1, f"Shape mismatch: {d} != {d0} * {d1}"
  grid = ((num_tokens + block_size_tokens - 1) // block_size_tokens,)
  in_specs = [pl.BlockSpec(block_shape=(block_size_tokens, d), index_map=lambda i: (i, 0))]
  out_spec = pl.BlockSpec(block_shape=(block_size_tokens, d0, d1), index_map=lambda i: (i, 0, 0))
  manual_axis_type = getattr(jax.typeof(x), "manual_axis_type", None)
  return pl.pallas_call(
      _ragged_unflatten_kernel,
      out_shape=jax.ShapeDtypeStruct((max_num_tokens, d0, d1), x.dtype, manual_axis_type=manual_axis_type),
      in_specs=in_specs, out_specs=out_spec, grid=grid,
  )(x)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _ragged_flatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"], num_tokens: jt.Num[jax.Array, ""] | int, block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  return _ragged_flatten_fwd(x, num_tokens, block_size_tokens)[0]


def _ragged_flatten_fwd(x, num_tokens, block_size_tokens):
  out = _pallas_ragged_flatten(x, num_tokens, block_size_tokens)
  return out, (num_tokens, x.shape[1:])


def _ragged_flatten_bwd(block_size_tokens, res, g):
  num_tokens, shape = res
  dx = _pallas_ragged_unflatten(g, shape, num_tokens, block_size_tokens)
  return dx, None


_ragged_flatten_vjp.defvjp(_ragged_flatten_fwd, _ragged_flatten_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 3))
def _ragged_unflatten_vjp(
    x: jt.Num[jax.Array, "max_num_tokens d"], shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int, block_size_tokens: int,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  return _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens)[0]


def _ragged_unflatten_fwd(x, shape, num_tokens, block_size_tokens):
  out = _pallas_ragged_unflatten(x, shape, num_tokens, block_size_tokens)
  return out, (num_tokens,)


def _ragged_unflatten_bwd(shape, block_size_tokens, res, g):
  del shape
  (num_tokens,) = res
  dx = _pallas_ragged_flatten(g, num_tokens, block_size_tokens)
  return dx, None


_ragged_unflatten_vjp.defvjp(_ragged_unflatten_fwd, _ragged_unflatten_bwd)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_flatten(
    x: jt.Num[jax.Array, "max_num_tokens d0 d1"], num_tokens: jt.Num[jax.Array, ""] | int,
    *, block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0*d1"]:
  """Flattens the first num_tokens of a ragged tensor using Pallas."""
  return _ragged_flatten_vjp(x, num_tokens, block_size_tokens)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def ragged_unflatten(
    x: jt.Num[jax.Array, "max_num_tokens d"], shape: tuple[int, int],
    num_tokens: jt.Num[jax.Array, ""] | int, *, block_size_tokens: int = 128,
) -> jt.Num[jax.Array, "max_num_tokens d0 d1"]:
  """Unflattens the first num_tokens of a ragged tensor using Pallas."""
  return _ragged_unflatten_vjp(x, shape, num_tokens, block_size_tokens)

