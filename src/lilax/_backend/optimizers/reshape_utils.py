"""Utilities for reshaping and resharding tensors for Muon."""

import math
from typing import Callable, Iterable

import jax
import jax.numpy as jnp

ReshapeFn = Callable[[jax.Array], jax.Array]


def get_dim_mesh_axes(sharding: jax.sharding.NamedSharding | None, dim: int) -> tuple[str, ...]:
  if sharding is None or dim < 0 or dim >= len(sharding.spec):
    return ()
  axis = sharding.spec[dim]
  if axis is None:
    return ()
  elif isinstance(axis, str):
    return (axis,)
  return tuple(axis)


def get_flat_mesh_axes(sharding: jax.sharding.NamedSharding | None, dims: Iterable[int]) -> tuple[str, ...]:
  if sharding is None:
    return ()
  axes = []
  for dim in dims:
    axes.extend(get_dim_mesh_axes(sharding, dim))
  return tuple(axes)


def get_reshape_fns(
    x: jax.Array,
    reduction_axes: tuple[int, ...],
    output_axes: tuple[int, ...],
    sharding: jax.sharding.NamedSharding | None = None,
    use_all_to_all: bool = True,
) -> tuple[ReshapeFn, ReshapeFn, jax.sharding.NamedSharding | None]:
  if sharding is not None:
    is_explicit_axes(sharding)
  shape = x.shape
  ndim = len(shape)
  reduction_axes = tuple(ax % ndim for ax in reduction_axes)
  output_axes = tuple(ax % ndim for ax in output_axes)
  batch_axes = tuple(sorted(set(range(ndim)) - set(reduction_axes) - set(output_axes)))
  reduction_size = math.prod(shape[ax] for ax in reduction_axes)
  output_size = math.prod(shape[ax] for ax in output_axes)
  if output_size > reduction_size:
    row_axes, col_axes = reduction_axes, output_axes
    row_size, column_size = reduction_size, output_size
  else:
    row_axes, col_axes = output_axes, reduction_axes
    row_size, column_size = output_size, reduction_size
  sharded_batch_axes = tuple(ax for ax in batch_axes if get_dim_mesh_axes(sharding, ax))
  unsharded_batch_axes = tuple(ax for ax in batch_axes if not get_dim_mesh_axes(sharding, ax))
  matrix_mesh_axes = get_flat_mesh_axes(sharding, row_axes + col_axes)
  use_all_to_all = bool(use_all_to_all and unsharded_batch_axes and matrix_mesh_axes)
  sharded_batch_size = math.prod(shape[ax] for ax in sharded_batch_axes)
  unsharded_batch_size = math.prod(shape[ax] for ax in unsharded_batch_axes)
  perm = sharded_batch_axes + unsharded_batch_axes + row_axes + col_axes
  permuted_shape = tuple(shape[ax] for ax in perm)
  inv_perm = [0] * ndim
  for i, p in enumerate(perm):
    inv_perm[p] = i
  padding = 0
  if use_all_to_all and sharding is not None:
    mesh_shape = getattr(sharding.mesh, "shape", {})
    shards = math.prod(mesh_shape[s] for s in matrix_mesh_axes if s in mesh_shape)
    if shards > 1:
      padding = -unsharded_batch_size % shards
  flat_sharding = None
  sharded_batch_mesh_axes = None
  if sharding is not None:
    sharded_batch_mesh_axes = get_flat_mesh_axes(sharding, sharded_batch_axes) or None
    orig_row_mesh_axes = get_flat_mesh_axes(sharding, row_axes) or None
    orig_col_mesh_axes = get_flat_mesh_axes(sharding, col_axes) or None
    if use_all_to_all:
      unsharded_batch_mesh_axes = matrix_mesh_axes or None
      row_mesh_axes = None
      col_mesh_axes = None
    else:
      unsharded_batch_mesh_axes = None
      row_mesh_axes = orig_row_mesh_axes
      col_mesh_axes = orig_col_mesh_axes
    flat_spec = jax.sharding.PartitionSpec(
        sharded_batch_mesh_axes, unsharded_batch_mesh_axes, row_mesh_axes, col_mesh_axes,
    )
    flat_sharding = jax.sharding.NamedSharding(sharding.mesh, flat_spec)

  def reshape_fn(x: jax.Array) -> jax.Array:
    x_flat = jnp.transpose(x, perm).reshape((sharded_batch_size, unsharded_batch_size, row_size, column_size))
    if padding > 0:
      pad_shape = (sharded_batch_size, padding, row_size, column_size)
      zeros = jnp.zeros(pad_shape, dtype=x_flat.dtype)
      x_flat = jnp.concatenate([x_flat, zeros], axis=1)
    if flat_sharding is not None:
      x_flat = reshard_or_constrain(x_flat, flat_sharding)
    return x_flat

  def unreshape_fn(x_flat: jax.Array) -> jax.Array:
    if use_all_to_all:
      if padding > 0:
        pre_all_to_all_spec = jax.sharding.PartitionSpec(
            sharded_batch_mesh_axes, None, orig_row_mesh_axes, orig_col_mesh_axes,
        )
        pre_all_to_all_sharding = (
            jax.sharding.NamedSharding(sharding.mesh, pre_all_to_all_spec) if sharding is not None else None
        )
        if pre_all_to_all_sharding is not None:
          x_flat = reshard_or_constrain(x_flat, pre_all_to_all_sharding)
        x_flat = jax.lax.slice(
            x_flat, (0, 0, 0, 0), (sharded_batch_size, unsharded_batch_size, row_size, column_size),
        )
    x_unreshaped = jnp.reshape(x_flat, permuted_shape).transpose(inv_perm)
    if sharding is not None:
      x_unreshaped = reshard_or_constrain(x_unreshaped, sharding)
    return x_unreshaped

  return reshape_fn, unreshape_fn, flat_sharding


def is_explicit_axes(sharding: jax.sharding.NamedSharding | None) -> bool:
  if sharding is None:
    return False
  axis_types = tuple(sharding.mesh.axis_types)
  if not axis_types:
    return False
  explicit_count = sum(1 for t in axis_types if t == jax.sharding.AxisType.Explicit)
  if 0 < explicit_count < len(axis_types):
    raise ValueError(
        "Mixed mesh axis types (both Explicit and Auto/Unconstrained) are not" f" supported. Found {axis_types=}."
    )
  return explicit_count == len(axis_types)


def reshard_or_constrain(x: jax.Array, target_sharding: jax.sharding.NamedSharding) -> jax.Array:
  if is_explicit_axes(target_sharding):
    return jax.reshard(x, target_sharding)
  return jax.lax.with_sharding_constraint(x, target_sharding)
