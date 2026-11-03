"""Mesh sharding, batch dimension reshaping, and metric extraction utilities for DiLoCo."""

from collections.abc import Sequence

import drjax
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np


def add_diloco_to_sharding(pytree: PyTree) -> PyTree:
  """Prepend the DiLoCo axis to applicable NamedSharding leaves."""

  def map_fn(leaf):
    if isinstance(leaf, jax.sharding.NamedSharding):
      if "diloco" not in leaf.mesh.axis_names:
        return leaf
      new_spec = jax.sharding.PartitionSpec("diloco", *leaf.spec)
      return jax.sharding.NamedSharding(mesh=leaf.mesh, spec=new_spec)
    return leaf

  return jax.tree_util.tree_map(map_fn, pytree)


def reshape_first_axis_with_diloco(num_diloco_replicas: int, pytree: PyTree) -> PyTree:
  """Reshape the first dimension of each array to include a DiLoCo axis."""

  def extend_pspec(
      pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = (),
  ) -> jax.sharding.PartitionSpec:
    if pspec and isinstance(pspec[0], (tuple, list)) and len(pspec[0]) > 0 and pspec[0][0] == "diloco":
      remaining = tuple(pspec[0][1:])
      if len(remaining) == 1:
        return jax.sharding.PartitionSpec("diloco", remaining[0], *pspec[1:])
      elif len(remaining) > 1:
        return jax.sharding.PartitionSpec("diloco", remaining, *pspec[1:])
      else:
        return jax.sharding.PartitionSpec("diloco", *pspec[1:])
    return jax.sharding.PartitionSpec("diloco", *pspec)

  def reshape_for_diloco(arr):
    if not hasattr(arr, "shape"):
      return arr
    if (
        hasattr(arr, "ndim")
        and arr.ndim >= 3
        and arr.shape[0] == num_diloco_replicas
        and hasattr(arr, "sharding")
        and isinstance(arr.sharding, jax.sharding.NamedSharding)
        and arr.sharding.spec
        and arr.sharding.spec[0] == "diloco"
        and isinstance(arr.sharding.spec[0], str)
    ):
      return arr
    batch_dim, *example_shape = arr.shape
    if batch_dim % num_diloco_replicas != 0:
      raise ValueError(f"Batch dimension {batch_dim} is not divisible by num_diloco_replicas {num_diloco_replicas}.")
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    if hasattr(arr, "sharding") and arr.sharding is not None:
      sharding = arr.sharding
      sharding = jax.sharding.NamedSharding(mesh=sharding.mesh, spec=extend_pspec(sharding.spec))
      return jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), sharding)
    return jnp.reshape(arr, shape=diloco_shape)

  return jax.tree.map(reshape_for_diloco, pytree)


def extract_replica_0(metrics: PyTree) -> PyTree:
  """Extract metrics from replica 0 across DiLoCo islands."""

  def select_first_replica(value):
    if not hasattr(value, "shape") or len(value.shape) == 0:
      return value
    replicas = value.shape[0]
    mask = (jnp.arange(replicas) == 0).reshape((replicas,) + (1,) * (value.ndim - 1))
    return drjax.reduce_sum(value * mask)

  return jax.tree.map(select_first_replica, metrics)


def extract_per_island_metrics(metrics: PyTree, num_diloco_replicas: int) -> PyTree:
  """Extract replica 0 metrics and append per-island loss metrics."""
  default_metrics = extract_replica_0(metrics)
  if isinstance(metrics, dict) and "scalar" in metrics and "learning/loss" in metrics["scalar"]:
    for index in range(num_diloco_replicas):
      mask = jnp.arange(num_diloco_replicas) == index
      loss = drjax.reduce_sum(metrics["scalar"]["learning/loss"] * mask)
      default_metrics["scalar"][f"learning/loss_island_{index}"] = loss
  return default_metrics


def split_mesh_along_axis(mesh: jax.sharding.Mesh, axis_name: str = "diloco") -> list[jax.sharding.Mesh]:
  """Split a mesh into submeshes with the given axis removed."""
  if axis_name not in mesh.axis_names:
    raise ValueError(f"Axis {axis_name!r} not found in mesh axis names {mesh.axis_names}.")
  axis_index = mesh.axis_names.index(axis_name)
  keep = [index for index in range(len(mesh.axis_names)) if index != axis_index]
  axis_names = tuple(mesh.axis_names[index] for index in keep)
  axis_types = tuple(mesh.axis_types[index] for index in keep)
  return [
      jax.sharding.Mesh(np.take(mesh.devices, index, axis=axis_index), axis_names, axis_types=axis_types)
      for index in range(mesh.shape[axis_name])
  ]


def remove_mesh_axis_from_rules(
    logical_axis_rules: Sequence[tuple[str, str | Sequence[str] | None]], axis_name: str = "diloco"
) -> list[tuple[str, str | tuple[str, ...] | None]]:
  """Remove a physical mesh axis from logical axis rules."""
  new_rules = []
  for logical_axis, physical_axes in logical_axis_rules:
    if isinstance(physical_axes, str):
      if physical_axes == axis_name:
        continue
    elif physical_axes is not None:
      stripped = tuple(axis for axis in physical_axes if axis != axis_name)
      if physical_axes and not stripped:
        continue
      physical_axes = stripped
    new_rules.append((logical_axis, physical_axes))
  return new_rules
