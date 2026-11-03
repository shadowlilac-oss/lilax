"""Placement of train-state leaves onto the engine mesh."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax


def mesh_sharding(mesh: jax.sharding.Mesh, leaf: Any) -> jax.sharding.Sharding | None:
  if leaf is None:
    return None
  leaf_sharding = getattr(leaf, "sharding", None)
  if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == mesh:
    return leaf_sharding
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def place_state_on_mesh(
    mesh: jax.sharding.Mesh, state_pure: Any, place_leaf: Callable[[Any, jax.sharding.Sharding], Any]
) -> Any:
  """Returns `state_pure` with off-mesh leaves placed on `mesh`, or None when every leaf is already there."""
  moved = False

  def place(leaf):
    nonlocal moved
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
      return leaf
    leaf_sharding = getattr(leaf, "sharding", None)
    if isinstance(leaf_sharding, jax.sharding.NamedSharding) and leaf_sharding.mesh == mesh:
      return leaf
    moved = True
    return place_leaf(leaf, mesh_sharding(mesh, leaf))

  placed = jax.tree.map(place, state_pure)
  return placed if moved else None
