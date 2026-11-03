"""Zero-1: sharding the optimizer state and its parameter view over the data axis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from absl import logging
from flax import nnx
import jax
from lilax._backend.common import common_types
from lilax._backend.training_engine import pure_state
from lilax._backend.training_engine import state_placement
from lilax._backend.training_engine.gradient_sharding import DATA_AXIS
from lilax._backend.utils import sharding

_ZERO1_DECLINED_WARNING = (
    "`shard_optimizer_over_data` (Zero-1) is set, but this engine cannot honour it (%s), so the "
    "optimizer state stays replicated over the data axis. Logged once per engine instance."
)


def zero1_active(config: Any, mesh: Any) -> str | None:
  if not getattr(config, "shard_optimizer_over_data", False):
    return "it is not enabled"
  if getattr(config, "shard_mode", None) != common_types.ShardMode.EXPLICIT:
    return "it needs shard_mode=explicit"
  if mesh is None:
    return "the engine has no mesh"
  if mesh.shape.get(DATA_AXIS, 1) <= 1:
    return f"the mesh has no {DATA_AXIS!r} axis to shard the optimizer over"
  if any(axis_type != jax.sharding.AxisType.Explicit for axis_type in mesh.axis_types):
    return "the mesh has non-Explicit axes"
  return None


def zero1_sharding(mesh: Any, aval: Any, base: jax.sharding.NamedSharding | None) -> jax.sharding.NamedSharding | None:
  if base is None or not hasattr(aval, "shape"):
    return None
  try:
    target = sharding.add_data_to_sharding(mesh, (), aval, base)
  except AssertionError:
    return None
  return None if target == base else target


def shard_optimizer_state(
    mesh: Any, state_pure: Any, place_leaf: Callable[[Any, jax.sharding.Sharding], Any]
) -> Any:
  """Returns the optimizer subtree placed over the data axis, or None when nothing moves."""
  if pure_state.OPTIMIZER_STATE_KEY not in state_pure:
    return None
  moved = False

  def place(leaf):
    nonlocal moved
    target = zero1_sharding(mesh, leaf, state_placement.mesh_sharding(mesh, leaf))
    if target is None:
      return leaf
    moved = True
    return place_leaf(leaf, target)

  optimizer_pure = jax.tree.map(place, state_pure[pure_state.OPTIMIZER_STATE_KEY])
  return optimizer_pure if moved else None


def reshard_model_params(state_pure: Any, params_shardings: Any) -> Any:
  params_pure, rest_pure = nnx.split_state(state_pure[pure_state.MODEL_STATE_KEY], nnx.Param, ...)
  params_pure = jax.tree.map(jax.sharding.reshard, params_pure, params_shardings)
  return pure_state.with_model_state(state_pure, nnx.merge_state(params_pure, rest_pure))


class Zero1Planner:
  """Decides the Zero-1 parameter shardings, saying once when Zero-1 was asked for and cannot run."""

  def __init__(self, config: Any) -> None:
    self._config = config
    self._warned = False

  def note_declined(self, reason: str) -> None:
    if getattr(self._config, "shard_optimizer_over_data", False) and not self._warned:
      self._warned = True
      logging.warning(_ZERO1_DECLINED_WARNING, reason)

  def params_shardings(self, mesh: Any, params_pure: Any, params_shardings: Any) -> Any:
    declined = zero1_active(self._config, mesh)
    if declined is not None:
      self.note_declined(declined)
      return None

    def target(leaf, base):
      sharded = zero1_sharding(mesh, leaf, base)
      return base if sharded is None else sharded

    return jax.tree.map(target, params_pure, params_shardings)
