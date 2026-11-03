"""Gradient and parameter sharding plan for the engine kernels, including the deferred data-axis all-reduce."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
from lilax._backend.common import common_types
from lilax._backend.distributed import sharding

DATA_AXIS = "data"


def _tag_sharding(named_sharding: jax.sharding.NamedSharding, field: str) -> jax.sharding.NamedSharding:
  if DATA_AXIS in sharding.get_mesh_axes_used_by_tensor_spec(named_sharding.spec.partitions):
    return named_sharding
  return named_sharding.update(spec=named_sharding.spec.update(**{field: {DATA_AXIS}}))


def deferred_all_reduce_shardings(config: Any, mesh: Any, params_shardings: Any) -> tuple[Any, Any]:
  if getattr(config, "shard_mode", None) != common_types.ShardMode.EXPLICIT:
    return None, None
  if mesh is None or mesh.shape.get(DATA_AXIS, 1) <= 1:
    return None, None
  if any(axis_type != jax.sharding.AxisType.Explicit for axis_type in mesh.axis_types):
    return None, None
  if any(size > 1 for axis, size in mesh.shape.items() if axis != DATA_AXIS):
    return None, None
  try:
    batch_axes = sharding.batch_mesh_axes(mesh, rules=config.logical_axis_rules)
  except (KeyError, ValueError, IndexError):
    return None, None
  if batch_axes != frozenset({DATA_AXIS}):
    return None, None
  return (
      jax.tree.map(lambda s: _tag_sharding(s, "reduced"), params_shardings),
      jax.tree.map(lambda s: _tag_sharding(s, "unreduced"), params_shardings),
  )


def conform_accumulator(value: Any, target: jax.sharding.NamedSharding) -> Any:
  current = getattr(value, "sharding", None)
  if current == target:
    return value
  if getattr(current, "spec", None) is not None and current.spec.unreduced:
    return jax.sharding.reshard(value, target)
  return jax.device_put(value, target)


@dataclasses.dataclass(frozen=True)
class GradientShardingPlan:
  """Shardings the fwd/bwd and update kernels reshard parameters and gradients onto; all None without a mesh."""

  reduced_params: Any = None
  grads: Any = None
  plain_grads: Any = None
  zero1_params: Any = None
  gathered_params: Any = None


def plan_gradient_shardings(
    params_shardings: Any, reduced_params: Any, unreduced_grads: Any, zero1_params: Any
) -> GradientShardingPlan:
  return GradientShardingPlan(
      reduced_params=reduced_params,
      grads=params_shardings if unreduced_grads is None else unreduced_grads,
      plain_grads=None if unreduced_grads is None else params_shardings,
      zero1_params=zero1_params,
      gathered_params=params_shardings if zero1_params is not None else None,
  )
