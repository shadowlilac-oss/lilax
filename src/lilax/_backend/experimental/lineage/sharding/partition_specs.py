"""Translation of logical PartitionSpecs to physical mesh axes."""

from collections.abc import Mapping

import jax


def physical_pspec(
    logical_pspec: jax.sharding.PartitionSpec,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jax.sharding.PartitionSpec:
  """Translates logical axes in a PartitionSpec to physical mesh axes via axis_mapping."""

  def _map_axis(axis: str | tuple[str, ...] | None) -> str | tuple[str, ...] | None:
    if axis is None:
      return None
    if isinstance(axis, str):
      return axis_mapping.get(axis, axis)
    out = []
    for sub_axis in axis:
      mapped = axis_mapping.get(sub_axis, sub_axis)
      if isinstance(mapped, tuple):
        out.extend(mapped)
      else:
        out.append(mapped)
    return tuple(out)

  def _map_set(s: frozenset[str] | None) -> frozenset[str]:
    if not s:
      return frozenset()
    out = set()
    for p in s:
      mapped = _map_axis(p)
      if isinstance(mapped, tuple):
        out.update(mapped)
      elif mapped is not None:
        out.add(mapped)
    return frozenset(out)

  physical_partitions = tuple(_map_axis(p) for p in logical_pspec.partitions)
  reduced = _map_set(logical_pspec.reduced)
  unreduced = _map_set(logical_pspec.unreduced)
  return jax.sharding.PartitionSpec(*physical_partitions, reduced=reduced, unreduced=unreduced)
