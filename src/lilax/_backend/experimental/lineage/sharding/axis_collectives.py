"""All-gathers along mesh axes and the reductions of their cotangents."""

from collections.abc import Callable, Collection, Mapping
import dataclasses
import math
from typing import Any, Sequence

import jax
import jax.experimental.compute_on
import jaxtyping as jt
import typeguard


def _merged_gather_dims(
    local_shape: Sequence[int],
    lead: tuple[int, tuple[str, ...], tuple[str, ...]],
    trail: tuple[int, tuple[str, ...], tuple[str, ...]],
) -> bool:
  """Whether two collected dimensions can share a merged all-gather."""
  lead_dim, lead_part, lead_axes = lead
  trail_dim, trail_part, trail_axes = trail
  return (
      trail_dim == lead_dim + 1
      and local_shape[lead_dim] == 1
      and tuple(lead_part[len(lead_part) - len(lead_axes) :]) == lead_axes
      and tuple(trail_part) == trail_axes
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_along_axis(
    xs: jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]],
    axis_name: str | Sequence[str] | jt.PyTree[Any],
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    *,
    compute_type: str | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]]:
  """All-gathers each array along the given axis names if it is sharded along them."""

  def _to_tuple(m: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(m, str):
      return (m,)
    return tuple(m)

  def _is_axis_spec(x: Any) -> bool:
    return isinstance(x, str) or (
        isinstance(x, (tuple, list)) and not hasattr(x, "_fields") and all(isinstance(e, str) for e in x)
    )

  def _get_physical_axes(leaf_axis_name: str | Sequence[str]) -> tuple[tuple[str, ...], frozenset[str]]:
    names: list[str] = []
    for name in _to_tuple(leaf_axis_name):
      for ax in _to_tuple(axis_mapping.get(name, name)):
        if ax not in names:
          names.append(ax)
    return tuple(names), frozenset(names)

  leaves = jax.tree.leaves(xs)
  if not leaves:
    return xs
  physical_axes_tree = jax.tree.map(
      lambda spec, sub_xs: jax.tree.map(lambda _: _get_physical_axes(spec), sub_xs),
      axis_name,
      xs,
      is_leaf=_is_axis_spec,
  )
  flat_physical_axes = jax.tree.leaves(
      physical_axes_tree,
      is_leaf=lambda x: (isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], frozenset)),
  )
  if not any(names for names, _ in flat_physical_axes):
    return xs
  mesh = jax.typeof(leaves[0]).sharding.mesh

  def _remove_axis_names(
      partitions: tuple[str | tuple[str, ...] | None, ...], physical_axis_set: frozenset[str]
  ) -> tuple[str | tuple[str, ...] | None, ...]:
    out = []
    for a in partitions:
      if a is None:
        out.append(None)
      elif isinstance(a, str):
        out.append(a if a not in physical_axis_set else None)
      else:
        out.append(tuple(b for b in a if b not in physical_axis_set))
    return tuple(out)

  def _get_in_spec(arr):
    return jax.typeof(arr).sharding.spec

  def _get_target_spec(in_spec: jax.sharding.PartitionSpec, physical_axes: tuple[tuple[str, ...], frozenset[str]]):
    _, physical_axis_set = physical_axes
    return jax.sharding.PartitionSpec(
        *_remove_axis_names(in_spec.partitions, physical_axis_set),
        reduced=in_spec.reduced | physical_axis_set,
        unreduced=in_spec.unreduced,
    )

  in_specs = jax.tree.map(_get_in_spec, xs)
  out_specs = jax.tree.map(_get_target_spec, in_specs, physical_axes_tree)

  def _collect_leaf(arr, in_spec: jax.sharding.PartitionSpec, physical_axes: tuple[tuple[str, ...], frozenset[str]]):
    physical_axis_names, physical_axis_set = physical_axes
    sharded_axes = set()
    gathers = []
    for dim_idx, part in enumerate(in_spec.partitions):
      if part is None:
        continue
      part_axes = (part,) if isinstance(part, str) else part
      sharded_axes.update(part_axes)
      gather_axes = tuple(ax for ax in part_axes if ax in physical_axis_set)
      if gather_axes:
        gathers.append((dim_idx, part_axes, gather_axes))
    i = 0
    while i < len(gathers):
      dim_idx, _, gather_axes = gathers[i]
      if i + 1 < len(gathers) and _merged_gather_dims(arr.shape, gathers[i], gathers[i + 1]):
        next_dim, _, next_axes = gathers[i + 1]
        shape = arr.shape
        merged = arr.reshape(*shape[:dim_idx], shape[dim_idx] * shape[next_dim], *shape[next_dim + 1 :])
        merged = jax.lax.all_gather(merged, gather_axes + next_axes, axis=dim_idx, tiled=True, to="reduced")
        gathered_lead = math.prod(jax.lax.axis_size(a) for a in gather_axes)
        arr = merged.reshape(*shape[:dim_idx], gathered_lead, -1, *shape[next_dim + 1 :])
        i += 2
        continue
      arr = jax.lax.all_gather(arr, gather_axes, axis=dim_idx, tiled=True, to="reduced")
      i += 1
    for ax in physical_axis_names:
      if ax not in sharded_axes and ax not in in_spec.reduced and ax not in in_spec.unreduced:
        arr = jax.lax.pcast(arr, ax, to="reduced")
    return arr

  def _collect_pytree(xs_local):
    return jax.tree.map(_collect_leaf, xs_local, in_specs, physical_axes_tree)

  if compute_type is not None:
    _collect_pytree = jax.experimental.compute_on.compute_on(
        compute_type=compute_type, out_memory_spaces=jax.memory.Space.Device, compiler_options=compiler_options
    )(_collect_pytree)
  return jax.shard_map(_collect_pytree, mesh=mesh, in_specs=(in_specs,), out_specs=out_specs)(xs)


@dataclasses.dataclass(frozen=True)
class DeferredReduce:
  """A reduction that the caller runs inside its own shard_map."""

  operand: list[jax.Array]
  local_fn: Callable[[list[jax.Array]], list[jax.Array]]
  out_specs: list[jax.sharding.PartitionSpec]
  finish: Callable[[list[jax.Array]], Any]


@jt.jaxtyped(typechecker=typeguard.typechecked)
def reduce_along_axis(
    xs: jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]],
    axis_name: str | Sequence[str] | jt.PyTree[Any],
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    like: jt.Shaped[jax.Array, "..."] | jt.PyTree[jt.Shaped[jax.Array, "..."]],
    *,
    axes: Collection[str] | None = None,
    split_scatters: bool | jt.PyTree[bool] = False,
    defer: bool = False,
    compute_type: str | None = None,
    compiler_options: dict[str, Any] | None = None,
) -> jt.Num[jax.Array, "*input_dims"] | jt.PyTree[jt.Num[jax.Array, "..."]] | DeferredReduce:
  """Reduces cotangents of collect_along_axis back to its input shardings."""
  if defer and compute_type is not None:
    raise ValueError("defer runs the reduction in the caller's shard_map, which rules out" " compute_type.")
  if axes is not None:
    axes = frozenset(axes)

  def _to_tuple(m: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(m, str):
      return (m,)
    return tuple(m)

  def _is_axis_spec(x: Any) -> bool:
    return isinstance(x, str) or (
        isinstance(x, (tuple, list)) and not hasattr(x, "_fields") and all(isinstance(e, str) for e in x)
    )

  def _get_physical_axes(leaf_axis_name: str | Sequence[str]) -> tuple[tuple[str, ...], frozenset[str]]:
    names: list[str] = []
    for name in _to_tuple(leaf_axis_name):
      for ax in _to_tuple(axis_mapping.get(name, name)):
        if ax not in names:
          names.append(ax)
    return tuple(names), frozenset(names)

  def _nothing_to_reduce() -> Any:
    if not defer:
      return xs
    return DeferredReduce(operand=[], local_fn=lambda operand: operand, out_specs=[], finish=lambda reduced: xs)

  leaves = jax.tree.leaves(xs)
  if not leaves:
    return _nothing_to_reduce()
  physical_axes_tree = jax.tree.map(
      lambda spec, sub_xs: jax.tree.map(lambda _: _get_physical_axes(spec), sub_xs),
      axis_name, xs, is_leaf=_is_axis_spec,
  )
  flat_physical_axes = jax.tree.leaves(
      physical_axes_tree,
      is_leaf=lambda x: (isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], frozenset)),
  )
  if not any(names for names, _ in flat_physical_axes):
    return _nothing_to_reduce()
  mesh = jax.typeof(leaves[0]).sharding.mesh
  split_tree = jax.tree.map(lambda split, sub_xs: jax.tree.map(lambda _: split, sub_xs), split_scatters, xs)
  in_specs = jax.tree.map(lambda arr: jax.typeof(arr).sharding.spec, xs)
  like_specs = jax.tree.map(lambda arr: jax.typeof(arr).sharding.spec, like)

  def _leaf_collectives(
      in_spec: jax.sharding.PartitionSpec,
      like_spec: jax.sharding.PartitionSpec,
      physical_axes: tuple[tuple[str, ...], frozenset[str]],
  ) -> tuple[list[tuple[int, tuple[str, ...]]], list[str], frozenset[str]]:
    physical_axis_names, physical_axis_set = physical_axes
    pending = physical_axis_set & in_spec.unreduced
    selected = pending if axes is None else pending & axes
    sharded_axes = set()
    scatters = []
    for dim_idx, part in reversed(list(enumerate(like_spec.partitions))):
      if part is None:
        continue
      part_axes = (part,) if isinstance(part, str) else part
      sharded_axes.update(part_axes)
      dim_pending = [ax for ax in part_axes if ax in pending]
      dim_selected = [ax for ax in dim_pending if ax in selected]
      if dim_selected and len(dim_selected) < len(dim_pending):
        raise ValueError(
            f"Reduce-scattering along only {dim_selected} of {dim_pending},"
            f" which shard dimension {dim_idx} of {like_spec} together, would" " scramble the shards."
        )
      if dim_selected:
        scatters.append((dim_idx, tuple(dim_selected)))
    psums = [
        ax for ax in reversed(physical_axis_names)
        if ax in selected and ax not in sharded_axes and ax not in like_spec.reduced and ax not in like_spec.unreduced
    ]
    return scatters, psums, pending - selected

  def _out_spec(in_spec, like_spec, physical_axes):
    spec = like_spec.to_ct_spec()
    _, _, skipped = _leaf_collectives(in_spec, like_spec, physical_axes)
    if not skipped:
      return spec

    def _unscattered(part):
      if part is None or isinstance(part, str):
        return None if part in skipped else part
      return tuple(ax for ax in part if ax not in skipped) or None

    return jax.sharding.PartitionSpec(
        *map(_unscattered, spec.partitions), reduced=spec.reduced, unreduced=spec.unreduced | skipped,
    )

  out_specs = jax.tree.map(_out_spec, in_specs, like_specs, physical_axes_tree)

  def _reduce_leaf(arr, in_spec, like_spec, physical_axes, split):
    scatters, psums, _ = _leaf_collectives(in_spec, like_spec, physical_axes)

    def _part_axes(dim_idx):
      part = like_spec.partitions[dim_idx]
      return (part,) if isinstance(part, str) else tuple(part or ())

    def _psum_scatter(arr, dim_idx, scatter_axes):
      groups = [(ax,) for ax in scatter_axes] if split else [scatter_axes]
      for group in groups:
        arr = jax.lax.psum_scatter(arr, group, scatter_dimension=dim_idx, tiled=True)
      return arr

    i = 0
    while i < len(scatters):
      dim_idx, scatter_axes = scatters[i]
      if i + 1 < len(scatters):
        lead_dim, lead_axes = scatters[i + 1]
        lead_size = math.prod(jax.lax.axis_size(a) for a in lead_axes)
        local_shape = list(arr.shape)
        local_shape[lead_dim] //= lead_size
        if _merged_gather_dims(
            local_shape, (lead_dim, _part_axes(lead_dim), lead_axes), (dim_idx, _part_axes(dim_idx), scatter_axes),
        ):
          shape = arr.shape
          merged = arr.reshape(*shape[:lead_dim], shape[lead_dim] * shape[dim_idx], *shape[dim_idx + 1 :])
          merged = _psum_scatter(merged, lead_dim, lead_axes + scatter_axes)
          arr = merged.reshape(*shape[:lead_dim], 1, -1, *shape[dim_idx + 1 :])
          i += 2
          continue
      arr = _psum_scatter(arr, dim_idx, scatter_axes)
      i += 1
    for ax in psums:
      arr = jax.lax.psum(arr, ax)
    return arr

  if defer:
    xs_leaves, xs_treedef = jax.tree.flatten(xs)
    leaf_specs = list(zip(
        xs_treedef.flatten_up_to(in_specs), xs_treedef.flatten_up_to(like_specs),
        xs_treedef.flatten_up_to(physical_axes_tree),
    ))
    leaf_splits = xs_treedef.flatten_up_to(split_tree)
    leaf_out_specs = xs_treedef.flatten_up_to(out_specs)
    reduced_idxs = []
    for i, specs in enumerate(leaf_specs):
      scatters, psums, _ = _leaf_collectives(*specs)
      if scatters or psums:
        reduced_idxs.append(i)

    def _local_fn(operand):
      return [_reduce_leaf(arr, *leaf_specs[i], split=leaf_splits[i]) for arr, i in zip(operand, reduced_idxs)]

    def _finish(reduced):
      merged = list(xs_leaves)
      for i, arr in zip(reduced_idxs, reduced):
        merged[i] = arr
      return xs_treedef.unflatten(merged)

    return DeferredReduce(
        operand=[xs_leaves[i] for i in reduced_idxs], local_fn=_local_fn,
        out_specs=[leaf_out_specs[i] for i in reduced_idxs], finish=_finish,
    )

  def _reduce_pytree(xs_local):
    return jax.tree.map(_reduce_leaf, xs_local, in_specs, like_specs, physical_axes_tree, split_tree)

  if compute_type is not None:
    _reduce_pytree = jax.experimental.compute_on.compute_on(
        compute_type=compute_type, out_memory_spaces=jax.memory.Space.Device, compiler_options=compiler_options,
    )(_reduce_pytree)
  return jax.shard_map(_reduce_pytree, mesh=mesh, in_specs=(in_specs,), out_specs=out_specs)(xs)
