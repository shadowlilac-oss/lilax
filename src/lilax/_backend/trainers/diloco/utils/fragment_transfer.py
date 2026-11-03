"""Fragment extraction, application and outer optimization for streaming DiLoCo."""

import dataclasses
import functools
import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from lilax._backend.trainers.diloco.utils.fragmenter import (
    BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX, FragmentedTreeManipulator,
)

TransferFragment = dict[str, jax.Array]


@dataclasses.dataclass(frozen=True)
class LeafLayout:
  """Static layout of one transfer fragment entry."""

  key: str
  shape: tuple[int, ...]
  spec: PartitionSpec | None


def _spec_axes(spec: PartitionSpec) -> list[str]:
  axes = []
  for entry in spec:
    if entry is not None:
      axes.extend((entry,) if isinstance(entry, str) else entry)
  return axes


def _shard_local_spec(shape: tuple[int, ...], spec: PartitionSpec, mesh: Mesh) -> PartitionSpec | None:
  if not _spec_axes(spec):
    return None
  for dim, entry in zip(shape, tuple(spec) + (None,) * (len(shape) - len(spec))):
    axes = () if entry is None else ((entry,) if isinstance(entry, str) else entry)
    if dim % math.prod(mesh.shape[axis] for axis in axes) != 0:
      return None
  return spec


def _flat_spec(spec: PartitionSpec | None) -> PartitionSpec:
  return PartitionSpec() if spec is None else PartitionSpec(tuple(_spec_axes(spec)))


def _flatten(value: jax.Array, mesh: Mesh, spec: PartitionSpec | None) -> jax.Array:
  if spec is None:
    return jnp.reshape(value, (-1,))
  flat_spec = _flat_spec(spec)
  return jax.shard_map(
      lambda array: jnp.reshape(array, (-1,)), mesh=mesh, in_specs=spec, out_specs=flat_spec, check_vma=False
  )(value)


def _unflatten(value: jax.Array, mesh: Mesh, leaf: LeafLayout) -> jax.Array:
  if leaf.spec is None:
    return jnp.reshape(value, leaf.shape)
  local_shape = list(leaf.shape)
  for index, entry in enumerate(leaf.spec):
    for axis in () if entry is None else ((entry,) if isinstance(entry, str) else entry):
      local_shape[index] //= mesh.shape[axis]
  flat_spec = _flat_spec(leaf.spec)
  return jax.shard_map(
      lambda array: jnp.reshape(array, local_shape), mesh=mesh, in_specs=flat_spec, out_specs=leaf.spec, check_vma=False
  )(value)


def _base_keystr(key: str) -> str:
  for suffix in (BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX):
    if key.endswith(suffix):
      return key[: -len(suffix)]
  return key


class FragmentTransfer:
  """Extract and apply flattened parameter fragments on a learner's mesh."""

  def __init__(self, manipulator: FragmentedTreeManipulator, params: Any, alpha: float = 0.0):
    self.manipulator = manipulator
    keyed = jax.tree_util.tree_flatten_with_path(params)[0]
    self._leaf_keystrs = [jax.tree_util.keystr(path) for path, _ in keyed]
    if manipulator.leaf_keystrs and list(manipulator.leaf_keystrs) != self._leaf_keystrs:
      raise ValueError("FragmentTransfer needs `manipulator` built from a tree with the same structure as `params`.")
    shardings = {key: getattr(leaf, "sharding", None) for key, (_, leaf) in zip(self._leaf_keystrs, keyed, strict=True)}
    for key, sharding in shardings.items():
      if not isinstance(sharding, NamedSharding):
        raise ValueError(f"FragmentTransfer needs a NamedSharding on every parameter; {key} has {sharding!r}.")
    meshes = {sharding.mesh for sharding in shardings.values()}
    if len(meshes) != 1:
      raise ValueError(f"FragmentTransfer needs all parameters on one mesh; found {len(meshes)} meshes.")
    self.mesh = meshes.pop()
    self.num_fragments = manipulator.num_fragments
    abstract = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), params)
    self.layouts = {frag: self._build_layout(manipulator.get_flat_fragment(abstract, frag), shardings) for frag in (0, 1)}
    self._flat_shardings = {
        frag: {leaf.key: NamedSharding(self.mesh, _flat_spec(leaf.spec)) for leaf in layout}
        for frag, layout in self.layouts.items()
    }
    self._first_layer, self._layer_start_step, self._layer_stride, self._num_layers = self._layer_progression()
    num_layer_fragments = len(manipulator.fragment_to_layer_indices)
    self._layer_grid = (
        self._layer_stride > 1
        and self._first_layer + (num_layer_fragments - 1) * self._layer_start_step < self._layer_stride
    )
    params_shardings = jax.tree.map(lambda value: value.sharding, params)
    apply_kwargs = {"out_shardings": params_shardings, "donate_argnums": 0, "keep_unused": True}
    self._extract_non_scanned = self._jit_extract(layer_fragment=False)
    self._extract_layer = self._jit_extract(layer_fragment=True)
    self._apply_non_scanned = jax.jit(functools.partial(self._apply, layer_fragment=False, alpha=alpha), **apply_kwargs)
    self._apply_layer = jax.jit(functools.partial(self._apply, layer_fragment=True, alpha=alpha), **apply_kwargs)

  def _build_layout(self, abstract_fragment: dict[str, jax.ShapeDtypeStruct], shardings) -> tuple[LeafLayout, ...]:
    layout = []
    for key in sorted(abstract_fragment):
      shape = tuple(abstract_fragment[key].shape)
      spec = shardings[_base_keystr(key)].spec
      layout.append(LeafLayout(key, shape, _shard_local_spec(shape, spec, self.mesh)))
    return tuple(layout)

  def _transfer_shardings(self, layer_fragment: bool) -> dict[str, NamedSharding]:
    return self._flat_shardings[1 if layer_fragment else 0]

  def _jit_extract(self, *, layer_fragment: bool):
    out_shardings = self._transfer_shardings(layer_fragment)
    return jax.jit(functools.partial(self._extract, layer_fragment=layer_fragment), out_shardings=out_shardings)

  def _layer_progression(self) -> tuple[int, int, int, int]:
    indices = self.manipulator.fragment_to_layer_indices
    if not indices:
      return 0, 0, 1, 0
    first = indices[1]
    num_layers = len(first)
    stride = first[1] - first[0] if num_layers > 1 else 1
    start_step = indices[2][0] - first[0] if len(indices) > 1 else 0
    for frag, layers in indices.items():
      expected = tuple(first[0] + (frag - 1) * start_step + stride * index for index in range(num_layers))
      if tuple(layers) != expected:
        raise ValueError(f"Layer fragment {frag} indices {layers} are not an arithmetic progression {expected}.")
    return first[0], start_step, stride, num_layers

  def _as_layer_grid(self, leaf: jax.Array, axis: int) -> jax.Array | None:
    if not self._layer_grid or leaf.shape[axis] != self._num_layers * self._layer_stride:
      return None
    return jnp.reshape(leaf, leaf.shape[:axis] + (self._num_layers, self._layer_stride) + leaf.shape[axis + 1 :])

  def _layer_slice(self, leaf: jax.Array, axis: int, fragment_idx: jax.Array) -> jax.Array:
    start = self._first_layer + (fragment_idx - 1) * self._layer_start_step
    if self._layer_stride == 1:
      return jax.lax.dynamic_slice_in_dim(leaf, start, self._num_layers, axis=axis)
    grid = self._as_layer_grid(leaf, axis)
    if grid is not None:
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(grid, start, 1, axis=axis + 1), axis + 1)
    return jnp.take(leaf, start + self._layer_stride * jnp.arange(self._num_layers), axis=axis)

  def _set_layer_slice(self, leaf: jax.Array, axis: int, fragment_idx: jax.Array, value: jax.Array) -> jax.Array:
    start = self._first_layer + (fragment_idx - 1) * self._layer_start_step
    if self._layer_stride == 1:
      return jax.lax.dynamic_update_slice_in_dim(leaf, value, start, axis=axis)
    grid = self._as_layer_grid(leaf, axis)
    if grid is not None:
      grid = jax.lax.dynamic_update_slice_in_dim(grid, jnp.expand_dims(value, axis + 1), start, axis=axis + 1)
      return jnp.reshape(grid, leaf.shape)
    index = [slice(None)] * leaf.ndim
    index[axis] = start + self._layer_stride * jnp.arange(self._num_layers)
    return leaf.at[tuple(index)].set(value)

  def _bucket_range(self, key: str, fragment_idx: jax.Array) -> tuple[Any, int, int]:
    spec = self.manipulator.bucketized_leaves[_base_keystr(key)]
    if key.endswith(BUCKET_KEY_SUFFIX):
      return (fragment_idx - 1) * spec.chunk_size, spec.chunk_size, spec.axis
    return self.manipulator.num_layer_fragments * spec.chunk_size, spec.remainder, spec.axis

  def _get(self, leaf: jax.Array, key: str, fragment_idx: jax.Array) -> jax.Array:
    if key.endswith((BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX)):
      start, size, axis = self._bucket_range(key, fragment_idx)
      return jax.lax.dynamic_slice_in_dim(leaf, start, size, axis=axis)
    if self.manipulator.keypath_to_is_scanned[key]:
      return self._layer_slice(leaf, self.manipulator.param_scan_axis, fragment_idx)
    return leaf

  def _set(self, leaf: jax.Array, key: str, fragment_idx: jax.Array, value: jax.Array) -> jax.Array:
    if key.endswith((BUCKET_KEY_SUFFIX, REMAINDER_KEY_SUFFIX)):
      start, _, axis = self._bucket_range(key, fragment_idx)
      return jax.lax.dynamic_update_slice_in_dim(leaf, value, start, axis=axis)
    if self.manipulator.keypath_to_is_scanned[key]:
      return self._set_layer_slice(leaf, self.manipulator.param_scan_axis, fragment_idx, value)
    return value

  def _layout(self, layer_fragment: bool) -> tuple[LeafLayout, ...]:
    return self.layouts[1 if layer_fragment else 0]

  def _extract(self, params: Any, fragment_idx: jax.Array, *, layer_fragment: bool) -> TransferFragment:
    leaves = dict(zip(self._leaf_keystrs, jax.tree.leaves(params), strict=True))
    return {
        leaf.key: _flatten(self._get(leaves[_base_keystr(leaf.key)], leaf.key, fragment_idx), self.mesh, leaf.spec)
        for leaf in self._layout(layer_fragment)
    }

  def _apply(self, params: Any, fragment_idx: jax.Array, fragment: TransferFragment, *, layer_fragment: bool, alpha: float):
    leaves, treedef = jax.tree.flatten(params)
    if len(leaves) != len(self._leaf_keystrs):
      raise ValueError(f"`params` has {len(leaves)} leaves; this FragmentTransfer was built for {len(self._leaf_keystrs)}.")
    index = {key: position for position, key in enumerate(self._leaf_keystrs)}
    for leaf in self._layout(layer_fragment):
      position = index[_base_keystr(leaf.key)]
      value = _unflatten(fragment[leaf.key], self.mesh, leaf).astype(jnp.float32)
      if alpha:
        value = alpha * self._get(leaves[position], leaf.key, fragment_idx).astype(jnp.float32) + (1.0 - alpha) * value
      leaves[position] = self._set(leaves[position], leaf.key, fragment_idx, value.astype(leaves[position].dtype))
    return jax.tree.unflatten(treedef, leaves)

  def same_layout_as(self, other: "FragmentTransfer") -> bool:
    """Return whether fragments use the same flattening order."""
    return self.layouts == other.layouts and self.mesh.shape == other.mesh.shape

  def _check_fragment_idx(self, fragment_idx: int) -> None:
    if not 0 <= fragment_idx < self.num_fragments:
      raise ValueError(f"fragment_idx ({fragment_idx}) must be in [0, {self.num_fragments}).")

  def extract(self, params: Any, fragment_idx: int) -> TransferFragment:
    """Extract one fragment on the learner mesh."""
    self._check_fragment_idx(fragment_idx)
    fn = self._extract_layer if fragment_idx > 0 else self._extract_non_scanned
    return fn(params, np.int32(fragment_idx))

  def _check_fragment_layout(self, fragment_idx: int, fragment: TransferFragment) -> None:
    expected = self._transfer_shardings(fragment_idx > 0)
    unexpected = sorted(set(fragment) - set(expected))
    if unexpected:
      raise ValueError(f"Fragment {fragment_idx} has entries {unexpected} that are not in this learner's transfer layout.")
    for leaf in self._layout(fragment_idx > 0):
      value = fragment.get(leaf.key)
      size = math.prod(leaf.shape)
      sharding = getattr(value, "sharding", None)
      if value is None or value.shape != (size,) or sharding is None or not sharding.is_equivalent_to(expected[leaf.key], 1):
        got = "no entry" if value is None else f"shape {value.shape} with sharding {sharding}"
        raise ValueError(
            f"Fragment entry {leaf.key} does not match this learner's transfer layout: expected shape ({size},) with"
            f" sharding {expected[leaf.key]}, got {got}."
        )

  def apply(self, params: Any, fragment_idx: int, fragment: TransferFragment) -> Any:
    """Apply a synchronized fragment to the donated parameter tree."""
    self._check_fragment_idx(fragment_idx)
    self._check_fragment_layout(fragment_idx, fragment)
    fn = self._apply_layer if fragment_idx > 0 else self._apply_non_scanned
    return fn(params, np.int32(fragment_idx), fragment)


def move_fragment(fragment: TransferFragment, mesh: Mesh) -> TransferFragment:
  """Move a transfer fragment onto a mesh without changing its PartitionSpecs."""
  return jax.device_put(fragment, {key: NamedSharding(mesh, value.sharding.spec) for key, value in fragment.items()})


def nesterov_outer_step(
    outer: TransferFragment, trace: TransferFragment, learner_fragments: Sequence[TransferFragment], *,
    learning_rate: float, momentum: float,
) -> tuple[TransferFragment, TransferFragment]:
  """Apply a Nesterov SGD step to the mean DiLoCo pseudo-gradient."""
  if not learner_fragments:
    raise ValueError("nesterov_outer_step needs at least one learner fragment.")
  shardings = tuple((key, outer[key].sharding, trace[key].sharding) for key in sorted(outer))
  return _jit_outer_step(shardings, learning_rate, momentum)(outer, trace, learner_fragments)


@functools.lru_cache(maxsize=None)
def _jit_outer_step(shardings: tuple[tuple[str, Any, Any], ...], learning_rate: float, momentum: float):
  out_shardings = ({key: outer for key, outer, _ in shardings}, {key: trace for key, _, trace in shardings})
  return jax.jit(
      functools.partial(_outer_step, learning_rate=learning_rate, momentum=momentum),
      out_shardings=out_shardings, donate_argnums=1,
  )


def _outer_step(
    outer: TransferFragment, trace: TransferFragment, learner_fragments: Sequence[TransferFragment], *,
    learning_rate: float, momentum: float,
) -> tuple[TransferFragment, TransferFragment]:
  new_outer, new_trace = {}, {}
  for key, outer_value in outer.items():
    outer_f32 = outer_value.astype(jnp.float32)
    pseudo_grad = sum(outer_f32 - frag[key].astype(jnp.float32) for frag in learner_fragments) / len(learner_fragments)
    trace_value = momentum * trace[key].astype(jnp.float32) + pseudo_grad
    update = learning_rate * (pseudo_grad + momentum * trace_value)
    new_outer[key] = (outer_f32 - update).astype(outer_value.dtype)
    new_trace[key] = trace_value.astype(trace[key].dtype)
  return new_outer, new_trace
