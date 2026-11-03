"""Parameter fragment manipulation and scheduling for streaming DiLoCo."""

import re
from typing import Any, Iterator, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.utils import max_logging

BUCKET_KEY_SUFFIX = "#bucket"
REMAINDER_KEY_SUFFIX = "#remainder"


class BucketSpec(NamedTuple):
  """A non-scanned leaf split across layer fragments."""

  chunk_size: int
  remainder: int
  axis: int = 0


def _slice_axis(leaf: Any, axis: int, start: int, size: int) -> Any:
  if isinstance(leaf, jax.ShapeDtypeStruct):
    shape = list(leaf.shape)
    shape[axis] = size
    return jax.ShapeDtypeStruct(tuple(shape), leaf.dtype, sharding=getattr(leaf, "sharding", None))
  return jax.lax.slice_in_dim(leaf, start, start + size, axis=axis)


def _is_sharding_leaf(value: Any) -> bool:
  return value is None or isinstance(value, jax.sharding.Sharding)


def _sharding_leaves(kvs: list[tuple[Any, Any]], shardings: Any) -> list[Any]:
  if shardings is not None:
    sharding_kvs = jax.tree_util.tree_flatten_with_path(shardings, is_leaf=_is_sharding_leaf)[0]
    if len(sharding_kvs) != len(kvs):
      raise ValueError(f"shardings has {len(sharding_kvs)} leaves but the parameter tree has {len(kvs)}.")
    for (param_path, _), (sharding_path, _) in zip(kvs, sharding_kvs):
      param_key, sharding_key = jax.tree_util.keystr(param_path), jax.tree_util.keystr(sharding_path)
      if param_key != sharding_key:
        raise ValueError(
            f"shardings does not have the structure of the parameter tree: found {sharding_key} in shardings where"
            f" the parameter tree has {param_key}."
        )
    return [sharding for _, sharding in sharding_kvs]
  return [None if isinstance(leaf, jax.core.Tracer) else getattr(leaf, "sharding", None) for _, leaf in kvs]


def _axis_shard_counts(shape: tuple[int, ...], sharding: Any) -> list[int]:
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return [1] * len(shape)
  spec = tuple(sharding.spec) + (None,) * (len(shape) - len(sharding.spec))
  counts = []
  for entry in spec[: len(shape)]:
    if entry is None:
      counts.append(1)
    elif isinstance(entry, str):
      counts.append(int(sharding.mesh.shape[entry]))
    elif isinstance(entry, tuple):
      counts.append(int(np.prod([sharding.mesh.shape[axis] for axis in entry])))
    else:
      counts.append(0)
  return counts


class FragmentedTreeManipulator:
  """Partition and manipulate fragments of a JAX PyTree."""

  def __init__(
      self, keypath_to_is_scanned: dict[str, bool], fragment_to_layer_indices: dict[int, tuple[int, ...]],
      num_fragments: int, param_scan_axis: int = 0, leaf_keystrs: list[str] | None = None,
      bucketized_leaves: dict[str, BucketSpec] | None = None,
  ):
    self.keypath_to_is_scanned = keypath_to_is_scanned
    self.fragment_to_layer_indices = fragment_to_layer_indices
    self.num_fragments = num_fragments
    self.param_scan_axis = param_scan_axis
    self.leaf_keystrs = leaf_keystrs or []
    self.bucketized_leaves = bucketized_leaves or {}

  @property
  def num_layer_fragments(self) -> int:
    return self.num_fragments - 1

  @classmethod
  def create(cls, params_tree: Any, config: Any, shardings: Any = None) -> "FragmentedTreeManipulator":
    """Create a fragment manipulator from model parameters and configuration."""
    kvs, _ = jax.tree_util.tree_flatten_with_path(params_tree)
    num_layers = config.num_decoder_layers
    num_fragments = config.num_diloco_fragments
    num_transformer_fragments = num_fragments - 1
    if num_transformer_fragments <= 0:
      raise ValueError(
          f"num_diloco_fragments ({num_fragments}) must be at least 2 (1 for non-scanned parameters, at least 1 for"
          " scanned layers)."
      )
    if num_layers % num_transformer_fragments != 0:
      raise ValueError(
          f"num_decoder_layers ({num_layers}) must be divisible by "
          f"num_diloco_fragments - 1 ({num_transformer_fragments}) for now."
      )
    num_synced = num_layers // num_transformer_fragments
    use_sequential = config.use_sequential_layers
    param_scan_axis = getattr(config, "param_scan_axis", 0)
    fragment_to_layer_indices = {}
    for index in range(1, num_fragments):
      sync_id = index - 1
      if use_sequential:
        indices = list(range(sync_id * num_synced, (sync_id + 1) * num_synced))
      else:
        indices = list(range(sync_id, num_layers, num_transformer_fragments))
      fragment_to_layer_indices[index] = tuple(indices)
    scanned_regex = re.compile(r"/(?:layers|blocks|moe_layers|dense_layers|layers_outside_pipeline)(?:/|$)")
    keypath_to_is_scanned = {}
    leaf_keystrs = []
    for keypath, value in kvs:
      parts = []
      for key in keypath:
        parts.append(str(key.key) if hasattr(key, "key") else (str(key.idx) if hasattr(key, "idx") else str(key)))
      serialized_path = "/" + "/".join(parts)
      keystr = jax.tree_util.keystr(keypath)
      leaf_keystrs.append(keystr)
      is_scanned = (
          bool(scanned_regex.search(serialized_path)) and hasattr(value, "shape") and len(value.shape) > 0
          and value.shape[param_scan_axis] == num_layers
      )
      keypath_to_is_scanned[keystr] = is_scanned
    bucketized_leaves = {}
    if getattr(config, "diloco_bucketize_non_scanned", False):
      sharding_leaves = _sharding_leaves(kvs, shardings)
      if not any(isinstance(sharding, jax.sharding.NamedSharding) for sharding in sharding_leaves):
        max_logging.warning(
            "DiLoCo bucketization found no NamedSharding for any parameter (the given `shardings` are all None, or"
            " none were given and the leaves are tracers, host arrays or not placed with a NamedSharding); every axis"
            " is treated as unsharded, so a sharded axis may be split."
        )
      for (keypath, value), sharding in zip(kvs, sharding_leaves):
        keystr = jax.tree_util.keystr(keypath)
        shape = tuple(getattr(value, "shape", ()))
        if keypath_to_is_scanned[keystr] or len(shape) < 2:
          continue
        long_axes = [axis for axis in range(len(shape)) if shape[axis] >= num_transformer_fragments]
        shard_counts = _axis_shard_counts(shape, sharding)
        candidates = [axis for axis in long_axes if shard_counts[axis] == 1]
        if candidates:
          axis = max(candidates, key=shape.__getitem__)
          bucketized_leaves[keystr] = BucketSpec(*divmod(shape[axis], num_transformer_fragments), axis=axis)
          max_logging.log(f"DiLoCo: bucketizing {keystr} {shape} -> {bucketized_leaves[keystr]}")
        elif long_axes:
          max_logging.log(
              f"DiLoCo: not bucketizing {keystr} {shape}: every axis with >= {num_transformer_fragments} indices is"
              f" sharded (shards per axis {shard_counts}); it stays whole in fragment 0."
          )
    return cls(
        keypath_to_is_scanned=keypath_to_is_scanned, fragment_to_layer_indices=fragment_to_layer_indices,
        num_fragments=num_fragments, param_scan_axis=param_scan_axis, leaf_keystrs=leaf_keystrs,
        bucketized_leaves=bucketized_leaves,
    )

  def _iter_keyed_leaves(self, tree: Any) -> Iterator[tuple[str, Any]]:
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == len(self.leaf_keystrs):
      yield from zip(self.leaf_keystrs, leaves)
    else:
      for keypath, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        yield jax.tree_util.keystr(keypath), leaf

  def _scan_axis(self, leaf: Any, has_replica_dim: bool) -> int:
    return self.param_scan_axis + 1 if has_replica_dim and leaf.ndim > self.param_scan_axis + 1 else self.param_scan_axis

  def _bucket_rows(self, keystr: str, fragment_idx: int) -> tuple[int, int] | None:
    spec = self.bucketized_leaves[keystr]
    if fragment_idx > 0:
      return (fragment_idx - 1) * spec.chunk_size, spec.chunk_size
    if spec.remainder:
      return self.num_layer_fragments * spec.chunk_size, spec.remainder
    return None

  def _bucket_axis(self, keystr: str, has_replica_dim: bool) -> int:
    return self.bucketized_leaves[keystr].axis + (1 if has_replica_dim else 0)

  def _layer_indices(self, fragment_idx: int) -> tuple[tuple[int, ...], bool]:
    layer_indices = tuple(int(index) for index in self.fragment_to_layer_indices.get(fragment_idx, (fragment_idx - 1,)))
    is_contiguous = len(layer_indices) > 0 and list(layer_indices) == list(range(layer_indices[0], layer_indices[-1] + 1))
    return layer_indices, is_contiguous

  def _check_fragment_idx(self, fragment_idx: int) -> None:
    if not 0 <= fragment_idx < self.num_fragments:
      raise ValueError(f"fragment_idx ({fragment_idx}) must be in [0, {self.num_fragments}).")

  def get_flat_fragment(self, tree: Any, fragment_idx: int, has_replica_dim: bool = False) -> dict[str, Any]:
    """Extract a flat dictionary of parameters belonging to one fragment."""
    self._check_fragment_idx(fragment_idx)
    flat_frag = {}
    layer_indices, is_contiguous = self._layer_indices(fragment_idx) if fragment_idx > 0 else ((), False)
    for keystr, value in self._iter_keyed_leaves(tree):
      if self.keypath_to_is_scanned.get(keystr, False):
        if fragment_idx == 0:
          continue
        axis = self._scan_axis(value, has_replica_dim)
        if isinstance(value, jax.ShapeDtypeStruct) or is_contiguous:
          flat_frag[keystr] = _slice_axis(value, axis, layer_indices[0], len(layer_indices))
        else:
          flat_frag[keystr] = jnp.take(value, np.array(layer_indices, dtype=np.int32), axis=axis)
      elif keystr in self.bucketized_leaves:
        rows = self._bucket_rows(keystr, fragment_idx)
        if rows is not None:
          suffix = BUCKET_KEY_SUFFIX if fragment_idx > 0 else REMAINDER_KEY_SUFFIX
          flat_frag[keystr + suffix] = _slice_axis(value, self._bucket_axis(keystr, has_replica_dim), *rows)
      elif fragment_idx == 0:
        flat_frag[keystr] = value
    return flat_frag

  def apply_flat_fragment(
      self, tree: Any, fragment_idx: int, flat_fragment: dict[str, Any], has_replica_dim: bool = False,
  ) -> Any:
    """Merge a flat fragment into the full parameter PyTree."""
    self._check_fragment_idx(fragment_idx)
    _, treedef = jax.tree_util.tree_flatten(tree)
    layer_indices, is_contiguous = self._layer_indices(fragment_idx) if fragment_idx > 0 else ((), False)
    new_leaves = []
    for keystr, value in self._iter_keyed_leaves(tree):
      if self.keypath_to_is_scanned.get(keystr, False):
        if fragment_idx > 0 and keystr in flat_fragment and not isinstance(value, jax.ShapeDtypeStruct):
          axis = self._scan_axis(value, has_replica_dim)
          update = jnp.asarray(flat_fragment[keystr], dtype=value.dtype)
          if is_contiguous:
            value = jax.lax.dynamic_update_slice_in_dim(value, update, layer_indices[0], axis=axis)
          else:
            index = tuple(slice(None) if dim != axis else np.array(layer_indices, dtype=np.int32)
                          for dim in range(value.ndim))
            value = value.at[index].set(update)
      elif keystr in self.bucketized_leaves:
        suffix = BUCKET_KEY_SUFFIX if fragment_idx > 0 else REMAINDER_KEY_SUFFIX
        rows = self._bucket_rows(keystr, fragment_idx)
        if rows is not None and keystr + suffix in flat_fragment and not isinstance(value, jax.ShapeDtypeStruct):
          axis = self._bucket_axis(keystr, has_replica_dim)
          update = jnp.asarray(flat_fragment[keystr + suffix], dtype=value.dtype)
          value = jax.lax.dynamic_update_slice_in_dim(value, update, rows[0], axis=axis)
      elif fragment_idx == 0 and keystr in flat_fragment:
        value = flat_fragment[keystr]
      new_leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def get_streaming_schedule(config: Any) -> tuple[int, int]:
  """Compute the steps between synchronizations and the synchronization period."""
  num_fragments = config.num_diloco_fragments
  steps_between_syncs = int(round(config.diloco_sync_period / num_fragments))
  steps_between_syncs = max(1, steps_between_syncs)
  period = num_fragments * steps_between_syncs
  return steps_between_syncs, period
