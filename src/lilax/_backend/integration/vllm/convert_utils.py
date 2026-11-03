"""Weight-conversion primitives for MaxText to vLLM weight synchronization."""

import functools
import os
from typing import Mapping, Any, Callable, Dict, Tuple, Optional

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_TPU_NUM_LANES = 128
MOE_MLP_WEIGHTS = frozenset({"wi", "wi_0", "wi_1", "wo"})


def pad_to_tpu_lanes(dim: int, lane_size: int = DEFAULT_TPU_NUM_LANES) -> int:
  """Rounds dim up to the nearest multiple of lane_size."""
  if dim <= 0:
    return 0
  return ((dim + lane_size - 1) // lane_size) * lane_size


def compute_padded_moe_mlp_dim(
    hidden_size: Optional[int],
    moe_mlp_tp_size: int,
    num_lanes: int = DEFAULT_TPU_NUM_LANES,
) -> Optional[int]:
  """Computes the padded MoE intermediate size GMM_v2 requires."""
  if hidden_size is None or moe_mlp_tp_size <= 0 or num_lanes <= 0:
    return hidden_size
  min_required = 2 * num_lanes * moe_mlp_tp_size
  if (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
    return ((max(hidden_size, min_required) + min_required - 1) // min_required) * min_required
  return hidden_size


def _delete_target_buffers(tgt_flat: Mapping[Any, Any], src_flat: Mapping[Any, Any]):
  """Leave buffer lifetime to JAX garbage collection and donation."""
  del tgt_flat, src_flat


def _reshard_in_chunks(
    src_flat: Dict[tuple, Any],
    spec_flat: Dict[tuple, Any],
    reshard_fn: Callable,
    chunk_size: int,
    delete_dst_buffers: bool,
) -> Dict[tuple, Any]:
  """Batch resharding to limit contiguous memory fragmentation."""
  resharded_flat = {}
  keys = [key for key in src_flat.keys() if key in spec_flat]
  for index in range(0, len(keys), chunk_size):
    chunk_keys = keys[index : index + chunk_size]
    src_chunk = {key: src_flat[key] for key in chunk_keys}
    spec_chunk = {key: spec_flat[key] for key in chunk_keys}
    if delete_dst_buffers:
      _delete_target_buffers(spec_chunk, src_chunk)
    resharded_chunk = reshard_fn(source=src_chunk, target=spec_chunk)
    resharded_flat.update(resharded_chunk)
  return resharded_flat


def _fuse_moe_weights(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    tgt_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Fuse MoE wi_0 and wi_1 into wi in the source tree."""
  new_src_flat = dict(src_flat)
  sample_tgt_wi = None
  for tgt_key, tgt_val in tgt_flat.items():
    if tgt_key and tgt_key[-1] == "wi":
      sample_tgt_wi = tgt_val
      break
  for src_key in list(new_src_flat.keys()):
    if not src_key or src_key[-1] != "wi_0":
      continue
    wi_1_key = src_key[:-1] + ("wi_1",)
    wi_target_key = src_key[:-1] + ("wi",)
    if wi_1_key not in new_src_flat:
      continue
    wi_0 = new_src_flat.pop(src_key)
    wi_1 = new_src_flat.pop(wi_1_key)
    axis = len(wi_0.shape) - 1
    matching_tgt = tgt_flat.get(wi_target_key, sample_tgt_wi)
    if matching_tgt is not None:
      tgt_axis = len(matching_tgt.shape) - 1
      n_shards = _get_n_shards(matching_tgt, tgt_axis)
      target_dim = matching_tgt.shape[tgt_axis]
    else:
      n_shards = 1
      target_dim = wi_0.shape[axis] + wi_1.shape[axis]
    fused_shape = list(wi_0.shape)
    fused_shape[axis] = target_dim
    fused_shape_tuple = tuple(fused_shape)
    logging.info(
        "Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d",
        ".".join(str(key) for key in wi_target_key), wi_0.shape, wi_1.shape, fused_shape_tuple, axis,
    )
    new_src_flat[wi_target_key] = _interleave_moe_weights(wi_0, wi_1, fused_shape_tuple, n_shards, axis=axis)
    del wi_0, wi_1
  return new_src_flat


class ShapeMismatchError(ValueError):
  """Raised when source and target shapes are incompatible."""


def normalize_dtype(tgt_dtype: Any) -> Any:
  """Normalize string or dtype representations into a JAX dtype."""
  if tgt_dtype is None:
    return None
  if isinstance(tgt_dtype, str):
    if tgt_dtype in ("bfloat16", "bf16"):
      return jnp.bfloat16
    if tgt_dtype in ("float32", "fp32"):
      return jnp.float32
    return jnp.dtype(tgt_dtype)
  return tgt_dtype


def _apply_dtype_cast(val: Any, tgt_dtype: Any, src_key: str) -> Any:
  """Cast a value to the target dtype when necessary."""
  tgt_dtype = normalize_dtype(tgt_dtype)
  if isinstance(val, jax.ShapeDtypeStruct):
    if tgt_dtype is not None and val.dtype != tgt_dtype:
      return jax.ShapeDtypeStruct(val.shape, tgt_dtype)
    return val
  if not hasattr(val, "dtype"):
    return val
  if tgt_dtype is not None and val.dtype != tgt_dtype:
    logging.log_first_n(logging.WARNING, "Type mismatch on %s: %s -> %s", 1, src_key, val.dtype, tgt_dtype)
    return val.astype(tgt_dtype)
  return val


def _shapes_are_repeatable(candidate_shape: tuple[int, ...], tgt_shape: tuple[int, ...]) -> bool:
  """Return whether candidate_shape can be repeated to match tgt_shape."""
  if len(candidate_shape) != len(tgt_shape):
    return False
  for src_dim, tgt_dim in zip(candidate_shape, tgt_shape):
    if src_dim > tgt_dim or tgt_dim % src_dim != 0:
      return False
  return True


def _unstack_scanned_param(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
    scan_axis: Optional[int] = None,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Unstack a scanned parameter, detecting the scan axis when unspecified."""
  if not (hasattr(src_val, "shape") and hasattr(tgt_val, "shape")):
    return (src_val,)
  src_shape = src_val.shape
  tgt_shape = tgt_val.shape
  if src_shape == tgt_shape:
    return (src_val,)
  if len(src_shape) == len(tgt_shape) + 1:
    if scan_axis is None:
      for axis in range(len(src_shape)):
        candidate = src_shape[:axis] + src_shape[axis + 1 :]
        if _shapes_are_repeatable(candidate, tgt_shape):
          scan_axis = axis
          break
    if scan_axis is not None:
      if scan_axis != 0:
        perm = (scan_axis,) + tuple(axis for axis in range(len(src_shape)) if axis != scan_axis)
        if hasattr(src_val, "transpose"):
          src_val = src_val.transpose(perm)
        elif isinstance(src_val, np.ndarray):
          src_val = np.transpose(src_val, perm)
      return tuple(jnp.unstack(src_val))
    logging.warning(
        "Shape mismatch in scanned param '%s'. Src: %s, Tgt: %s. Cannot determine scan axis.",
        key_path, src_shape, tgt_shape,
    )
  return (src_val,)


def _partition_size(partition: Optional[str | Tuple[str, ...]], mesh: jax.sharding.Mesh) -> int:
  """Return the total mesh-axis size sharding one tensor axis."""
  if partition is None:
    return 1
  names = (partition,) if isinstance(partition, str) else tuple(partition)
  size = 1
  for name in names:
    size *= mesh.shape[name]
  return size


def _spec_at_axis(sharding: Optional[jax.sharding.Sharding], axis: int) -> Optional[str | Tuple[str, ...]]:
  """Return the PartitionSpec entry at axis, or None if absent."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return None
  spec = sharding.spec
  return spec[axis] if axis < len(spec) else None


def _get_n_shards(arr: jax.Array | np.ndarray, axis: int) -> int:
  """Return the number of shards for an array axis."""
  sharding = getattr(arr, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    return _partition_size(_spec_at_axis(sharding, axis), sharding.mesh)
  return 1


def _device_ids(x: Any) -> Tuple[int, ...]:
  """Return sorted device IDs backing an array's sharding."""
  arr = getattr(x, "value", x)
  sharding = getattr(arr, "sharding", None)
  if sharding is None:
    return ()
  mesh = getattr(sharding, "mesh", None)
  if mesh is not None and hasattr(mesh, "devices"):
    devices = np.asarray(mesh.devices).flatten().tolist()
  elif hasattr(sharding, "device_set"):
    devices = list(sharding.device_set)
  else:
    return ()
  return tuple(sorted(int(device.id) for device in devices if hasattr(device, "id")))


def _sharding_summary(x: Any) -> str:
  """Describe an array's placement for weight-sync debugging."""
  arr = getattr(x, "value", x)
  shape = getattr(arr, "shape", None)
  sharding = getattr(arr, "sharding", None)
  if sharding is None:
    return f"<unsharded> shape={shape}"
  ids = _device_ids(arr)
  span = f"[{ids[0]}..{ids[-1]}]({len(ids)})" if ids else "[unknown]"
  spec = getattr(sharding, "spec", None)
  return f"{type(sharding).__name__} shape={shape} devices={span} spec={spec}"


@functools.partial(jax.jit, static_argnames=("pad_specs",))
def _jit_zero_pad_axes(arr, pad_specs):
  """Tail-pad multiple axes separately within each shard."""
  out = arr
  for axis, n_shards, per_shard_extra in pad_specs:
    if per_shard_extra <= 0:
      continue
    src_dim = out.shape[axis]
    src_chunk_size = src_dim // n_shards
    split_shape = list(out.shape)
    split_shape.insert(axis + 1, src_chunk_size)
    split_shape[axis] = n_shards
    arr_split = out.reshape(split_shape)
    pad_width = [(0, 0)] * arr_split.ndim
    pad_width[axis + 1] = (0, per_shard_extra)
    arr_padded = jnp.pad(arr_split, pad_width)
    final_shape = list(out.shape)
    final_shape[axis] = src_dim + per_shard_extra * n_shards
    out = arr_padded.reshape(final_shape)
  return out


@functools.partial(jax.jit, static_argnames=("axis",))
def _jit_unstack(arr, axis: int):
  """Split an array along an axis inside one XLA program."""
  return tuple(jnp.unstack(arr, axis=axis))


@functools.partial(jax.jit, static_argnames=("repeats",))
def _jit_repeat_axes(arr, repeats):
  """Repeat multiple axes inside one XLA program."""
  out = arr
  for axis, count in repeats:
    out = jnp.repeat(out, count, axis=axis)
  return out


def _align_per_axis(
    arr: jax.Array | np.ndarray | jax.ShapeDtypeStruct,
    tgt_shape: Tuple[int, ...],
    tgt_sharding: Optional[jax.sharding.Sharding],
    key_path: str,
) -> jax.Array | np.ndarray | jax.ShapeDtypeStruct:
  """Align axes using shard-aware MoE zero-padding or repetition."""
  if isinstance(arr, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(tgt_shape, arr.dtype)
  if not hasattr(arr, "shape"):
    return arr
  if arr.shape == tgt_shape:
    return arr
  if len(arr.shape) != len(tgt_shape):
    raise ShapeMismatchError(f"Rank mismatch for {key_path}: src={arr.shape} vs tgt={tgt_shape}")
  mismatches = []
  for axis, (src_dim, tgt_dim) in enumerate(zip(arr.shape, tgt_shape)):
    if src_dim == tgt_dim:
      continue
    if tgt_dim < src_dim:
      raise ShapeMismatchError(f"Cannot shrink axis {axis} for {key_path}: src={src_dim} -> tgt={tgt_dim}")
    mismatches.append((axis, src_dim, tgt_dim))
  if not mismatches:
    return arr
  if key_path.split(".")[-1] in MOE_MLP_WEIGHTS:
    if isinstance(tgt_sharding, jax.sharding.NamedSharding):
      mesh = tgt_sharding.mesh
      pad_specs = []
      for axis, src_dim, tgt_dim in mismatches:
        n_shards = _partition_size(_spec_at_axis(tgt_sharding, axis), mesh)
        if tgt_dim % n_shards != 0:
          raise ValueError(
              f"Target dimension {tgt_dim} on axis {axis} for {key_path} is not "
              f"divisible by n_shards={n_shards}; the target shape itself "
              f"is misconfigured for the requested sharding."
          )
        if (tgt_dim - src_dim) % n_shards != 0 or src_dim % n_shards != 0:
          raise ValueError(
              f"Cannot interleave pad axis {axis} for {key_path}: src_dim "
              f"({src_dim}) or extra ({tgt_dim - src_dim}) is not cleanly divisible by "
              f"n_shards ({n_shards}). Ensure the source tensor is evenly "
              f"partitionable."
          )
        pad_specs.append((axis, n_shards, (tgt_dim - src_dim) // n_shards))
    else:
      pad_specs = [(axis, 1, tgt_dim - src_dim) for axis, src_dim, tgt_dim in mismatches]
    return _jit_zero_pad_axes(arr, tuple(pad_specs))
  repeats = []
  for axis, src_dim, tgt_dim in mismatches:
    if tgt_dim % src_dim != 0:
      raise ShapeMismatchError(
          f"Cannot align axis {axis} for {key_path}: src={src_dim} -> tgt={tgt_dim} "
          f"is not an integer multiple and the key is not a recognized "
          f"MoE pattern."
      )
    repeats.append((axis, tgt_dim // src_dim))
  return _jit_repeat_axes(arr, tuple(repeats))


@functools.partial(jax.jit, static_argnames=("tgt_shape", "n_shards", "axis", "lane_size"))
def _interleave_moe_weights(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    n_shards: int,
    axis: Optional[int] = None,
    lane_size: Optional[int] = 0,
) -> jax.Array | np.ndarray:
  """Fuse gate and up weights per shard in the TPU GMM layout."""
  if lane_size is None:
    lane_size = 0
  if axis is None:
    axis = len(tgt_shape) - 1
  elif axis < 0:
    axis = len(tgt_shape) + axis
  target_half_dim = tgt_shape[axis] // 2
  target_chunk_size = target_half_dim // n_shards

  def _pad_and_chunk(arr):
    current_total_size = arr.shape[axis]
    chunk_size = current_total_size // n_shards
    new_shape = list(arr.shape)
    new_shape[axis] = n_shards
    new_shape.insert(axis + 1, chunk_size)
    arr_reshaped = arr.reshape(new_shape)
    pad_amount = target_chunk_size - chunk_size
    if pad_amount > 0:
      pad_widths = [(0, 0)] * arr_reshaped.ndim
      pad_widths[axis + 1] = (0, pad_amount)
      arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
    return arr_reshaped

  padded_wi_0 = _pad_and_chunk(wi_0)
  padded_wi_1 = _pad_and_chunk(wi_1)
  if lane_size > 0 and target_chunk_size % lane_size == 0:
    num_lanes = target_chunk_size // lane_size
    shape_lanes = list(padded_wi_0.shape)
    shape_lanes[axis + 1] = num_lanes
    shape_lanes.insert(axis + 2, lane_size)
    padded_wi_0 = padded_wi_0.reshape(shape_lanes)
    padded_wi_1 = padded_wi_1.reshape(shape_lanes)
    combined = jnp.stack([padded_wi_0, padded_wi_1], axis=axis + 2)
  else:
    combined = jnp.concatenate([padded_wi_0, padded_wi_1], axis=axis + 1)
  return combined.reshape(tgt_shape)


@functools.partial(
    jax.jit,
    static_argnames=("scan_axis", "n_shards", "tgt_shape", "scan_fused_axis", "tgt_fused_axis"),
)
def _fuse_and_unstack_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    scan_axis: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    scan_fused_axis: int,
    tgt_fused_axis: int,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Fuse scanned gate and up weights per unstacked layer."""
  unstacked_0 = jnp.unstack(wi_0, axis=scan_axis)
  unstacked_1 = jnp.unstack(wi_1, axis=scan_axis)
  return tuple(
      _interleave_moe_weights(gate, up, tgt_shape, n_shards, axis=tgt_fused_axis)
      for gate, up in zip(unstacked_0, unstacked_1)
  )


def _align_to_model_shape(
    src_val: jax.Array | np.ndarray | jax.ShapeDtypeStruct,
    tgt_val: jax.Array | np.ndarray | jax.ShapeDtypeStruct,
    key_path: str,
) -> jax.Array | np.ndarray | jax.ShapeDtypeStruct:
  """Align a source value to the target's shape and sharding."""
  if not (hasattr(src_val, "shape") and hasattr(tgt_val, "shape")):
    return src_val
  if src_val.shape == tgt_val.shape:
    return src_val
  tgt_sharding = getattr(tgt_val, "sharding", None)
  return _align_per_axis(src_val, tgt_val.shape, tgt_sharding, key_path)


def _bulk_align_and_unstack(
    arr: jax.Array | np.ndarray | jax.ShapeDtypeStruct,
    scan_axis: int,
    per_layer_tgt_val: jax.Array | np.ndarray | jax.ShapeDtypeStruct,
    key_path: str,
) -> Tuple[jax.Array | np.ndarray | jax.ShapeDtypeStruct, ...]:
  """Align the full scanned tensor before unstacking it by layer."""
  per_layer_shape = per_layer_tgt_val.shape
  if isinstance(arr, jax.ShapeDtypeStruct):
    num_layers = arr.shape[scan_axis]
    tgt_dtype = getattr(per_layer_tgt_val, "dtype", getattr(arr, "dtype", jnp.float32))
    return tuple(jax.ShapeDtypeStruct(per_layer_shape, tgt_dtype) for _ in range(num_layers))
  scanned_tgt_shape = per_layer_shape[:scan_axis] + (arr.shape[scan_axis],) + per_layer_shape[scan_axis:]
  scanned_tgt_sharding = _scanned_sharding_from_per_layer(getattr(per_layer_tgt_val, "sharding", None), scan_axis)
  if arr.shape == scanned_tgt_shape:
    return _jit_unstack(arr, scan_axis)
  aligned = _align_per_axis(arr, scanned_tgt_shape, scanned_tgt_sharding, key_path)
  return _jit_unstack(aligned, scan_axis)


def _scanned_sharding_from_per_layer(
    per_layer_sharding: Optional[jax.sharding.Sharding],
    scan_axis: int,
) -> Optional[jax.sharding.NamedSharding]:
  """Insert an unsharded scan axis into a per-layer NamedSharding."""
  if not isinstance(per_layer_sharding, jax.sharding.NamedSharding):
    return None
  spec = list(per_layer_sharding.spec)
  spec.insert(scan_axis, None)
  return jax.sharding.NamedSharding(
      per_layer_sharding.mesh,
      jax.sharding.PartitionSpec(*spec),
      memory_kind=per_layer_sharding.memory_kind,
  )


def resolve_rollout_tp(config: Any, tp: int = 1) -> int:
  """Resolve rollout TP from override, config, or environment."""
  if tp > 1:
    return int(tp)
  config_tp = 0
  if config is not None:
    raw = (
        getattr(config, "rollout_tensor_parallelism", 0)
        or getattr(getattr(config, "cluster", None), "rollout_tensor_parallelism", 0)
        or getattr(config, "rollout_mesh_tp", 0)
        or 0
    )
    if raw and int(raw) > 0:
      config_tp = int(raw)
  if not config_tp:
    env_tp = os.environ.get("ROLLOUT_TENSOR_PARALLEL_SIZE") or os.environ.get("ROLLOUT_MESH_TP")
    if env_tp and int(env_tp) > 0:
      config_tp = int(env_tp)
  return int(config_tp or 1)


def resolve_prefuse_moe_weights(config: Any, prefuse_moe_weights: Optional[bool] = None) -> bool:
  """Resolve the MoE prefuse flag from override, config, or environment."""
  if prefuse_moe_weights is not None:
    return bool(prefuse_moe_weights)
  if "ROLLOUT_PREFUSE_MOE_WEIGHTS" in os.environ:
    return os.environ["ROLLOUT_PREFUSE_MOE_WEIGHTS"].lower() in ("1", "true", "yes")
  if config is not None and getattr(config, "rollout_prefuse_moe_weights", None) is not None:
    return bool(config.rollout_prefuse_moe_weights)
  rollout_backend = getattr(config, "rollout_backend", None) or os.environ.get("ROLLOUT_BACKEND", "maxtext")
  if rollout_backend == "maxtext":
    return True
  if config is not None and getattr(config, "prefuse_moe_weights", None) is not None:
    return bool(config.prefuse_moe_weights)
  return os.environ.get("PREFUSE_MOE_WEIGHTS", "0").lower() in ("1", "true", "yes")


def is_verify_weights_enabled() -> bool:
  """Return whether weight verification is active."""
  return os.environ.get("VERIFY_WEIGHTS", "").lower() == "true"
