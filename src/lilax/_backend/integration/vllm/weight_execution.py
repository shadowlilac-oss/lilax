"""Execute direct rollout weight conversion groups."""

import jax
import jax.numpy as jnp

from lilax._backend.integration.vllm.convert_utils import (
    MOE_MLP_WEIGHTS, _align_per_axis, _apply_dtype_cast, _bulk_align_and_unstack,
    _fuse_and_unstack_moe, _get_n_shards, _jit_repeat_axes, _jit_unstack,
    _scanned_sharding_from_per_layer,
)


def _take_cycle_slot(val, group, scan_axis, cycle, error_type):
  if group.local_index is None:
    return val
  slot_axis = scan_axis + 1
  if val.ndim <= slot_axis or val.shape[slot_axis] != cycle - 1:
    raise error_type(
        f"Expected a nested cycle-slot axis of {cycle - 1} slots "
        f"on axis {slot_axis} of {group.source_path}, found shape "
        f"{val.shape}. Check param_scan_axis and "
        "inhomogeneous_layer_cycle_interval."
    )
  if isinstance(val, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(val.shape[:slot_axis] + val.shape[slot_axis + 1 :], val.dtype)
  return jax.lax.index_in_dim(val, group.local_index, axis=slot_axis, keepdims=False)


def _check_scan_axis(val, path, scan_axis, num_blocks, error_type):
  if val.shape[scan_axis] != num_blocks:
    raise error_type(
        f"Expected {num_blocks} scanned blocks on axis "
        f"{scan_axis} of {path}, found shape {val.shape}. Check "
        "param_scan_axis and inhomogeneous_layer_cycle_interval."
    )


def _fuse_moe_bulk(wi_0, wi_1, tgt_val, path, scan_axis, layout, tp, kv_tp_size, moe_mlp_tp_size, error_type):
  tgt_shape = tgt_val.shape
  tgt_fused_axis = len(tgt_shape) - 1
  scan_fused_axis = tgt_fused_axis if tgt_fused_axis < scan_axis else tgt_fused_axis + 1
  if layout == "per_shard_interleave":
    if isinstance(getattr(tgt_val, "sharding", None), jax.sharding.NamedSharding):
      n_shards = _get_n_shards(tgt_val, tgt_fused_axis)
    elif moe_mlp_tp_size > 1:
      n_shards = moe_mlp_tp_size
    elif kv_tp_size > 1 and moe_mlp_tp_size == 1:
      n_shards = 1
    else:
      n_shards = tp if tp > 1 else _get_n_shards(tgt_val, tgt_fused_axis)
    return _fuse_and_unstack_moe(wi_0, wi_1, scan_axis, n_shards, tgt_shape, scan_fused_axis, tgt_fused_axis)
  if layout == "concat":
    half = tgt_shape[tgt_fused_axis] // 2
    half_shape = tgt_shape[:tgt_fused_axis] + (half,) + tgt_shape[tgt_fused_axis + 1 :]
    scanned_half_shape = half_shape[:scan_axis] + (wi_0.shape[scan_axis],) + half_shape[scan_axis:]
    scanned_sharding = _scanned_sharding_from_per_layer(getattr(tgt_val, "sharding", None), scan_axis)
    padded = [_align_per_axis(weight, scanned_half_shape, scanned_sharding, path) for weight in (wi_0, wi_1)]
    return _jit_unstack(jnp.concatenate(padded, axis=scan_fused_axis), scan_axis)
  raise error_type(f"Unknown moe_fused_layout: {layout!r}")


def _slice_bulk_target_free(val, path, scan_axis, padded_dim, kv_replication, base_num_kv_heads):
  last_key = path.split(".")[-1]
  is_kv = "key.kernel" in path or "value.kernel" in path
  if isinstance(val, jax.ShapeDtypeStruct):
    unrolled_shape = list(val.shape[:scan_axis] + val.shape[scan_axis + 1 :])
    if last_key in MOE_MLP_WEIGHTS and padded_dim is not None:
      if last_key == "wo":
        if padded_dim > unrolled_shape[1]:
          unrolled_shape[1] = padded_dim
      elif last_key in ("wi_0", "wi_1", "wi"):
        if padded_dim > unrolled_shape[-1]:
          unrolled_shape[-1] = padded_dim
    if is_kv and kv_replication > 1 and (base_num_kv_heads is None or unrolled_shape[-2] == base_num_kv_heads):
      unrolled_shape[-2] = unrolled_shape[-2] * kv_replication
    return tuple(jax.ShapeDtypeStruct(tuple(unrolled_shape), val.dtype) for _ in range(val.shape[scan_axis]))
  if last_key in MOE_MLP_WEIGHTS and padded_dim is not None:
    if last_key == "wo":
      intermediate_axis = 2 if scan_axis <= 1 else 1
      if padded_dim > val.shape[intermediate_axis]:
        pad_amount = padded_dim - val.shape[intermediate_axis]
        pad_spec = [(0, 0)] * val.ndim
        pad_spec[intermediate_axis] = (0, pad_amount)
        val = jnp.pad(val, pad_spec)
    elif last_key in ("wi_0", "wi_1"):
      intermediate_axis = len(val.shape) - 1
      if padded_dim > val.shape[intermediate_axis]:
        pad_amount = padded_dim - val.shape[intermediate_axis]
        pad_spec = [(0, 0)] * val.ndim
        pad_spec[intermediate_axis] = (0, pad_amount)
        val = jnp.pad(val, pad_spec)
  if is_kv and kv_replication > 1 and (base_num_kv_heads is None or val.shape[-2] == base_num_kv_heads):
    val = _jit_repeat_axes(val, ((-2, kv_replication),))
  return _jit_unstack(val, scan_axis)


def _fuse_moe_bulk_target_free(wi_0, wi_1, scan_axis, padded_dim, layout, tp, kv_tp_size, moe_mlp_tp_size, error_type):
  unpadded_dim = wi_0.shape[-1]
  target_intermediate = padded_dim if padded_dim is not None and padded_dim > unpadded_dim else unpadded_dim
  unrolled_shape = list(wi_0.shape[:scan_axis] + wi_0.shape[scan_axis + 1 :])
  tgt_shape = tuple(unrolled_shape[:-1] + [2 * target_intermediate])
  if isinstance(wi_0, jax.ShapeDtypeStruct):
    return tuple(jax.ShapeDtypeStruct(tgt_shape, wi_0.dtype) for _ in range(wi_0.shape[scan_axis]))
  tgt_fused_axis = len(tgt_shape) - 1
  scan_fused_axis = tgt_fused_axis if tgt_fused_axis < scan_axis else tgt_fused_axis + 1
  if layout == "per_shard_interleave":
    if moe_mlp_tp_size > 1:
      n_shards = moe_mlp_tp_size
    elif kv_tp_size > 1 and moe_mlp_tp_size == 1:
      n_shards = 1
    else:
      n_shards = tp if tp > 1 else _get_n_shards(wi_0, scan_fused_axis)
    return _fuse_and_unstack_moe(wi_0, wi_1, scan_axis, n_shards, tgt_shape, scan_fused_axis, tgt_fused_axis)
  if layout == "concat":
    if target_intermediate > unpadded_dim:
      pad_spec = [(0, 0)] * wi_0.ndim
      pad_spec[-1] = (0, target_intermediate - unpadded_dim)
      wi_0 = jnp.pad(wi_0, pad_spec)
      wi_1 = jnp.pad(wi_1, pad_spec)
    return _jit_unstack(jnp.concatenate([wi_0, wi_1], axis=scan_fused_axis), scan_axis)
  raise error_type(f"Unknown moe_fused_layout: {layout!r}")


def _cast_target_free(val, target_dtype, path):
  if target_dtype is None or not hasattr(val, "dtype") or not jnp.issubdtype(val.dtype, jnp.floating):
    return val
  return _apply_dtype_cast(val, target_dtype, path)


def execute_group(group, src_flat, tgt_flat=None, *, scan_axis, cycle, num_blocks, moe_fused_layout, tp,
                  kv_tp_size, moe_mlp_tp_size, padded_base_moe_mlp_dim, kv_replication, base_num_kv_heads,
                  target_dtype, error_type):
  path = group.source_path
  if tgt_flat is None:
    if group.op == "identity":
      val = _cast_target_free(src_flat[group.source_keys[0]], target_dtype, path)
      is_kv = "key.kernel" in path or "value.kernel" in path
      if is_kv and kv_replication > 1 and (base_num_kv_heads is None or val.shape[-2] == base_num_kv_heads):
        if isinstance(val, jax.ShapeDtypeStruct):
          new_shape = list(val.shape)
          new_shape[-2] = new_shape[-2] * kv_replication
          val = jax.ShapeDtypeStruct(tuple(new_shape), val.dtype)
        else:
          val = _jit_repeat_axes(val, ((-2, kv_replication),))
      return [(tgt_key, val) for _, tgt_key in group.targets]
    if group.op == "fuse_moe":
      raw_0 = src_flat[group.source_keys[0]]
      raw_1 = src_flat[group.source_keys[1]]
      wi_0, wi_1 = (
          _cast_target_free(_take_cycle_slot(raw_0, group, scan_axis, cycle, error_type), target_dtype, path),
          _cast_target_free(_take_cycle_slot(raw_1, group, scan_axis, cycle, error_type), target_dtype, path),
      )
      _check_scan_axis(wi_0, path, scan_axis, num_blocks, error_type)
      per_block = _fuse_moe_bulk_target_free(
          wi_0, wi_1, scan_axis, padded_base_moe_mlp_dim, moe_fused_layout, tp, kv_tp_size, moe_mlp_tp_size,
          error_type,
      )
    else:
      val = _cast_target_free(
          _take_cycle_slot(src_flat[group.source_keys[0]], group, scan_axis, cycle, error_type), target_dtype, path
      )
      _check_scan_axis(val, path, scan_axis, num_blocks, error_type)
      per_block = _slice_bulk_target_free(
          val, path, scan_axis, padded_base_moe_mlp_dim, kv_replication, base_num_kv_heads
      )
  else:
    first_tgt = tgt_flat[group.targets[0][1]]
    if group.op == "identity":
      val = _apply_dtype_cast(src_flat[group.source_keys[0]], first_tgt.dtype, path)
      out = _align_per_axis(val, first_tgt.shape, getattr(first_tgt, "sharding", None), path)
      return [(tgt_key, out) for _, tgt_key in group.targets]
    if group.op == "fuse_moe":
      wi_0, wi_1 = (
          _apply_dtype_cast(_take_cycle_slot(src_flat[key], group, scan_axis, cycle, error_type), first_tgt.dtype, path)
          for key in group.source_keys
      )
      _check_scan_axis(wi_0, path, scan_axis, num_blocks, error_type)
      per_block = _fuse_moe_bulk(
          wi_0, wi_1, first_tgt, path, scan_axis, moe_fused_layout, tp, kv_tp_size, moe_mlp_tp_size, error_type
      )
    else:
      val = _apply_dtype_cast(
          _take_cycle_slot(src_flat[group.source_keys[0]], group, scan_axis, cycle, error_type), first_tgt.dtype, path
      )
      _check_scan_axis(val, path, scan_axis, num_blocks, error_type)
      per_block = _bulk_align_and_unstack(val, scan_axis, first_tgt, path)
  return [(tgt_key, per_block[idx]) for idx, tgt_key in group.targets]
