import re
from typing import Any

import jax
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict


def unscan_layers(
    state: Any, num_layers: int, layer_container: str = "layers", scan_axis: int = 1, cycle_interval: int = 1,
) -> Any:
  if hasattr(state, "to_pure_dict"):
    pure = state.to_pure_dict()
  elif hasattr(state, "to_dict"):
    pure = state.to_dict()
  elif isinstance(state, dict):
    pure = state
  else:
    return state
  if cycle_interval <= 0:
    raise ValueError(f"unscan_layers: cycle_interval must be >= 1 (got {cycle_interval}).")
  if cycle_interval > 1 and num_layers % cycle_interval != 0:
    raise ValueError(
        f"unscan_layers: num_layers ({num_layers}) must be cleanly divisible by cycle_interval ({cycle_interval})."
    )
  flat = flatten_dict(pure)
  new_flat = {}
  unscanned_count = 0
  for key in list(flat.keys()):
    value = flat.pop(key)
    if layer_container not in key:
      new_flat[key] = value
      continue
    idx = key.index(layer_container)
    prefix = key[:idx]
    suffix = key[idx + 1:]
    slots = ((0, None),)
    if cycle_interval > 1:
      token = suffix[0] if suffix else "<empty>"
      if token == "local_layers":
        slots = tuple((index, index) for index in range(cycle_interval - 1))
      elif token == "global_layer":
        slots = ((cycle_interval - 1, None),)
      else:
        match = re.fullmatch(r"layer_(\d+)", token)
        if not match:
          raise ValueError(
              f"unscan_layers: cycle_interval={cycle_interval} > 1, but key {'.'.join(key)!r} "
              f"does not have a 'layer_<slot>' cycle prefix (or a nested 'local_layers'/'global_layer' "
              f"level) under {layer_container!r} (got {token!r})."
          )
        slot = int(match.group(1))
        if slot >= cycle_interval:
          raise ValueError(
              f"unscan_layers: slot {slot} parsed from key {'.'.join(key)!r} "
              f"must be less than cycle_interval {cycle_interval}."
          )
        slots = ((slot, None),)
      suffix = suffix[1:]
    arr = getattr(value, "value", value)
    if arr is None or not hasattr(arr, "shape") or getattr(arr, "ndim", 0) <= scan_axis:
      new_flat[key] = value
      continue
    del value
    expected = num_layers // cycle_interval
    if arr.shape[scan_axis] != expected:
      raise ValueError(
          f"unscan_layers: {'.'.join(key)!r} has shape {arr.shape}, expected axis {scan_axis} to be"
          f" {expected} (num_layers={num_layers}, cycle_interval={cycle_interval})."
      )
    nested = slots[0][1] is not None
    if nested and (arr.ndim <= scan_axis + 1 or arr.shape[scan_axis + 1] != len(slots)):
      raise ValueError(
          f"unscan_layers: {'.'.join(key)!r} has shape {arr.shape}, expected axis {scan_axis + 1} to hold"
          f" the {len(slots)} 'local_layers' cycle slots (cycle_interval={cycle_interval})."
      )
    for index in range(expected):
      block = jax.lax.index_in_dim(arr, index, axis=scan_axis, keepdims=False)
      for slot, local_index in slots:
        sliced = block
        if local_index is not None:
          sliced = jax.lax.index_in_dim(block, local_index, axis=scan_axis, keepdims=False)
        new_key = prefix + (f"{layer_container}_{index * cycle_interval + slot}",) + suffix
        new_flat[new_key] = sliced
      del block
    del arr
    unscanned_count += 1
  if unscanned_count == 0:
    raise ValueError(
        f"unscan_layers: found no scanned '{layer_container}' entries to unscan "
        "-- state may already be unscanned, or layer_container is wrong."
    )
  nested = unflatten_dict(new_flat)
  return jax.tree_util.tree_map(nnx.Param, nested)
