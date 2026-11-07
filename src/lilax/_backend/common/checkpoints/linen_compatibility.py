"""Validate checkpoint weights and map legacy Linen collections."""

from typing import Any

from flax import nnx
import jax
from lilax._backend.common.checkpoints.lora_filter import _filter_lora_trainable_state
from lilax._backend.observability import max_logging


def _deep_merge_dicts(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
  out = dict(target)
  for key, value in source.items():
    if key in out and isinstance(out[key], dict) and isinstance(value, dict):
      out[key] = _deep_merge_dicts(out[key], value)
    else:
      out[key] = value
  return out


def _weight_mismatches(want, have, path=(), check_missing: bool = True, is_quantized_param: bool = False):
  if isinstance(want, dict):
    out = []
    is_quant = is_quantized_param or any(key in want for key in ("qvalue", "qarray"))
    for key, value in want.items():
      nested = have.get(key) if isinstance(have, dict) else None
      if check_missing or nested is not None:
        out.extend(_weight_mismatches(value, nested, path + (key,), check_missing=check_missing, is_quantized_param=is_quant))
    return out
  name = "/".join(str(part) for part in path)
  is_missing = have is None or (check_missing and isinstance(have, jax.ShapeDtypeStruct))
  if is_missing and ("lora_a" in name or "lora_b" in name or "rngs" in path or "rng" in path):
    return []
  if is_missing and is_quantized_param:
    return []
  if is_missing:
    return [(name, f"missing (model expects {getattr(want, 'shape', '?')} {getattr(want, 'dtype', '?')})")] if check_missing else []
  want_shape, got_shape = getattr(want, "shape", None), getattr(have, "shape", None)
  if want_shape is not None and got_shape is None:
    return [(name, f"structural mismatch: model expects a tensor but checkpoint provides {type(have).__name__}")]
  if want_shape is not None and got_shape is not None and tuple(want_shape) != tuple(got_shape):
    return [(name, f"shape {tuple(got_shape)} but the model expects {tuple(want_shape)}")]
  return []


def _lookup_path(tree, path):
  node = tree
  for key in path:
    if not isinstance(node, dict) or key not in node:
      return None
    node = node[key]
  return node


def _alias_legacy_collections(params_collection, stored_collection_meta):
  if not isinstance(stored_collection_meta, dict) or not stored_collection_meta:
    return params_collection, {}
  stored_params = stored_collection_meta.get("params")
  if not isinstance(stored_params, dict):
    return params_collection, {}
  patched = dict(params_collection)
  params_flat = dict(nnx.traversals.flatten_mapping(patched.get("params", {})))
  aliased = {}
  for col, tree in params_collection.items():
    if col == "params" or col in stored_collection_meta or not isinstance(tree, dict):
      continue
    tree_flat = dict(nnx.traversals.flatten_mapping(tree))
    moved = []
    for path, leaf in tree_flat.items():
      if _lookup_path(stored_params, path) is None:
        continue
      params_flat[path] = leaf
      moved.append(path)
    if moved:
      aliased[col] = moved
      moved_set = set(moved)
      remaining = {path: value for path, value in tree_flat.items() if path not in moved_set}
      if remaining:
        patched[col] = nnx.traversals.unflatten_mapping(remaining)
      else:
        patched.pop(col, None)
      max_logging.log(
          f"Checkpoint predates the '{col}' collection; restoring its {len(moved)} weight(s) "
          "from the 'params' collection instead of skipping them."
      )
  if aliased:
    patched["params"] = nnx.traversals.unflatten_mapping(params_flat)
  return patched, aliased


def _unalias_legacy_collections(restored_collection, aliased):
  if not aliased or not isinstance(restored_collection, dict):
    return restored_collection
  out = dict(restored_collection)
  params_flat = dict(nnx.traversals.flatten_mapping(out.get("params", {})))
  for col, paths in aliased.items():
    col_flat = dict(nnx.traversals.flatten_mapping(out.get(col, {})))
    for path in paths:
      if path in params_flat:
        col_flat[path] = params_flat.pop(path)
    if col_flat:
      out[col] = nnx.traversals.unflatten_mapping(col_flat)
  out["params"] = nnx.traversals.unflatten_mapping(params_flat)
  return out


def _log_unconsumed_checkpoint_weights(want, stored_collection):
  if not isinstance(want, dict) or not isinstance(stored_collection, dict):
    return
  try:
    want_paths = set(nnx.traversals.flatten_mapping(want).keys())
    stored_paths = set(nnx.traversals.flatten_mapping(stored_collection).keys())
  except Exception:
    return
  unconsumed = sorted("/".join(str(part) for part in path) for path in stored_paths - want_paths)
  if not unconsumed:
    return
  preview = ", ".join(unconsumed[:5])
  max_logging.log(
      f"WARNING: {len(unconsumed)} checkpoint weight(s) were not requested by the model and "
      f"will be ignored (the model will use its initialised values). First few: {preview}"
  )


def _expected_and_restored_params(abstract_nnx_state, restored_linen):
  want = nnx.split_state(abstract_nnx_state, nnx.Param, ...)[0].to_pure_dict().get("model", {})
  have = restored_linen.get("params", {}).get("params", {})
  return want, have


def _raise_weight_problems(problems):
  if not problems:
    return
  lines = "\n".join(f"  - '{path}': {reason}" for path, reason in problems)
  raise ValueError(
      "Checkpoint does not match the model:\n"
      f"{lines}\n"
      "Verify the checkpoint matches the model architecture (emb_dim, mlp_dim, num layers, scan_layers)."
  )


def _is_custom_projector_problem(path: str, want: dict) -> bool:
  parts = [part for part in path.replace(".", "/").split("/") if part and part != "params"]
  if len(parts) >= 2 and parts[0] == "vision_encoder":
    proj_name = parts[1]
    proj_dict = want.get("vision_encoder", {}).get(proj_name, {})
    if isinstance(proj_dict, dict) and any("custom_linear" in key for key in proj_dict.keys()):
      max_logging.warning(
          f"===Warning: weight mismatch found in custom vision projector: {proj_name}.\n"
          f"Path: {path}\n"
          "This custom vision projector will be initialized with random weights.==="
      )
      return True
  return False


def _raise_on_weight_mismatch(want, have, config=None):
  if config and getattr(getattr(config, "lora", None), "enable_lora", False):
    want = _filter_lora_trainable_state(want)
  problems = _weight_mismatches(want, have)
  problems = [(path, reason) for path, reason in problems if not _is_custom_projector_problem(path, want)]
  _raise_weight_problems(problems)
