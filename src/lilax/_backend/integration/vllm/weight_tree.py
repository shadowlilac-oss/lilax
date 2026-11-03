"""Weight tree normalization and path handling for rollout conversion."""

import logging
from typing import Any, Dict, Mapping, Tuple

from flax import nnx, traverse_util


_NON_WEIGHT_PATH_PREFIXES = ("cache", "rngs", "intermediates")


def _is_non_weight_path(key_tuple: Tuple[Any, ...]) -> bool:
  return any(isinstance(part, str) and part.lstrip("_").startswith(_NON_WEIGHT_PATH_PREFIXES) for part in key_tuple)


def _non_param_paths(state: Any) -> frozenset:
  flat_state = getattr(state, "flat_state", None)
  if flat_state is None or not callable(flat_state):
    return frozenset()
  return frozenset(tuple(path) for path, var in flat_state() if not isinstance(var, nnx.Param))


def _rekey_to_target(flat_dotted: Dict[str, Any], target_state: Any) -> Dict[str, Any]:
  if target_state is None:
    return traverse_util.unflatten_dict(flat_dotted, sep=".")
  target_flat = traverse_util.flatten_dict(_to_pure_dict(target_state))
  by_dotted = {".".join(str(part) for part in key): key for key in target_flat}
  matched, unmatched = {}, []
  for dotted, value in flat_dotted.items():
    key = by_dotted.get(dotted)
    if key is None:
      unmatched.append(dotted)
    else:
      matched[key] = value
  if unmatched:
    logging.warning(
        "%d converted tensors have no counterpart in the rollout state and " "were dropped, e.g. %s",
        len(unmatched),
        sorted(unmatched)[:5],
    )
  return traverse_util.unflatten_dict(matched)


def _to_pure_dict(tree: Any) -> Mapping[str, Any]:
  if hasattr(tree, "to_pure_dict"):
    tree = tree.to_pure_dict()
  if hasattr(tree, "unfreeze"):
    tree = tree.unfreeze()
  if hasattr(tree, "items"):
    return {key: _to_pure_dict(value) for key, value in tree.items()}
  if hasattr(tree, "value"):
    return tree.value
  return tree


def _strip_root(flat: Mapping[Tuple[Any, ...], Any], root: str) -> Tuple[Dict[Tuple[Any, ...], Any], Tuple[Any, ...]]:
  stripped, depth = dict(flat), 0
  while stripped and all(key and key[0] == root for key in stripped):
    stripped = {key[1:]: value for key, value in stripped.items()}
    depth += 1
  return stripped, (root,) * depth
