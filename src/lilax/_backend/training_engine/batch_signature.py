"""Batch splitting into traced and closed-over halves, and the signature that decides recompilation."""

from __future__ import annotations

from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

_UNCOMPARABLE_SIGNATURE_WARNING = (
    "Could not compare %s between fwd_bwd calls (%s), so the engine cannot tell whether "
    "the compiled kernel is still valid and will recompile on EVERY fwd_bwd from now on. %s"
)
_UNCOMPARABLE_STATIC_HINT = (
    "This happens when a `gen_model_input_fn` returns a fresh object each call that holds "
    "an array -- a dataclass or SimpleNamespace whose `__eq__` then compares elementwise, "
    "where `bool()` is ambiguous. (Reusing one instance is fine: comparison short-circuits "
    "on identity.) Fix by returning arrays as top-level entries of the batch dict, so they "
    "are traced rather than closed over, or by holding the config object fixed across calls."
)
_UNCOMPARABLE_STRUCTURE_HINT = (
    "This is unexpected: the structural half is built from pytree treedefs, shapes and "
    "dtypes, which compare cleanly. Recompilation stays correct, but the cost is now paid "
    "every step, so this is worth reporting rather than living with."
)


def _is_jax_dynamic(value: Any) -> bool:
  leaves = jax.tree.leaves(value)
  if not leaves:
    return True
  return any(isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct, np.ndarray, np.generic)) for leaf in leaves)


def split_static_and_dynamic(batch: Any) -> tuple[Any, dict[str, Any]]:
  if not isinstance(batch, dict):
    return batch, {}
  dynamic: dict[str, Any] = {}
  static: dict[str, Any] = {}
  for key, value in batch.items():
    if isinstance(value, (int, float, bool)) or not _is_jax_dynamic(value):
      static[key] = value
    else:
      dynamic[key] = value
  return dynamic, static


def batch_signature(dynamic_batch: Any, static_batch: dict[str, Any]) -> Any:
  leaves, treedef = jax.tree.flatten(dynamic_batch)
  shapes = tuple((jnp.shape(leaf), jnp.result_type(leaf)) for leaf in leaves)
  return (treedef, shapes, static_batch)


def merge_static(dynamic: Any, static_batch: dict[str, Any]) -> Any:
  return {**dynamic, **static_batch} if isinstance(dynamic, dict) else dynamic


class RecompileDetector:
  """Compares batch signatures, warning once when a comparison cannot be made."""

  def __init__(self) -> None:
    self._warned = False

  def _warn_uncomparable(self, what: str, hint: str, exc: Exception) -> None:
    if self._warned:
      return
    self._warned = True
    logging.warning(_UNCOMPARABLE_SIGNATURE_WARNING, what, exc, hint)

  def needs_recompile(self, signature: Any, previous: Any) -> bool:
    if previous is None:
      return True
    try:
      if bool(previous[:2] != signature[:2]):
        return True
    except Exception as exc:
      self._warn_uncomparable("batch structure", _UNCOMPARABLE_STRUCTURE_HINT, exc)
      return True
    try:
      return bool(previous[2] != signature[2])
    except Exception as exc:
      self._warn_uncomparable("static loss arguments", _UNCOMPARABLE_STATIC_HINT, exc)
      return True
