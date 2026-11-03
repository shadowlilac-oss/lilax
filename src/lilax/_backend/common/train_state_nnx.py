"""The NNX Unified TrainState."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from lilax._backend.utils import max_logging


class TrainStateNNX(nnx.Module):
  """A unified container for NNX models and optimizers.

  This replaces Linen's TrainState for checkpointing.

  Linen TrainState pytree:
    {"params": {...}, "opt_state": {}...}
  TrainStateNNX state pytree:
    {"model": {...}, "optimizer": {"opt_state": {...}}}
  """

  def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer | None):
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any, **kwargs):
    """Updates the model and optimizer state, forwarding extra update arguments."""
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without"
          " an optimizer. This usually happens when the state was created for"
          " inference only."
      )
    self.optimizer.update(self.model, grads, **kwargs)


_NNX_RNG_STATE_KEYS = ("rngs", "dropout")


def _cast_step(step, dtype):
  """Casts concrete and abstract step values between NNX and Linen dtypes."""
  if isinstance(step, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(step.shape, dtype, sharding=getattr(step, "sharding", None))
  return jnp.asarray(step, dtype=dtype)


def _strip_rng_state(tree):
  """Removes NNX-only RNG subtrees and their empty parent dictionaries."""
  if not isinstance(tree, dict):
    return tree
  out = {}
  for k, v in tree.items():
    if k in _NNX_RNG_STATE_KEYS:
      continue
    stripped = _strip_rng_state(v)
    if isinstance(stripped, dict) and not stripped:
      continue
    out[k] = stripped
  return out


def _wrap_mu_nu_with_params(state):
  """Wraps mu/nu under an inner 'params' key (the Linen collection)."""
  if not isinstance(state, dict):
    return state
  return {k: {"params": v} if k in ("mu", "nu") and isinstance(v, dict) else v for k, v in state.items()}


def _as_chain_index(key):
  """Returns the int index for an int or digit-string key, else None."""
  if isinstance(key, int):
    return key
  if isinstance(key, str) and key.isdigit():
    return int(key)
  return None


def _opt_state_to_linen(opt_state):
  """Reshapes chained or flat NNX optimizer state to Linen's on-disk layout."""
  if not isinstance(opt_state, dict):
    return opt_state
  indices = [_as_chain_index(k) for k in opt_state.keys()]
  if not indices or any(i is None for i in indices):
    return _wrap_mu_nu_with_params(opt_state)
  chain = [None] * (max(indices) + 1)
  for key, idx in zip(opt_state.keys(), indices):
    chain[idx] = _wrap_mu_nu_with_params(opt_state[key])
  return chain


def to_linen_checkpoint_dict(nnx_pure_dict):
  """Reshapes a TrainStateNNX pure dict into the Linen on-disk layout."""
  if not isinstance(nnx_pure_dict, dict):
    return nnx_pure_dict
  result = {}
  if "model" in nnx_pure_dict:
    result["params"] = {"params": _strip_rng_state(nnx_pure_dict["model"])}
  optimizer = nnx_pure_dict.get("optimizer")
  if isinstance(optimizer, dict):
    if "step" in optimizer:
      result["step"] = _cast_step(optimizer["step"], jnp.int32)
    if "opt_state" in optimizer:
      result["opt_state"] = _opt_state_to_linen(optimizer["opt_state"])
  return result


def _strip_mu_nu_params(state):
  """Removes the inner 'params' wrap from mu/nu."""
  if not isinstance(state, dict):
    return state
  return {
      k: (v["params"] if k in ("mu", "nu") and isinstance(v, dict) and "params" in v else v) for k, v in state.items()
  }


def _opt_state_from_linen(opt_state):
  """Reshapes chained or flat Linen optimizer state to NNX layout."""
  if isinstance(opt_state, list):
    return {i: _strip_mu_nu_params(e) for i, e in enumerate(opt_state) if isinstance(e, dict)}
  if not isinstance(opt_state, dict):
    return opt_state
  return _strip_mu_nu_params(opt_state)


def from_linen_checkpoint_dict(linen_pure_dict):
  """Reshapes a Linen checkpoint into NNX layout without restoring NNX-only RNGs."""
  if not isinstance(linen_pure_dict, dict):
    return linen_pure_dict
  result = {}
  params = linen_pure_dict.get("params")
  if isinstance(params, dict) and "params" in params:
    result["model"] = params["params"]
  elif params is not None:
    result["model"] = params
  optimizer = {}
  if "step" in linen_pure_dict:
    optimizer["step"] = _cast_step(linen_pure_dict["step"], jnp.uint32)
  if "opt_state" in linen_pure_dict:
    optimizer["opt_state"] = _opt_state_from_linen(linen_pure_dict["opt_state"])
  if optimizer:
    result["optimizer"] = optimizer
  return result


def split_for_checkpoint(state: nnx.State):
  """Partitions state into Linen checkpoint, persistent auxiliary, and ephemeral state."""
  params, rng_state, batch_stats, caches, intermediates, rest = nnx.split_state(
      state, nnx.Param, nnx.RngState, nnx.BatchStat, nnx.Cache, nnx.Intermediate, ...
  )
  optimizer = nnx.State({"optimizer": rest["optimizer"]}) if "optimizer" in rest else nnx.State({})
  custom = nnx.State({"model": rest["model"]}) if "model" in rest else nnx.State({})
  linen_state = nnx.merge_state(params, optimizer)
  aux = nnx.merge_state(rng_state, batch_stats, custom)
  ephemeral = nnx.merge_state(caches, intermediates)
  return linen_state, aux, ephemeral


def to_checkpoint_dict(state: nnx.State | nnx.Module):
  """Reshapes concrete or abstract NNX state into the on-disk checkpoint layout."""
  if isinstance(state, nnx.Module):
    state = nnx.state(state)
  linen_state, aux_state, _ = split_for_checkpoint(state)
  pure = linen_state.to_pure_dict()
  linen_dict = to_linen_checkpoint_dict({"model": pure.get("model", {}), "optimizer": pure.get("optimizer", {})})
  aux = aux_state.to_pure_dict()
  if aux:
    linen_dict["nnx_aux"] = aux
  return linen_dict


def apply_checkpoint_aux(aux_state: nnx.State, nnx_aux: dict[str, Any]) -> None:
  """Fills auxiliary state from a checkpoint, skipping entries no longer present."""
  present = dict(nnx.to_flat_state(aux_state))
  flat = nnx.traversals.flatten_mapping(nnx_aux, is_leaf=lambda path, _: path in present)
  kept = {path: value for path, value in flat.items() if path in present}
  dropped = [path for path in flat if path not in present]
  if dropped:
    shown = ", ".join("/".join(str(k) for k in path) for path in sorted(dropped)[:20])
    more = f" (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""
    max_logging.log(f"Skipping {len(dropped)} nnx_aux entries absent from this model: {shown}{more}")
  if kept:
    for path, value in kept.items():
      current = present[path]
      present[path] = current.replace(value) if hasattr(current, "replace") else value
    aux_state.update(nnx.traversals.unflatten_mapping(present))
