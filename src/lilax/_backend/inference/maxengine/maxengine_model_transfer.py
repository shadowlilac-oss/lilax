"""Transfer persistent model state to MaxEngine inference graphs."""

import jax
from flax import nnx
from flax.core.spmd import logical_axis_rules

from lilax._backend.utils import sharding


def transfer_model_state(model, prefill_model, ar_model, mesh, axis_rules, rest_state):
  if type(model) is not type(prefill_model):
    raise ValueError("The loaded model is incompatible with the inference model type.")
  persistent = lambda path, var: not isinstance(var, (nnx.RngState, nnx.Cache, nnx.Intermediate, nnx.BatchStat))
  source = nnx.state(model).filter(persistent)
  prefill = nnx.state(prefill_model).filter(persistent)
  autoregressive = nnx.state(ar_model).filter(persistent)
  source_vars = dict(nnx.to_flat_state(source))
  prefill_vars = dict(nnx.to_flat_state(prefill))
  ar_vars = dict(nnx.to_flat_state(autoregressive))
  if source_vars.keys() != prefill_vars.keys() or prefill_vars.keys() != ar_vars.keys():
    raise ValueError("Loaded model persistent state does not match both inference graphs.")
  with logical_axis_rules(axis_rules):
    prefill_shardings = dict(nnx.to_flat_state(sharding.nnx_construct_named_sharding(prefill, mesh)))
    ar_shardings = dict(nnx.to_flat_state(sharding.nnx_construct_named_sharding(autoregressive, mesh)))
  transferred = []
  for path, target in prefill_vars.items():
    original = source_vars[path]
    ar_target = ar_vars[path]
    if type(original) is not type(target) or type(target) is not type(ar_target):
      raise ValueError(f"Incompatible variable type at {path}.")
    values = [var.get_value() for var in (original, target, ar_target)]
    structures = [jax.tree.structure(value) for value in values]
    if structures[0] != structures[1] or structures[1] != structures[2]:
      raise ValueError(f"Incompatible persistent state structure at {path}.")
    leaves = [jax.tree.leaves(value) for value in values]
    for source_leaf, target_leaf, ar_leaf in zip(*leaves):
      if (getattr(source_leaf, "shape", None), getattr(source_leaf, "dtype", None)) != (
          getattr(target_leaf, "shape", None), getattr(target_leaf, "dtype", None)
      ) or (getattr(target_leaf, "shape", None), getattr(target_leaf, "dtype", None)) != (
          getattr(ar_leaf, "shape", None), getattr(ar_leaf, "dtype", None)
      ):
        raise ValueError(f"Incompatible persistent state shape or dtype at {path}.")
    prefill_sharding = prefill_shardings[path].get_value()
    ar_sharding = ar_shardings[path].get_value()
    if jax.tree.structure(prefill_sharding) != jax.tree.structure(ar_sharding) or any(
        a != b for a, b in zip(jax.tree.leaves(prefill_sharding), jax.tree.leaves(ar_sharding))
    ):
      raise ValueError(f"Incompatible inference shardings at {path}.")
    transferred.append((path, target.replace(jax.device_put(values[0], prefill_sharding))))
  transferred_state = nnx.from_flat_state(transferred)
  params_state, persistent_rest = nnx.split_state(transferred_state, nnx.Param, ...)
  nnx.replace_by_pure_dict(rest_state, nnx.to_pure_dict(persistent_rest))
  return params_state
