"""NNX model parameter replacement and interpolation for DiLoCo."""

from typing import Any

from flax import nnx
import jax

from lilax._backend.common.train_state_nnx import TrainStateNNX
from lilax._backend.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator


def replace_nnx_model_params(state: Any, new_params: Any) -> Any:
  """Replace model parameters in an NNX train state or dictionary."""
  model = state["model"] if hasattr(state, "keys") else state.model
  optimizer = state["optimizer"] if hasattr(state, "keys") else state.optimizer
  graphdef, _, non_param_state = nnx.split(model, nnx.Param, ...)
  new_model = nnx.merge(graphdef, new_params, non_param_state)
  if isinstance(model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(model, dict):
    new_model = nnx.to_pure_dict(new_model)
  if hasattr(state, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(state)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      key = path[0]
      return getattr(key, "key", None) == "model" or getattr(key, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(path) else leaf for path, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    return TrainStateNNX(new_model, optimizer)


def replace_nnx_model_params_frag(
    state: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_frag_replica: dict[str, Any],
    alpha: float = 0.0,
) -> Any:
  """Replace one parameter fragment, optionally interpolating with the inner state."""
  model = state["model"] if hasattr(state, "keys") else state.model
  graphdef, full_params, non_param_state = nnx.split(model, nnx.Param, ...)
  full_params_dict = full_params.to_pure_dict()
  if alpha > 0.0:
    inner_frag = manipulator.get_flat_fragment(full_params_dict, frag_idx, has_replica_dim=False)
    merged_frag = jax.tree.map(lambda inner, outer: alpha * inner + (1 - alpha) * outer, inner_frag, outer_frag_replica)
  else:
    merged_frag = outer_frag_replica
  new_full_params = manipulator.apply_flat_fragment(full_params_dict, frag_idx, merged_frag, has_replica_dim=False)
  new_model = nnx.merge(graphdef, new_full_params, non_param_state)
  if isinstance(model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(model, dict):
    new_model = nnx.to_pure_dict(new_model)
  if hasattr(state, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(state)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      key = path[0]
      return getattr(key, "key", None) == "model" or getattr(key, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(path) else leaf for path, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    optimizer = state["optimizer"] if hasattr(state, "keys") else state.optimizer
    return TrainStateNNX(new_model, optimizer)
