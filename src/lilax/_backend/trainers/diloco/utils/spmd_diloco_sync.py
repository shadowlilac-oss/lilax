"""Collective state synchronization and initialization for SPMD DiLoCo."""

from typing import Any

import drjax
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax

from lilax._backend.trainers.diloco import diloco
from lilax._backend.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator
from lilax._backend.trainers.diloco.utils.nnx_state_utils import replace_nnx_model_params, replace_nnx_model_params_frag
from lilax._backend.utils.diloco_sharding import add_diloco_to_sharding


def synchronize_full_state(
    state: Any, outer_optimizer: optax.GradientTransformation, mesh: jax.sharding.Mesh | None = None
) -> Any:
  """Synchronize all parameters across DiLoCo replicas."""
  broadcast_outer_params = drjax.broadcast(state.params, mesh=mesh)
  _, inner_model_params, _ = nnx.split(state.inner_state.model, nnx.Param, ...)
  inner_model_params = inner_model_params.to_pure_dict()
  model_delta = jax.tree.map(lambda inner, outer: outer - inner, inner_model_params, broadcast_outer_params)
  averaged_pseudo_grad = drjax.reduce_mean(model_delta)
  updates, new_opt_state = outer_optimizer.update(averaged_pseudo_grad, state.outer_opt_state, state.params)
  new_outer_params = optax.apply_updates(state.params, updates)
  new_inner_state = drjax.map_fn(
      lambda inner_state: replace_nnx_model_params(inner_state, new_outer_params), state.inner_state, mesh=mesh
  )
  return state.replace(params=new_outer_params, outer_opt_state=new_opt_state, inner_state=new_inner_state)


def synchronize_fragment_state(
    state: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Synchronize one parameter fragment across DiLoCo replicas."""
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  inner_model_params = nnx.filter_state(state.inner_state.model, nnx.Param).to_pure_dict()
  inner_params_frag = manipulator.get_flat_fragment(inner_model_params, frag_idx, has_replica_dim=True)
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)
  unreduced_grads = jax.tree.map(lambda outer, inner: outer - inner, broadcast_outer_frag, inner_params_frag)
  averaged_pseudo_grad = drjax.reduce_mean(unreduced_grads)
  trace_frag = manipulator.get_flat_fragment(state.outer_opt_state[0].trace, frag_idx, has_replica_dim=False)
  opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())
  updates_frag, new_opt_state_frag = outer_optimizer.update(
      averaged_pseudo_grad, opt_state_frag, params=outer_params_frag
  )
  new_outer_params_frag = optax.apply_updates(outer_params_frag, updates_frag)
  new_params = manipulator.apply_flat_fragment(state.params, frag_idx, new_outer_params_frag, has_replica_dim=False)
  new_trace = manipulator.apply_flat_fragment(
      state.outer_opt_state[0].trace, frag_idx, new_opt_state_frag[0].trace, has_replica_dim=False
  )
  new_outer_opt_state = (optax.TraceState(trace=new_trace), state.outer_opt_state[1])
  return state.replace(params=new_params, outer_opt_state=new_outer_opt_state)


def apply_fragment_to_inner_state(
    state: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    alpha: float = 0.0,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Broadcast a synchronized fragment and update each replica's inner state."""
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)
  new_inner_state = drjax.map_fn(
      lambda inner_state, frag: replace_nnx_model_params_frag(inner_state, manipulator, frag_idx, frag, alpha=alpha),
      (state.inner_state, broadcast_outer_frag),
      mesh=mesh,
  )
  return state.replace(inner_state=new_inner_state)


def setup_diloco_initial_state(
    state: Any, config: Any, mesh: jax.sharding.Mesh, state_mesh_shardings: PyTree, restored: Any = None
) -> Any:
  """Build a DiLoCo train state from a single-replica state."""
  del restored
  if isinstance(state, diloco.DiLoCoTrainState):
    return state
  inner_state_shardings = add_diloco_to_sharding(state_mesh_shardings)
  if hasattr(state, "model"):
    _, outer_params, _ = nnx.split(state.model, nnx.Param, ...)
    outer_params = outer_params.to_pure_dict() if hasattr(outer_params, "to_pure_dict") else outer_params
  else:
    outer_params = getattr(state, "params", state)

  def _broadcast_to_replicas(leaf, sharding):
    if hasattr(leaf, "shape"):
      target_shape = (config.num_diloco_replicas, *leaf.shape)
      if isinstance(leaf, jax.ShapeDtypeStruct):
        sharding_arg = sharding if isinstance(sharding, jax.sharding.NamedSharding) else None
        return jax.ShapeDtypeStruct(target_shape, leaf.dtype, sharding=sharding_arg)
      if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.jit(lambda value: jnp.broadcast_to(value, target_shape), out_shardings=sharding)(leaf)
      return jnp.broadcast_to(leaf, target_shape)
    return leaf

  inner_state = jax.tree_util.tree_map(_broadcast_to_replicas, state, inner_state_shardings)
  outer_optimizer = optax.sgd(config.diloco_outer_lr, momentum=config.diloco_outer_momentum, nesterov=True)
  outer_opt_state = outer_optimizer.init(outer_params)
  step = getattr(getattr(state, "optimizer", None), "step", jnp.array(0, dtype=jnp.int32))
  return diloco.DiLoCoTrainState(inner_state=inner_state, params=outer_params, outer_opt_state=outer_opt_state, step=step)
