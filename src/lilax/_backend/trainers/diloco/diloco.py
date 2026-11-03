"""Distributed low-communication training state and train steps."""

from typing import Any, Callable

import drjax
from flax import nnx, struct
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int32, Key, PyTree, UInt32
import optax

from lilax._backend.trainers.diloco.fragmenter import FragmentedTreeManipulator, get_streaming_schedule
from lilax._backend.trainers.diloco import sync as spmd_diloco_sync
from lilax._backend.utils.diloco_sharding import add_diloco_to_sharding, extract_per_island_metrics

Batch = Any
Params = PyTree
Metrics = PyTree
OptState = optax.OptState
InnerOptStates = optax.OptState
PRNGKey = Key[Array, ""] | UInt32[Array, "2"]
Step = Int32[Array, ""]


class DiLoCoTrainState(struct.PyTreeNode):
  """The inner states, outer parameters, optimizer state and step of DiLoCo."""

  inner_state: Any
  params: Params
  outer_opt_state: OptState
  step: Step


def build_abstract_diloco_state(
    config: Any, abstract_state: PyTree, state_mesh_shardings: PyTree, mesh: jax.sharding.Mesh
) -> tuple[DiLoCoTrainState, DiLoCoTrainState, PyTree]:
  """Build abstract DiLoCo state and shardings for ahead-of-time compilation."""
  def add_diloco_dim(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
      return jax.ShapeDtypeStruct((config.num_diloco_replicas,) + tuple(value.shape), value.dtype)
    return value

  inner_state = jax.tree.map(add_diloco_dim, abstract_state)
  outer_optimizer = optax.sgd(config.diloco_outer_lr, momentum=config.diloco_outer_momentum, nesterov=True)
  _, model_params, _ = nnx.split(abstract_state.model, nnx.Param, ...)
  model_params = model_params.to_pure_dict()
  _, model_params_sharding, _ = nnx.split(state_mesh_shardings.model, nnx.Param, ...)
  model_params_sharding = model_params_sharding.to_pure_dict()
  outer_opt_state = jax.eval_shape(outer_optimizer.init, model_params)
  diloco_state = DiLoCoTrainState(
      inner_state=inner_state, params=model_params, outer_opt_state=outer_opt_state,
      step=jax.ShapeDtypeStruct((), jnp.int32),
  )
  inner_state_shardings = add_diloco_to_sharding(state_mesh_shardings)
  outer_opt_state_sharding = (optax.TraceState(trace=model_params_sharding), optax.EmptyState())
  diloco_state_shardings = DiLoCoTrainState(
      inner_state=inner_state_shardings, params=model_params_sharding,
      outer_opt_state=outer_opt_state_sharding, step=None,
  )
  return diloco_state, diloco_state_shardings, inner_state_shardings


def build_diloco_state(
    config: Any, initialize_state: Callable[[], Any], mesh: jax.sharding.Mesh | None = None
) -> tuple[DiLoCoTrainState, PyTree]:
  """Construct a DiLoCo state from a regular training state."""
  outer_optimizer = optax.sgd(config.diloco_outer_lr, momentum=config.diloco_outer_momentum, nesterov=True)

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def init_diloco_state() -> tuple[DiLoCoTrainState, PyTree]:
    state = initialize_state()
    inner_state = drjax.broadcast(state, mesh=mesh)
    _, outer_params, _ = nnx.split(state.model, nnx.Param, ...)
    outer_params = outer_params.to_pure_dict()
    outer_opt_state = outer_optimizer.init(outer_params)
    outer_opt_state_sharding = jax.tree_util.tree_map(lambda value: value.sharding, outer_opt_state)
    step = state.optimizer.step
    return (
        DiLoCoTrainState(inner_state=inner_state, params=outer_params, outer_opt_state=outer_opt_state, step=step),
        outer_opt_state_sharding,
    )

  return init_diloco_state()


def build_vanilla_diloco_train_step(
    config: Any, train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local train step into a vanilla DiLoCo train step."""
  outer_optimizer = optax.sgd(config.diloco_outer_lr, momentum=config.diloco_outer_momentum, nesterov=True)

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def vanilla_diloco_train_step(state: DiLoCoTrainState, batch: Batch, prng: PRNGKey):
    keys = jax.random.split(prng, config.num_diloco_replicas) if prng is not None else None
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, keys), mesh=mesh)
    default_metrics = extract_per_island_metrics(metrics, config.num_diloco_replicas)
    new_step = inner_state.optimizer.step[0]
    state = state.replace(inner_state=inner_state, step=new_step)
    state = jax.lax.cond(
        new_step % config.diloco_sync_period == 0,
        lambda current: spmd_diloco_sync.synchronize_full_state(current, outer_optimizer, mesh=mesh),
        lambda current: current, state,
    )
    return state, default_metrics

  return vanilla_diloco_train_step


def build_streaming_diloco_train_step(
    config: Any, train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None, outer_params_shardings: PyTree | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local train step into a streaming DiLoCo train step."""
  if getattr(config, "diloco_bucketize_non_scanned", False) and outer_params_shardings is None:
    raise ValueError("diloco_bucketize_non_scanned requires outer_params_shardings for SPMD streaming DiLoCo.")
  outer_optimizer = optax.sgd(config.diloco_outer_lr, momentum=config.diloco_outer_momentum, nesterov=True)
  num_fragments = config.num_diloco_fragments
  steps_between_syncs, period = get_streaming_schedule(config)
  delay_v = config.num_communication_overlapping_steps
  alpha = config.communication_overlapping_alpha

  @drjax.program(placements={"diloco": config.num_diloco_replicas})
  def streaming_diloco_train_step(state: DiLoCoTrainState, batch: Batch, prng: PRNGKey):
    keys = jax.random.split(prng, config.num_diloco_replicas) if prng is not None else None
    inner_state, metrics = drjax.map_fn(train_step, (state.inner_state, batch, keys), mesh=mesh)
    default_metrics = extract_per_island_metrics(metrics, config.num_diloco_replicas)
    new_step = inner_state.optimizer.step[0]
    state = state.replace(inner_state=inner_state, step=new_step)
    manipulator = FragmentedTreeManipulator.create(state.params, config, shardings=outer_params_shardings)
    is_sync_step = (new_step > 0) & (new_step % steps_between_syncs == 0)

    def do_sync(current):
      frag_idx = (new_step % period) // steps_between_syncs
      return jax.lax.switch(
          frag_idx,
          [lambda arg, idx=index: spmd_diloco_sync.synchronize_fragment_state(arg, manipulator, idx, outer_optimizer, mesh=mesh)
           for index in range(num_fragments)],
          current,
      )

    state = jax.lax.cond(is_sync_step, do_sync, lambda current: current, state)
    is_apply_step = (new_step - delay_v > 0) & ((new_step - delay_v) % steps_between_syncs == 0)

    def do_apply(current):
      frag_idx = ((new_step - delay_v) % period) // steps_between_syncs
      return jax.lax.switch(
          frag_idx,
          [lambda arg, idx=index: spmd_diloco_sync.apply_fragment_to_inner_state(arg, manipulator, idx, alpha=alpha, mesh=mesh)
           for index in range(num_fragments)],
          current,
      )

    state = jax.lax.cond(is_apply_step, do_apply, lambda current: current, state)
    return state, default_metrics

  return streaming_diloco_train_step


def build_diloco_train_step(
    config: Any, train_step: Callable[[Any, Batch, PRNGKey], tuple[Any, Metrics]],
    mesh: jax.sharding.Mesh | None = None, outer_params_shardings: PyTree | None = None,
) -> Callable[[DiLoCoTrainState, Batch, PRNGKey], tuple[DiLoCoTrainState, Metrics]]:
  """Convert a local train step to the configured DiLoCo variant."""
  if config.enable_streaming_diloco:
    return build_streaming_diloco_train_step(config, train_step, mesh=mesh, outer_params_shardings=outer_params_shardings)
  return build_vanilla_diloco_train_step(config, train_step, mesh=mesh)
