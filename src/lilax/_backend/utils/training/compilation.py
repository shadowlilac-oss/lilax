"""Compilation of training and evaluation steps."""

import functools

from flax.core.spmd import logical_axis_rules
import jax

from lilax._backend.trainers.diloco import diloco
from lilax._backend.observability import max_logging
from lilax._backend.utils import maxtext_utils
from lilax._backend.distributed import sharding


def jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings, mesh=None):
  """Return the compiled training step."""
  if config.enable_diloco:
    functional_train = train_step
    in_shardings = (state_mesh_shardings, data_sharding, None)
    out_shardings = (state_mesh_shardings, None)
    static_argnums = ()
    donate_argnums = 0
  else:
    functional_train, in_shardings, out_shardings, static_argnums, donate_argnums = (
        maxtext_utils.get_functional_train_with_signature(
            train_step, data_sharding, state_mesh_shardings, model, config, params_shardings
        )
    )
  if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    execution_mesh = mesh if mesh is not None else model.mesh
    execution_devices = execution_mesh.devices.flatten().tolist()
    with jax.set_mesh(execution_mesh), logical_axis_rules(config.logical_axis_rules):
      p_train_step = maxtext_utils.load_compiled(
          config, functional_train, state, execution_devices, data_sharding=data_sharding
      )
    max_logging.log("Loaded compiled function!")
  else:
    p_train_step = jax.jit(
        functional_train, in_shardings=in_shardings, out_shardings=out_shardings,
        static_argnums=static_argnums, donate_argnums=donate_argnums,
    )
  return p_train_step


def jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step):
  """Return the compiled evaluation step."""
  functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums = (
      maxtext_utils.get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config)
  )
  p_eval_step = None
  if config.compiled_trainstep_file == "":
    p_eval_step = jax.jit(
        functional_eval, in_shardings=in_shardings, out_shardings=out_shardings,
        static_argnums=static_argnums, donate_argnums=donate_argnums,
    )
  return p_eval_step


def jit_train_and_eval_step(
    config, model, mesh, state, state_mesh_shardings, train_step, eval_step=None,
    eval_data_iterator=None, params_shardings=None,
):
  """Return compiled training and evaluation steps."""
  if config.enable_diloco:
    train_step_partial = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
    if not isinstance(state_mesh_shardings, diloco.DiLoCoTrainState):
      raise TypeError(
          f"enable_diloco requires state_mesh_shardings to be a DiLoCoTrainState, got {type(state_mesh_shardings)}."
      )
    train_step = diloco.build_diloco_train_step(
        config, train_step_partial, mesh=mesh, outer_params_shardings=state_mesh_shardings.params
    )
  data_sharding_for_train = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules)
  data_sharding_for_eval = sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval)
  p_train_step = jit_train_step(
      config, model, state, state_mesh_shardings, data_sharding_for_train, train_step, params_shardings, mesh=mesh
  )
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(config, model, state_mesh_shardings, data_sharding_for_eval, eval_step)
  return p_train_step, p_eval_step
