"""Graphdef variants and jitted train/eval programs for the native pretraining loop."""

import functools

from flax import nnx

from lilax._backend.trainers.pre_train.moe.graphdefs import (
    build_dropless_graphdef, build_eval_graphdef, build_first_phase_graphdef,
)
from lilax._backend.trainers.pre_train.steps.eval_step import eval_step
from lilax._backend.trainers.pre_train.steps.train_step import train_step
from lilax._backend.utils import elastic_utils
from lilax._backend.distributed import sharding
from lilax._backend.utils import train_utils


def _split_graphdefs(config, model, state):
  jit_model_dropless = None
  jit_model_eval = None
  jit_model_first_phase = None
  if config.enable_diloco:
    jit_model = model
  else:
    jit_model, state = nnx.split(state)
    if config.moe_dropless_fallback == "step":
      jit_model_dropless = build_dropless_graphdef(config, jit_model, state)
      jit_model_first_phase = build_first_phase_graphdef(config, jit_model, state)
    jit_model_eval = build_eval_graphdef(config, jit_model, state)
  return state, jit_model, jit_model_dropless, jit_model_eval, jit_model_first_phase


def build_step_programs(config, model, mesh, state, state_mesh_shardings, eval_data_iterator, elastic_retry_cache):
  state, jit_model, jit_model_dropless, jit_model_eval, jit_model_first_phase = _split_graphdefs(config, model, state)
  if config.enable_diloco:
    params_shardings = state_mesh_shardings.params
  else:
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  p_train_step, p_eval_step = elastic_utils.get_or_build(
      elastic_retry_cache,
      "train_and_eval_step",
      functools.partial(
          train_utils.jit_train_and_eval_step, config, jit_model, mesh, state, state_mesh_shardings,
          train_step, eval_step, eval_data_iterator, params_shardings,
      ),
  )
  if jit_model_eval is not None and p_eval_step is not None:
    p_eval_step = train_utils.jit_eval_step(
        config, jit_model_eval, state_mesh_shardings,
        sharding.get_input_data_sharding(config, mesh, rules=config.logical_axis_rules_for_eval), eval_step,
    )
  p_train_step_dropless = None
  p_eval_step_dropless = None
  if jit_model_dropless is not None:
    p_train_step_dropless, p_eval_step_dropless = train_utils.jit_train_and_eval_step(
        config, jit_model_dropless, mesh, state, state_mesh_shardings, train_step,
        eval_step=eval_step, eval_data_iterator=eval_data_iterator, params_shardings=params_shardings,
    )
  p_train_step_first_phase = None
  if jit_model_first_phase is not None:
    p_train_step_first_phase, _ = train_utils.jit_train_and_eval_step(
        config, jit_model_first_phase, mesh, state, state_mesh_shardings, train_step, params_shardings=params_shardings
    )
  programs = {
      "p_train_step": p_train_step, "p_train_step_dropless": p_train_step_dropless,
      "p_train_step_first_phase": p_train_step_first_phase, "p_eval_step": p_eval_step,
      "p_eval_step_dropless": p_eval_step_dropless,
  }
  return state, state_mesh_shardings, programs
