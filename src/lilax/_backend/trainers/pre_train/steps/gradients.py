"""Loss and gradient computation for the native pretraining step."""

from flax import nnx
from flax.nnx import variablelib
import jax
from jax.sharding import NamedSharding

from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.trainers.pre_train.steps.loss import loss_fn
from lilax._backend.utils import maxtext_utils_nnx, sharding
from lilax._backend.utils.gradient_accumulation import gradient_accumulation_loss_and_grad
from lilax._backend.utils.xla_options import loop_unrolling


def _place_trainable_params(config, params_shardings, model, curr_params):
  if config.parameter_memory_host_offload:
    device_param_shardings = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_device, params_shardings, is_leaf=lambda value: isinstance(value, NamedSharding)
    )
    curr_params = jax.device_put(curr_params, device_param_shardings)
    nnx.update(model, curr_params)
  if config.shard_optimizer_over_data:
    param_sharding_lookup = {}
    for path, value in jax.tree_util.tree_leaves_with_path(
        params_shardings, is_leaf=lambda leaf: isinstance(leaf, (nnx.Variable, NamedSharding, jax.sharding.Sharding))
    ):
      param_sharding_lookup[path] = value.get_value() if isinstance(value, nnx.Variable) else value

    def _maybe_shard_param(path, var):
      if path in param_sharding_lookup:
        return sharding.maybe_shard_with_name(var, param_sharding_lookup[path], shard_mode=config.shard_mode)
      return var

    curr_params = jax.tree_util.tree_map_with_path(
        _maybe_shard_param, curr_params, is_leaf=lambda value: isinstance(value, nnx.Variable)
    )
    nnx.update(model, curr_params)
  return curr_params


def _single_pass_loss_and_grads(config, params_shardings, state, data):
  owg_type = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
  custom_param_filter = nnx.Any(owg_type)
  train_param_type = (
      getattr(nnx, "LoRAParam", nnx.Param)
      if getattr(getattr(config, "lora", None), "enable_lora", False)
      else nnx.Param
  )
  nnx.pop(state.model, nnx.Intermediate)
  model_graphdef, curr_params, custom_params, rest = nnx.split(state.model, train_param_type, custom_param_filter, ...)
  curr_params = _place_trainable_params(config, params_shardings, state.model, curr_params)

  def diff_wrapper(curr_params, custom_params, rest, config, data):
    local_model = nnx.merge(model_graphdef, curr_params, custom_params, rest, copy=True)
    loss, aux = loss_fn(local_model, config, data, None, None, is_train=True)
    non_param_rest = nnx.state(local_model, nnx.Not(nnx.Any(nnx.Param, custom_param_filter, nnx.Intermediate)))
    return loss, (aux, non_param_rest)

  grad_func = jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=True)
  with loop_unrolling.model_loop_unroll_context(config):
    (loss, (aux, non_param_rest)), (raw_grads, custom_grads) = grad_func(curr_params, custom_params, rest, config, data)
  nnx.update(state.model, nnx.State.merge(custom_grads, non_param_rest))
  return loss, aux, raw_grads


def compute_loss_and_grads(config, params_shardings, state, data):
  loss_model, loss_params, loss_rng = state.model, None, None
  if config.gradient_accumulation_steps > 1:
    loss, aux, raw_grads = gradient_accumulation_loss_and_grad(
        loss_fn, config, loss_model, loss_params, params_shardings, data, loss_rng
    )
    if lineage_adapter is not None and lineage_adapter.is_native(config):
      aux = lineage_adapter.finalize_accumulated_step(loss_model, aux)
    return loss, aux, raw_grads
  return _single_pass_loss_and_grads(config, params_shardings, state, data)
