"""Native pretraining train step."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.trainers.pre_train.moe.bias import apply_routed_bias_updates
from lilax._backend.trainers.pre_train.moe.capacity_aux import moe_capacity_scalar_metrics
from lilax._backend.trainers.pre_train.moe.overflow import _rollback_on_overflow
from lilax._backend.trainers.pre_train.steps.diagnostics import step_diagnostics
from lilax._backend.trainers.pre_train.steps.gradients import compute_loss_and_grads
from lilax._backend.trainers.pre_train.steps.train_metrics import (
    loss_scalar_metrics, train_step_metrics, update_norm_metrics, update_skip_step_metric,
)
from lilax._backend.utils import maxtext_utils, maxtext_utils_nnx
from lilax._backend.trainers.pre_train.steps import qk_clip
from lilax._backend.device_memory import memory_kinds
from lilax._backend.pytree import numerics


def _apply_gradients(config, state_mesh_shardings, state, grads, loss):
  if config.optimizer_memory_host_offload:
    device_opt_shardings = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_device, state_mesh_shardings.optimizer,
        is_leaf=lambda value: isinstance(value, NamedSharding),
    )
    opt_state = nnx.state(state.optimizer)
    new_opt_state = jax.device_put(opt_state, device_opt_shardings)
    nnx.update(state.optimizer, new_opt_state)
  if config.skip_step_on_spikes:
    grad_norm = numerics.l2norm_pytree(grads)
    state.apply_gradients(grads, loss=loss, grad_norm=grad_norm)
  else:
    state.apply_gradients(grads)


def train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
  del dropout_rng
  input_state_leaves = jax.tree_util.tree_leaves(state) if config.moe_dropless_fallback == "step" else None
  state = nnx.merge(model, state)
  loss, aux, raw_grads = compute_loss_and_grads(config, params_shardings, state, data)

  raw_grads = jax.tree_util.tree_map(
      lambda grad: grad.astype(config.grad_dtype) if grad.dtype == jnp.float32 else grad, raw_grads
  )
  if config.parameter_memory_host_offload:
    raw_grads = jax.device_put(raw_grads, memory_kinds.with_memory_kind(params_shardings, "device"))
  intermediate_outputs = aux["intermediate_outputs"]
  has_moe_overflow = aux.get("has_moe_overflow")
  bias_metrics = {}
  diag_bias_values, diag_bias_updates = [], []
  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(raw_grads, None, config.gradient_clipping_threshold)
  else:
    grads = raw_grads
  _apply_gradients(config, state_mesh_shardings, state, grads, loss)
  new_state = state
  if lineage_adapter.is_native(config):
    diag_bias_values = list(aux.get("diag_bias_values", ()))
    diag_bias_updates = list(aux.get("diag_bias_updates", ()))
  elif config.routed_bias and config.routed_bias_update_rate > 0.0:
    bias_metrics, diag_bias_values, diag_bias_updates = apply_routed_bias_updates(new_state.model, aux, config)
  scalar_metrics = loss_scalar_metrics(loss, aux)
  scalar_metrics.update(moe_capacity_scalar_metrics(config, aux))
  scalar_metrics.update(bias_metrics)
  if config.use_qk_clip:
    new_state = qk_clip.apply_qk_clip_nnx(new_state, intermediate_outputs, config)
    global_max_logit = qk_clip.calculate_max_logit_metric(intermediate_outputs)
    if global_max_logit is not None:
      scalar_metrics["learning/max_logits"] = global_max_logit
  update_norm_metrics(scalar_metrics, config, grads, raw_grads, new_state.model)
  if getattr(config, "log_step_diagnostics", False):
    scalar_metrics.update(step_diagnostics(raw_grads, has_moe_overflow, diag_bias_values, diag_bias_updates))
    if "learning/raw_grad_norm" not in scalar_metrics:
      scalar_metrics["learning/raw_grad_norm"] = numerics.l2norm_pytree(raw_grads)
      scalar_metrics["learning/grad_norm"] = numerics.l2norm_pytree(grads)
      scalar_metrics["learning/param_norm"] = numerics.l2norm_pytree(nnx.state(new_state.model, nnx.Param))
  update_skip_step_metric(scalar_metrics, config, new_state.optimizer)
  metrics = train_step_metrics(config, scalar_metrics, has_moe_overflow, aux, intermediate_outputs)
  out_state = nnx.state(new_state, nnx.Not(nnx.Intermediate))
  if input_state_leaves is not None and has_moe_overflow is not None:
    out_state = _rollback_on_overflow(out_state, input_state_leaves, has_moe_overflow)
  return out_state, metrics
