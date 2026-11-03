"""The traced computations of an engine step: forward/backward, optimizer update, and evaluation."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from lilax._backend.training_engine import gradient_sharding
from lilax._backend.training_engine import loss_invocation
from lilax._backend.training_engine import zero1
from lilax._backend.utils import max_utils
from lilax._backend.utils import maxtext_utils


def fwd_bwd_kernel(
    config: Any, loss_spec: loss_invocation.LossSpec, model_graphdef: Any, plan: gradient_sharding.GradientShardingPlan,
    params, rest, batch, acc_grads=None, acc_denom=None,
):
  loss_callable = loss_spec.resolve()

  def diff_wrapper(params_state, rest_state, inputs):
    model = nnx.merge(model_graphdef, params_state, rest_state, copy=True)
    out = loss_spec.call(loss_callable, config, model, inputs, is_train=True)
    _, _, new_rest = nnx.split(model, nnx.Param, ...)
    loss_out = loss_spec.normalize(out)
    return loss_out.primary_loss.unreduced_sum, (loss_out, new_rest)

  if plan.reduced_params is not None:
    params = jax.tree.map(jax.sharding.reshard, params, plan.reduced_params)
  grad_func = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)
  (_, (loss_out, new_rest)), micro_grads = grad_func(params, rest, batch)
  micro_grads = jax.tree.map(
      lambda grad: (
          grad.astype(config.grad_dtype)
          if hasattr(grad, "dtype") and jnp.issubdtype(grad.dtype, jnp.floating) and grad.dtype != config.grad_dtype
          else grad
      ),
      micro_grads,
  )
  denominator = loss_out.primary_loss.denominator.astype(jnp.float32)
  if acc_grads is None:
    return loss_out.primary_loss, loss_out.aux_metrics, new_rest, micro_grads, denominator
  acc_grads = jax.tree.map(jnp.add, acc_grads, micro_grads)
  return loss_out.primary_loss, loss_out.aux_metrics, new_rest, acc_grads, acc_denom + denominator


def _skipped_flag(local_state: Any, optimizer: Any) -> Any:
  opt_obj = getattr(local_state, "optimizer", optimizer)
  if opt_obj is None:
    return None
  opt_state = nnx.to_pure_dict(nnx.state(opt_obj)).get("opt_state", {})
  is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
  return None if is_skipped is None else is_skipped.astype(jnp.float32)


def update_kernel(
    config: Any, state_graphdef: Any, optimizer: Any, plan: gradient_sharding.GradientShardingPlan,
    state_pure, accumulated_grads, accumulated_denominator, mean_loss,
):
  grad_norm = None
  is_skipped_val = None
  if state_pure is None:
    return state_pure, grad_norm, is_skipped_val
  grad_target = plan.zero1_params
  if grad_target is None:
    grad_target = plan.plain_grads
  if grad_target is not None:
    accumulated_grads = jax.tree.map(jax.sharding.reshard, accumulated_grads, grad_target)
  if plan.zero1_params is not None:
    state_pure = zero1.reshard_model_params(state_pure, plan.zero1_params)
  has_weights = accumulated_denominator > 0
  safe_denominator = jnp.where(has_weights, accumulated_denominator, 1.0)
  grads = jax.tree.map(
      lambda grad: jnp.where(has_weights, grad / safe_denominator.astype(grad.dtype), jnp.zeros_like(grad)),
      accumulated_grads,
  )
  grad_norm = max_utils.l2norm_pytree(jax.tree.map(lambda grad: grad.astype(jnp.float32), grads))
  if config.gradient_clipping_threshold > 0:
    grads = maxtext_utils.apply_gradient_clipping(grads, None, config.gradient_clipping_threshold)
  local_state = nnx.merge(state_graphdef, state_pure, copy=True)
  if hasattr(local_state, "apply_gradients"):
    if config.skip_step_on_spikes:
      local_state.apply_gradients(grads, loss=mean_loss, grad_norm=grad_norm)
      is_skipped_val = _skipped_flag(local_state, optimizer)
    else:
      local_state.apply_gradients(grads)
  _, new_state_pure = nnx.split(local_state)
  if plan.zero1_params is not None:
    new_state_pure = zero1.reshard_model_params(new_state_pure, plan.gathered_params)
  return new_state_pure, grad_norm, is_skipped_val


def eval_kernel(config: Any, loss_spec: loss_invocation.LossSpec, model_graphdef: Any, params, rest, batch):
  loss_callable = loss_spec.resolve()
  model = nnx.merge(model_graphdef, params, rest, copy=True)
  loss_out = loss_spec.normalize(loss_spec.call(loss_callable, config, model, batch, is_train=False))
  return loss_out.primary_loss, loss_out.aux_metrics
