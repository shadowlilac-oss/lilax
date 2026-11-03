"""Metric assembly for the native pretraining step."""

from flax import nnx
import jax.numpy as jnp

from lilax._backend.common.metric_logger import record_activation_metrics
from lilax._backend.utils.globals import EPS
from lilax._backend.utils.tree_ops import numerics


def loss_scalar_metrics(loss, aux):
  xent_sum = aux["xent_sum"]
  total_weights = aux["total_weights"]
  lm_loss = xent_sum / (total_weights + EPS)
  return {
      "learning/loss": loss, "learning/lm_loss": lm_loss, "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": aux.get("z_loss", 0.0), "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/indexer_loss": aux.get("indexer_loss", 0.0), "learning/mtp_loss": aux.get("mtp_loss", 0.0),
      "learning/total_weights": total_weights,
  }


def update_norm_metrics(scalar_metrics, config, grads, raw_grads, model):
  if not config.optimizer_memory_host_offload:
    scalar_metrics["learning/grad_norm"] = numerics.l2norm_pytree(grads)
    scalar_metrics["learning/raw_grad_norm"] = numerics.l2norm_pytree(raw_grads)
    model_params = nnx.state(model, nnx.Param)
    scalar_metrics["learning/param_norm"] = numerics.l2norm_pytree(model_params)


def update_skip_step_metric(scalar_metrics, config, optimizer):
  if config.skip_step_on_spikes:
    opt_state = nnx.to_pure_dict(nnx.state(optimizer)).get("opt_state", {})
    is_skipped = opt_state.get("is_skipped") if isinstance(opt_state, dict) else None
    if is_skipped is not None:
      scalar_metrics["optim/step_skipped"] = is_skipped.astype(jnp.float32)


def train_step_metrics(config, scalar_metrics, has_moe_overflow, aux, intermediate_outputs):
  metrics = {"scalar": scalar_metrics, "scalars": {}}
  if config.moe_dropless_fallback == "step":
    metrics["has_moe_overflow"] = has_moe_overflow if has_moe_overflow is not None else jnp.bool_(False)
  if getattr(config, "log_required_ragged_buffer_factor", False) and "moe_required_rbf" in aux:
    metrics["moe_required_rbf"] = aux["moe_required_rbf"]
  if getattr(config, "record_internal_nn_metrics", False):
    record_activation_metrics(metrics, intermediate_outputs, config)
  return metrics
