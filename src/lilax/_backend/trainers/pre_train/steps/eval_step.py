"""Native pretraining evaluation step."""

from flax import nnx

from lilax._backend.layers.multi_token_prediction import calculate_mtp_acceptance_rate
from lilax._backend.trainers.pre_train.steps.loss import loss_fn


def eval_step(model, config, state, data, dropout_rng=None):
  del dropout_rng
  state = nnx.merge(model, state)
  loss, aux = loss_fn(state.model, config, data, None, None, is_train=False)
  mtp_acceptance_rate = 0.0
  if config.mtp_eval_target_module > 0:
    mtp_acceptance_rate = calculate_mtp_acceptance_rate(aux["intermediate_outputs"], config)
  metrics = {"scalar": {
      "evaluation/loss": loss,
      "evaluation/z_loss": aux.get("z_loss", 0.0),
      "evaluation/total_loss": aux["xent_sum"],
      "evaluation/total_weights": aux["total_weights"],
      "evaluation/moe_lb_loss": aux["moe_lb_loss"],
      "evaluation/indexer_loss": aux.get("indexer_loss", 0.0),
      "evaluation/mtp_loss": aux.get("mtp_loss", 0.0),
      "evaluation/mtp_acceptance_rate_percent": mtp_acceptance_rate,
  }}
  if config.moe_dropless_fallback == "step":
    metrics["has_moe_overflow"] = aux.get("has_moe_overflow", False)
  if config.moe_dropless_fallback == "layer":
    metrics["scalar"]["evaluation/moe_dropless_fallback_layers"] = aux["moe_dropless_fallback_layers"]
  return metrics
