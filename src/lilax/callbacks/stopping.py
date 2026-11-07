"""Loss-based stopping through completed-step and evaluation events."""

import math

from transformers import TrainerCallback


class TargetEvalLossCallback(TrainerCallback):
  def __init__(self, target_eval_loss=0.0):
    self.target_eval_loss = target_eval_loss

  def on_evaluate(self, args, state, control, metrics=None, metric_key_prefix="eval", **kwargs):
    loss = (metrics or {}).get(f"{metric_key_prefix}_loss")
    if self.target_eval_loss and loss is not None and loss <= self.target_eval_loss:
      control.should_training_stop = True
    return control


class NonFiniteLossCallback(TrainerCallback):
  def __init__(self, *, abort_on_nan_loss=True, abort_on_inf_loss=True):
    self.abort_on_nan_loss = abort_on_nan_loss
    self.abort_on_inf_loss = abort_on_inf_loss

  def _runtime_overrides(self):
    return {"abort_on_nan_loss": self.abort_on_nan_loss, "abort_on_inf_loss": self.abort_on_inf_loss}

  def on_step_end(self, args, state, control, loss=None, **kwargs):
    if loss is not None and (self.abort_on_nan_loss and math.isnan(loss)
                             or self.abort_on_inf_loss and math.isinf(loss)):
      raise SystemExit(1)
