"""Calling the configured loss function and normalizing its return into a `LossOutput`."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

from lilax._backend.trainers.pre_train.steps import loss as maxtext_loss
from lilax._backend.training_engine import abstract_engine


def _normalize_loss_output(out: Any, has_aux: bool) -> abstract_engine.LossOutput:
  if isinstance(out, abstract_engine.LossOutput):
    return out
  if isinstance(out, abstract_engine.WeightedMetric):
    return abstract_engine.LossOutput(primary_loss=out, aux_metrics={})
  if isinstance(out, (tuple, list)) and len(out) == 2:
    loss_val, aux = out
    if isinstance(loss_val, abstract_engine.WeightedMetric):
      primary_loss = loss_val
    elif isinstance(aux, dict) and "xent_sum" in aux and "total_weights" in aux:
      primary_loss = abstract_engine.WeightedMetric(unreduced_sum=aux["xent_sum"], denominator=aux["total_weights"])
    else:
      raise TypeError(
          f"Cannot construct WeightedMetric from 2-tuple loss return with elements "
          f"of type ({type(loss_val).__name__}, {type(aux).__name__}). Expected first element to be a "
          "WeightedMetric, or second element to be a dict containing 'xent_sum' and 'total_weights'."
      )
    return abstract_engine.LossOutput(primary_loss=primary_loss, aux_metrics=aux if (has_aux and isinstance(aux, dict)) else {})
  raise TypeError(
      f"Unsupported return type from loss function: {type(out)}. "
      "Expected abstract_engine.LossOutput, abstract_engine.WeightedMetric, "
      "or a 2-element tuple/list: (loss, aux_metrics)."
  )


@dataclasses.dataclass(frozen=True)
class LossSpec:
  """The loss to differentiate: a custom `fn` (MaxText's own loss when None) and how it is called."""

  fn: Callable[..., Any] | None
  has_aux: bool
  takes_kwargs: bool

  def call(self, loss_callable: Callable[..., Any], config: Any, model: Any, inputs: Any, is_train: bool) -> Any:
    if self.takes_kwargs:
      if not isinstance(inputs, dict):
        raise TypeError(
            "gen_model_input_fn must return a dict of loss-fn keyword arguments, got " f"{type(inputs).__name__}."
        )
      return loss_callable(model, **inputs)
    return loss_callable(model, config, inputs, None, None, is_train=is_train)

  def resolve(self) -> Callable[..., Any]:
    return self.fn if self.fn is not None else maxtext_loss.loss_fn

  def normalize(self, out: Any) -> abstract_engine.LossOutput:
    return _normalize_loss_output(out, self.has_aux)
