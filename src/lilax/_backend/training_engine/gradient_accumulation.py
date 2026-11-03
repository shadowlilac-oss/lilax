"""Gradients, denominators and losses accumulated across the micro steps of one train step."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from lilax._backend.training_engine import abstract_engine


class GradientAccumulation:
  """Micro-step accumulation state, folded in by fwd_bwd and consumed by update."""

  def __init__(self) -> None:
    self.grads: Any = None
    self.denominator: Any = None
    self.losses: list[abstract_engine.WeightedMetric | jax.Array] = []
    self.micro_step_count = 0

  def add(self, loss: Any, grads: Any, denominator: Any) -> None:
    self.losses.append(loss)
    self.grads = grads
    self.denominator = denominator
    self.micro_step_count += 1

  def mean_loss(self, skip_step_on_spikes: bool) -> Any:
    if not skip_step_on_spikes:
      return None
    if not self.losses:
      return jnp.array(0.0)
    loss_values = [loss.compute() if isinstance(loss, abstract_engine.WeightedMetric) else loss for loss in self.losses]
    return jnp.mean(jnp.stack(loss_values)) if len(loss_values) > 1 else loss_values[0]

  def reset(self) -> None:
    self.losses.clear()
    self.grads = None
    self.denominator = None
    self.micro_step_count = 0
