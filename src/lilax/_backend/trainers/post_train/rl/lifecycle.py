"""Adapt Tunix actor-update calls to completed-step Trainer events."""

import sys

from tunix.rl.rl_cluster import RLEngine

from lilax._backend.trainers.post_train.lifecycle import TrainerIterator, finish_step, finish_training
from lilax._backend.trainers.stop_training import StopTraining


class TrainerRLEngine(RLEngine):
  def __init__(self, *, lifecycle, evaluation_only=False, **kwargs):
    self._lifecycle = lifecycle
    self._evaluation_only = evaluation_only
    self._lifecycle_finished = False
    super().__init__(**kwargs)

  def update_actor(self, train_ds, eval_ds, skip_jit=False):
    lifecycle = self._lifecycle
    super().update_actor(TrainerIterator(train_ds, self.actor_trainer, lifecycle), eval_ds, skip_jit)
    finish_step(self.actor_trainer, lifecycle)
    if (lifecycle.control.should_epoch_stop
        or lifecycle.control.should_training_stop and self.actor_trainer.train_steps < lifecycle.config.steps):
      raise StopTraining("A Trainer callback stopped the actor updates.")

  def close(self):
    if self._lifecycle_finished:
      return
    try:
      if (self._lifecycle._started and not self._evaluation_only
          and (sys.exc_info()[0] is None or isinstance(sys.exc_info()[1], StopTraining))):
        finish_training(self.actor_trainer, self._lifecycle)
    finally:
      self.actor_trainer.checkpoint_manager.allow_saves = False
      super().close()
      self._lifecycle_finished = True
