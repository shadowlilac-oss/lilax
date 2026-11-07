"""Adapt completed Tunix host steps to the shared Trainer lifecycle."""

from .scalar_metrics import trainer_scalar_metrics


class TrainerDataHooks:
  def __init__(self, data_hooks, lifecycle, initial_iteration):
    self._data_hooks = data_hooks
    self._lifecycle = lifecycle
    self._last_iteration = initial_iteration

  def __getattr__(self, name):
    return getattr(self._data_hooks, name)

  @property
  def eval_data_iterator(self):
    return self._data_hooks.eval_data_iterator

  @eval_data_iterator.setter
  def eval_data_iterator(self, value):
    self._data_hooks.eval_data_iterator = value

  def load_next_train_batch(self, trainer):
    completed_update = trainer.train_steps > self._lifecycle.state.global_step
    finish_step(trainer, self._lifecycle)
    lifecycle = self._lifecycle
    if lifecycle.control.should_training_stop or lifecycle.control.should_epoch_stop:
      return None
    batch = self._data_hooks.load_next_train_batch(trainer)
    if batch is None:
      return None
    if not completed_update and trainer.iter_steps > self._last_iteration:
      lifecycle._event("on_substep_end")
    if not lifecycle._step_active:
      lifecycle._begin_step()
    self._last_iteration = trainer.iter_steps
    if lifecycle.control.should_training_stop or lifecycle.control.should_epoch_stop:
      return None
    return batch


class TrainerIterator:
  def __init__(self, iterator, trainer, lifecycle):
    self._iterator = iter(iterator)
    self._trainer = trainer
    self._lifecycle = lifecycle
    self._last_iteration = trainer.iter_steps

  def __iter__(self):
    return self

  def __next__(self):
    trainer, lifecycle = self._trainer, self._lifecycle
    completed_update = trainer.train_steps > lifecycle.state.global_step
    finish_step(trainer, lifecycle)
    if lifecycle.control.should_training_stop or lifecycle.control.should_epoch_stop:
      raise StopIteration
    batch = next(self._iterator)
    if not completed_update and trainer.iter_steps > self._last_iteration:
      lifecycle._event("on_substep_end")
    if not lifecycle._step_active:
      lifecycle._begin_step()
    self._last_iteration = trainer.iter_steps
    if lifecycle.control.should_training_stop or lifecycle.control.should_epoch_stop:
      raise StopIteration
    return batch


def finish_step(trainer, lifecycle):
  if trainer.train_steps > lifecycle.state.global_step:
    metrics = trainer._prev_buffered_train_metrics or trainer._buffered_train_metrics
    lifecycle._end_step(trainer.train_steps, metrics.loss)
  if lifecycle.control.should_log:
    metrics = trainer._prev_buffered_train_metrics or trainer._buffered_train_metrics
    if metrics is not None:
      lifecycle.log(trainer_scalar_metrics(trainer, metrics))
  if lifecycle.control.should_evaluate:
    lifecycle.evaluate()
  manager = trainer.checkpoint_manager
  receipts, manager.save_receipts = manager.save_receipts, []
  if lifecycle.control.should_save and not any(receipt.outcome != "no_save" for receipt in receipts):
    manager.save(trainer.train_steps, trainer.model, trainer.optimizer,
                 save_only_lora_params=trainer._lora_enabled, force=True,
                 custom_metadata=trainer.custom_checkpoint_metadata())
    receipts.extend(manager.save_receipts)
    manager.save_receipts = []
  for receipt in receipts:
    lifecycle._resolve_save(receipt, manager._checkpointer)


def begin_training(trainer, lifecycle):
  trainer.is_managed_externally = True
  lifecycle.native_trainer = trainer
  lifecycle.callback_handler.optimizer = trainer.optimizer
  trainer.checkpoint_manager.lifecycle = lifecycle
  lifecycle._begin_training(trainer.train_steps)
  if lifecycle.args.eval_on_start:
    lifecycle.evaluate()
  if lifecycle.config.save_checkpoint_on_start and trainer.train_steps == 0:
    trainer.checkpoint_manager.save(
        trainer.train_steps, trainer.model, trainer.optimizer, save_only_lora_params=trainer._lora_enabled,
        custom_metadata=trainer.custom_checkpoint_metadata(),
    )
  finish_step(trainer, lifecycle)


def finish_training(trainer, lifecycle):
  finish_step(trainer, lifecycle)
  lifecycle._event("on_epoch_end")
  finish_step(trainer, lifecycle)
  if lifecycle.config.save_checkpoint_on_completion:
    trainer._save_last_checkpoint()
    finish_step(trainer, lifecycle)
  lifecycle._event("on_train_end")
  trainer.checkpoint_manager.allow_saves = False
