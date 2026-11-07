"""Hugging Face callback dispatch and public Trainer event state."""

import time
import uuid
from contextlib import contextmanager

import jax
from lilax._backend.configs.pyconfig import HyperParameters
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback
from transformers.trainer_utils import TrainOutput

from lilax.callbacks import MetricsCallback, TensorBoardCallback, TrackioCallback, WandbCallback
from ..telemetry._profile_artifacts import log_profile_artifacts
from lilax.callbacks._configuration import callback_overrides


class TrainingFlowCallback(DefaultFlowCallback):
  def on_step_end(self, args, state, control, **kwargs):
    control.should_log |= args.logging_steps > 0 and state.global_step % args.logging_steps == 0
    control.should_evaluate |= args.eval_steps > 0 and state.global_step % args.eval_steps == 0
    control.should_training_stop |= state.global_step >= state.max_steps
    return control

  def on_epoch_end(self, args, state, control, **kwargs):
    return control


class TrainerEvents:
  def _init_events(self, callbacks):
    self.state = TrainerState(max_steps=self.args.max_steps, num_train_epochs=self.args.num_train_epochs,
                              is_local_process_zero=True,
                              is_world_process_zero=jax.process_index() == 0)
    self.control = TrainerControl()
    callbacks = [callback() if isinstance(callback, type) else callback for callback in (callbacks or [])]
    for name, callback_type in (("tensorboard", TensorBoardCallback), ("wandb", WandbCallback), ("trackio", TrackioCallback)):
      if name in self.args.report_to and not any(isinstance(callback, callback_type) for callback in callbacks):
        callbacks.append(callback_type())
    for callback in callbacks:
      if isinstance(callback, (TensorBoardCallback, MetricsCallback)):
        callback._config = self.config
    self.callback_handler = CallbackHandler([TrainingFlowCallback(), *callbacks], self.model,
                                            self.processing_class, self.optimizers[0], self.optimizers[1])
    self._event("on_init_end")
    self._loss_sum = 0.0
    self._loss_count = 0
    self._started = False
    self._step_active = False
    self._save_receipts = {}

  def _event(self, name, **kwargs):
    self.control = getattr(self.callback_handler, name)(self.args, self.state, self.control, **kwargs)
    return self.control

  @contextmanager
  def _reporting_context(self):
    if self._started:
      self._started = False
      self._step_active = False
      self._loss_sum, self._loss_count = 0.0, 0
      self.control = TrainerControl()
    try:
      yield
    finally:
      for callback in self.callback_handler.callbacks:
        if isinstance(callback, (TensorBoardCallback, WandbCallback, TrackioCallback)):
          callback._close()

  def _log_capture(self, *, logdir, step, process_index):
    trackio_enabled = any(isinstance(callback, TrackioCallback) and callback._active
                         for callback in self.callback_handler.callbacks)
    wandb_run = next((callback.run for callback in self.callback_handler.callbacks
                     if isinstance(callback, WandbCallback) and callback.run is not None), None)
    log_profile_artifacts(logdir, step, process_index, trackio_enabled, wandb_run)

  def add_callback(self, callback):
    callback = callback() if isinstance(callback, type) else callback
    if isinstance(callback, (TensorBoardCallback, MetricsCallback)):
      callback._config = self.config
    if isinstance(callback, MetricsCallback):
      callback._bind(self.config)
    overrides = callback_overrides([callback])
    if overrides:
      self.config = HyperParameters(self.config._pydantic_config.model_copy(update=overrides))
    self.callback_handler.add_callback(callback)

  def pop_callback(self, callback):
    removed = self.callback_handler.pop_callback(callback)
    if isinstance(removed, (TensorBoardCallback, WandbCallback, TrackioCallback)):
      removed._close()
    return removed

  def remove_callback(self, callback):
    selected = callback
    if isinstance(callback, type):
      selected = next((item for item in self.callback_handler.callbacks if isinstance(item, callback)), None)
    self.callback_handler.remove_callback(callback)
    if isinstance(selected, (TensorBoardCallback, WandbCallback, TrackioCallback)):
      selected._close()

  def log(self, logs, start_time=None):
    values = dict(logs)
    if start_time is not None:
      values["train_runtime"] = time.time() - start_time
    self.state.log_history.append(values | {"step": self.state.global_step})
    self._event("on_log", logs=values)

  def _train_output(self):
    loss = self._loss_sum / max(1, self._loss_count)
    return TrainOutput(self.state.global_step, loss, {"train_loss": loss})

  def _begin_training(self, step):
    self.state.global_step = step
    self._step_active = False
    self.state.max_steps = self.config.steps
    if not self._started:
      self._started = True
      self._event("on_train_begin")
      self._event("on_epoch_begin")
    self.control.should_training_stop |= step >= self.state.max_steps

  def _save_action(self, completed_step):
    return {"action_id": uuid.uuid4().hex, "completed_step": completed_step}

  def _begin_step(self):
    self._step_active = True
    pending = {name: getattr(self.control, name) for name in ("should_log", "should_evaluate", "should_save")}
    self._event("on_step_begin")
    for name, value in pending.items():
      setattr(self.control, name, getattr(self.control, name) or value)

  def _end_step(self, step, loss):
    self._step_active = False
    self.state.global_step = step
    self._loss_sum += float(loss)
    self._loss_count += 1
    self._event("on_step_end", loss=float(loss))

  def _resolve_save(self, receipt, manager):
    self.control.should_save = False
    if receipt.outcome == "no_save":
      return receipt
    previous = self._save_receipts.get(receipt.action_id)
    if previous is not None and previous.outcome == "completed":
      return previous
    listeners = any(type(callback).on_save is not TrainerCallback.on_save
                    for callback in self.callback_handler.callbacks)
    if not listeners:
      if receipt.reason is not None:
        self._save_receipts[receipt.action_id] = receipt
      return receipt
    receipt = receipt.wait(manager)
    self._save_receipts[receipt.action_id] = receipt
    if receipt.outcome == "completed":
      if self.state.global_step != receipt.completed_step:
        raise RuntimeError("A checkpoint completion event must match the current completed optimizer step.")
      self._event("on_save", action_id=receipt.action_id, completed_step=receipt.completed_step,
                  checkpoint_step=receipt.checkpoint_step, tier=receipt.tier)
    return receipt

  def _evaluation_metrics(self, metrics, metric_key_prefix):
    metrics = {metric_key_prefix + key[4:] if key.startswith("eval") else key: value
               for key, value in metrics.items()}
    self._event("on_evaluate", metrics=metrics, metric_key_prefix=metric_key_prefix)
    return metrics

  def _terminal_actions(self):
    if self.control.should_log:
      self.log({"train_loss": self._loss_sum / self._loss_count} if self._loss_count else {})
    if self.control.should_evaluate:
      self.evaluate()
    if self.control.should_save:
      self.save_state()
