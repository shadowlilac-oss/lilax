"""Experiment reporting through the Transformers callback lifecycle."""

import os
from types import SimpleNamespace

import trackio
import wandb
from transformers import TrainerCallback

from lilax._backend.common.vertex_tensorboard import VertexTensorboardManager
from lilax._backend.observability.tensorboard_logging import summary_writing


class TensorBoardCallback(TrainerCallback):
  def __init__(self, *, use_vertex_tensorboard=False, project="", region=""):
    self.use_vertex_tensorboard = use_vertex_tensorboard
    self.project, self.region = project, region
    self.writer = None
    self._vertex = None
    self._config = None

  def on_train_begin(self, args, state, control, **kwargs):
    if state.is_world_process_zero and self.writer is None:
      config = self._config
      directory = args.logging_dir or config.tensorboard_dir
      if self.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
        self._vertex = VertexTensorboardManager()
        self._vertex.configure_vertex_tensorboard(SimpleNamespace(
            use_vertex_tensorboard=self.use_vertex_tensorboard, vertex_tensorboard_project=self.project,
            vertex_tensorboard_region=self.region, tensorboard_dir=directory, run_name=config.run_name,
        ))
      self.writer = summary_writing.initialize_summary_writer(directory, config.run_name)

  def on_log(self, args, state, control, logs=None, **kwargs):
    if self.writer is None:
      self.on_train_begin(args, state, control)
    if self.writer is not None:
      for name, value in (logs or {}).items():
        if isinstance(value, (int, float)):
          self.writer.add_scalar(name, value, state.global_step)
      self.writer.flush()

  def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    self.on_log(args, state, control, logs=metrics)

  def on_train_end(self, args, state, control, **kwargs):
    self._close()

  def _close(self):
    summary_writing.close_summary_writer(self.writer)
    self.writer = None
    self._vertex = None


class WandbCallback(TrainerCallback):
  def __init__(self, *, project="maxtext", name=""):
    self.project, self.name = project, name
    self.run = None

  def on_train_begin(self, args, state, control, **kwargs):
    if state.is_world_process_zero and self.run is None:
      self.run = wandb.init(project=self.project, name=self.name or args.run_name, resume="allow")

  def on_log(self, args, state, control, logs=None, **kwargs):
    if self.run is None:
      self.on_train_begin(args, state, control)
    if self.run is not None:
      self.run.log(dict(logs or {}), step=state.global_step)

  def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    self.on_log(args, state, control, logs=metrics)

  def on_train_end(self, args, state, control, **kwargs):
    self._close()

  def _close(self):
    if self.run is not None:
      run, self.run = self.run, None
      run.finish()


class TrackioCallback(TrainerCallback):
  def __init__(self, *, project="maxtext", name=""):
    self.project, self.name = project, name
    self._active = False

  def on_train_begin(self, args, state, control, **kwargs):
    if state.is_world_process_zero and not self._active:
      trackio.init(project=self.project, name=self.name or args.run_name, resume="allow")
      self._active = True

  def on_log(self, args, state, control, logs=None, **kwargs):
    if not self._active:
      self.on_train_begin(args, state, control)
    if self._active:
      trackio.log(dict(logs or {}), step=state.global_step)

  def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    self.on_log(args, state, control, logs=metrics)

  def on_train_end(self, args, state, control, **kwargs):
    self._close()

  def _close(self):
    if self._active:
      self._active = False
      trackio.finish()
