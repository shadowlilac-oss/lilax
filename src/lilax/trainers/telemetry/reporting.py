"""Experiment reporting for native training metrics."""

import jax
import trackio
import wandb

from lilax._backend.observability.tensorboard_logging import summary_writing
from ._profile_artifacts import log_profile_artifacts


class MetricReporter:
  def __init__(self, config, report_to, *, tensorboard=False):
    self.trackio = False
    self.wandb_run = None
    self.writer = None
    if jax.process_index() != 0:
      return
    try:
      if tensorboard and "tensorboard" in report_to:
        self.writer = summary_writing.initialize_summary_writer(config.tensorboard_dir, config.run_name, True)
      if "trackio" in report_to:
        trackio.init(project=config.wandb_project_name, name=config.run_name, resume="allow")
        self.trackio = True
      if "wandb" in report_to:
        self.wandb_run = wandb.init(project=config.wandb_project_name, name=config.wandb_run_name, resume="allow")
    except BaseException:
      self.close()
      raise

  def log(self, metrics, step, metric_type):
    del metric_type
    values = {name: float(value) for name, value in metrics.get("scalar", {}).items()}
    for name, entries in metrics.get("scalars", {}).items():
      values.update({f"{name}/{key}": float(value) for key, value in entries.items()})
    if self.trackio:
      trackio.log(values, step=step)
    if self.wandb_run is not None:
      self.wandb_run.log(values, step=step)
    if self.writer is not None:
      for name, value in values.items():
        self.writer.add_scalar(name, value, step)
      self.writer.flush()

  def log_capture(self, *, logdir, step, process_index):
    log_profile_artifacts(logdir, step, process_index, self.trackio, self.wandb_run)

  def close(self):
    try:
      if self.trackio:
        self.trackio = False
        trackio.finish()
    finally:
      try:
        if self.wandb_run is not None:
          run, self.wandb_run = self.wandb_run, None
          run.finish()
      finally:
        if self.writer is not None:
          writer, self.writer = self.writer, None
          summary_writing.close_summary_writer(writer)
