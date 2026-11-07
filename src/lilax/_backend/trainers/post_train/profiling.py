"""Owned XProf captures for Tunix post-training.

Captures stop at the next data fetch or outer exit and can include work after
the configured window. The lock coordinates only users of this class.
"""

import os
import threading
import uuid

import jax
from tunix.sft import profiler

from lilax._backend.observability import max_logging


class TrainingCapture:
  _lock = threading.Lock()

  def __init__(self, config, trainer, *, profile_dir=None, on_capture_complete=None):
    self.config = config
    self.on_capture_complete = on_capture_complete
    self.base_output_dir = config.tensorboard_dir if profile_dir is None else profile_dir
    self.first_step = trainer.iter_steps + config.skip_first_n_steps_for_profiler
    max_steps = trainer.config.max_steps
    max_iter = None if max_steps is None else max_steps * trainer.config.get_with_default("gradient_accumulation_steps", 1)
    self.last_step = self.first_step + config.profiler_steps
    if max_iter is not None:
      self.last_step = min(self.last_step, max_iter)
    self.enabled = bool(config.profiler) and jax.process_index() == 0 and self.first_step < self.last_step
    self.active = False
    self.reportable = True
    self.logdir = None

  def on_fetch(self, step, reporter):
    if not self.enabled:
      return
    if self.active and step >= self.last_step:
      self.finish(step, reporter, completed=True)
    elif step == self.first_step:
      if not self._lock.acquire(blocking=False):
        max_logging.log("Another cooperative post-training profiler is active; skipping capture.")
        self.enabled = False
        return
      try:
        options = profiler.ProfilerOptions(
            log_dir=self.base_output_dir, skip_first_n_steps=self.config.skip_first_n_steps_for_profiler,
            profiler_steps=self.config.profiler_steps,
        )
        profile_options = None
        pathways = jax.extend.backend.get_backend().platform_version.strip().startswith("Pathways")
        if not pathways or self.config.enable_continuous_profiling:
          profile_options = jax.profiler.ProfileOptions()
          if not pathways:
            profile_options.host_tracer_level = options.host_tracer_level
            profile_options.python_tracer_level = options.python_tracer_level
          if self.config.enable_continuous_profiling:
            profile_options.advanced_configuration = {"enable_continuous_profiling": True}
        self.logdir = os.path.join(self.base_output_dir, f"capture-{uuid.uuid4().hex}")
        jax.profiler.start_trace(self.logdir, profiler_options=profile_options)
        self.active = True
      except BaseException:
        self._lock.release()
        raise

  def finish(self, step, reporter, *, completed=False):
    if not self.active:
      return
    try:
      jax.profiler.stop_trace()
    except BaseException:
      self.reportable = False
      raise
    finally:
      self.active = False
      self.enabled = False
      self._lock.release()
    if completed and self.reportable:
      if self.on_capture_complete is not None:
        self.on_capture_complete(logdir=self.logdir, step=step, process_index=jax.process_index())
      elif reporter is not None:
        reporter.log_capture(logdir=self.logdir, step=step, process_index=jax.process_index())
