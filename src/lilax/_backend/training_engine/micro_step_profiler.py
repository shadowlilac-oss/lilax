"""Micro-step driven profiling for the training engine."""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

from absl import logging
import jax
from lilax._backend.configs import pyconfig
from lilax._backend.configs import types


def profiled_step(name: str, control_profile: bool = False) -> Callable[..., Any]:
  """Wraps an engine step in a trace annotation, opening and closing the engine's profile around it when asked."""

  def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
      step_to_profile = self.total_micro_steps
      current_micro_step_count = self.micro_step_count
      if control_profile:
        self._profiler.maybe_activate(step_to_profile, current_micro_step_count, self.train_step, blocking_object=self._state)
      with jax.profiler.StepTraceAnnotation(name, step_num=self.train_step):
        result = method(self, *args, **kwargs)
      if control_profile:
        self._profiler.maybe_deactivate(step_to_profile, current_micro_step_count, self.train_step, blocking_object=self._state)
      return result

    return wrapper

  return decorator


class MicroStepProfiler:
  """Opens and closes profiles around ranges of engine micro steps."""

  def __init__(self, config: pyconfig.HyperParameters) -> None:
    self.do_not_profile = False
    if (
        config.profiler != types.ProfilerType.XPLANE
        or config.profiler_steps == 0
        or (not config.upload_all_profiler_results and jax.process_index() != 0)
    ):
      self.do_not_profile = True
      return

    self.output_path = config.tensorboard_dir
    self.first_profile_step = config.skip_first_n_steps_for_profiler
    self.last_profile_step = self.first_profile_step + config.profiler_steps - 1
    if self.first_profile_step > self.last_profile_step:
      self.do_not_profile = True
      logging.warning(
          "Skipping profiling: the first profile step %d exceeds the last profile step %d.",
          self.first_profile_step,
          self.last_profile_step,
      )
      return
    self.profiling_options = jax.profiler.ProfileOptions()
    advanced_config: dict[str, Any] = {}
    if config.profile_power_events:
      advanced_config.update(
          {
              "tpu_power_trace_level": int(config.xprof_tpu_power_trace_level),
              "e2e_enable_fw_throttle_event": config.xprof_e2e_enable_fw_throttle_event,
              "e2e_enable_fw_power_level_event": config.xprof_e2e_enable_fw_power_level_event,
              "e2e_enable_fw_thermal_event": config.xprof_e2e_enable_fw_thermal_event,
          }
      )
    if config.enable_tpu_profiling_options:
      advanced_config.update(
          {
              "tpu_num_chips_to_profile_per_task": config.tpu_num_chips_to_profile_per_task,
              "tpu_num_sparse_core_tiles_to_trace": config.tpu_num_sparse_core_tiles_to_trace,
              "tpu_num_sparse_cores_to_trace": config.tpu_num_sparse_cores_to_trace,
          }
      )
    if config.enable_continuous_profiling:
      advanced_config["enable_continuous_profiling"] = True
    if advanced_config:
      self.profiling_options.advanced_configuration = advanced_config
    self.profile_active: bool = False
    self.profiler_period = config.profile_periodically_period
    if 0 < self.profiler_period < config.profiler_steps:
      self.do_not_profile = True
      logging.warning(
          "profile_periodically_period=%d is shorter than profiler_steps=%d. The period is the gap between window "
          "starts, so it must be at least the window length or the windows overlap.",
          self.profiler_period,
          config.profiler_steps,
      )

  def maybe_activate(
      self, total_micro_steps: int, current_micro_step: int, train_step: int, blocking_object: Any = None
  ) -> None:
    if self.do_not_profile or self.profile_active or not self.should_activate_profile(total_micro_steps):
      return
    self.profile_active = True
    logging.info("Started profiling at micro step %d and train step %d.", current_micro_step, train_step)
    if blocking_object is not None:
      jax.block_until_ready(blocking_object)
    jax.profiler.start_trace(self.output_path, profiler_options=self.profiling_options)

  def maybe_deactivate(
      self, total_micro_steps: int, current_micro_step: int, train_step: int, blocking_object: Any = None
  ) -> None:
    if self.do_not_profile or not self.profile_active or not self.should_deactivate_profile(total_micro_steps):
      return
    if blocking_object is not None:
      jax.block_until_ready(blocking_object)
    jax.profiler.stop_trace()
    logging.info("Stopped profiling at micro step %d and train step %d.", current_micro_step, train_step)
    self.profile_active = False

  def should_activate_profile(self, step):
    if self.profiler_period > 0:
      return (step - self.first_profile_step) % self.profiler_period == 0
    return step == self.first_profile_step

  def should_deactivate_profile(self, step):
    if self.profiler_period > 0:
      return (step - self.last_profile_step) % self.profiler_period == 0
    return step == self.last_profile_step

  def close(self, blocking_object: Any = None) -> None:
    if self.do_not_profile or not self.profile_active:
      return
    logging.warning("Profiling was still active when engine close was called, stopping it now.")
    if blocking_object is not None:
      jax.block_until_ready(blocking_object)
    jax.profiler.stop_trace()
    self.profile_active = False
