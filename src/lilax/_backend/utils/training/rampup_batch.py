"""Batch size management during training rampup."""

import math

from lilax._backend.common import checkpointing


class RampupBatchManager:
  """Track the current batch size for a training step."""

  def __init__(self, config, step_num):
    self._verify_inputs(config)
    self._init_values(config)
    self.num_accum_samples = 0
    self._recover_states(step_num)

  def _verify_inputs(self, config):
    diff_batch_size = config.per_device_batch_size - config.per_device_batch_size_start
    if diff_batch_size <= 0:
      raise ValueError(
          "per_device_batch_size must be greater than per_device_batch_size_start. "
          f"get batch size is {config.per_device_batch_size} and "
          f"batch size start is {config.per_device_batch_size_start}."
      )
    if diff_batch_size % config.per_device_batch_size_increment:
      raise ValueError(
          "Expect rampup batch size change divisible by batch size increment."
          f"Got per_device_batch_size={config.per_device_batch_size} and "
          f"per_device_batch_size_start={config.per_device_batch_size_start}."
      )

  def _init_values(self, config):
    diff_batch_size = config.per_device_batch_size - config.per_device_batch_size_start
    num_increments = diff_batch_size // config.per_device_batch_size_increment
    self.samples_per_increment = config.global_rampup_samples / num_increments
    num_devices = int(config.num_target_devices)
    self.global_batch_size_end = int(num_devices * config.per_device_batch_size)
    self.global_batch_size_start = int(num_devices * config.per_device_batch_size_start)
    self.increment = int(num_devices * config.per_device_batch_size_increment)
    self.global_rampup_samples = config.global_rampup_samples
    self.global_batch_size_current = self.global_batch_size_start
    self.total_rampup_steps = self._compute_total_rampup_steps(config)
    self.total_used_samples = 0

  def _compute_total_rampup_steps(self, config):
    batch_size_start = config.per_device_batch_size_start
    batch_size_end = config.per_device_batch_size
    batch_size_increment = config.per_device_batch_size_increment
    diff_batch_size = batch_size_end - batch_size_start
    num_increments = diff_batch_size // batch_size_increment
    rampup_samples = config.global_rampup_samples / config.num_target_devices
    rampup_samples_per_increment = rampup_samples / num_increments
    total_rampup_steps = 0
    current_batch_size = batch_size_start
    while current_batch_size < batch_size_end:
      steps_for_this_stage = math.ceil(rampup_samples_per_increment / current_batch_size)
      total_rampup_steps += steps_for_this_stage
      current_batch_size += batch_size_increment
    return total_rampup_steps

  def _recover_states(self, step_num):
    if step_num < 0:
      return
    for _ in range(step_num):
      _ = self.update()
    return

  def update(self):
    """Update sample counters and report whether rampup remains active."""
    self.total_used_samples += self.global_batch_size_current
    self.num_accum_samples += self.global_batch_size_current
    is_time_to_increment = self.num_accum_samples >= self.samples_per_increment
    if is_time_to_increment:
      self.global_batch_size_current = min(self.increment + self.global_batch_size_current, self.global_batch_size_end)
      self.num_accum_samples = 0
    return self.global_batch_size_current < self.global_batch_size_end


def create_rampup_manager(config, checkpoint_manager):
  if not config.enable_rampup_batch_size:
    return None
  current_step = -1
  if checkpoint_manager and checkpointing.latest_step(checkpoint_manager):
    current_step = checkpointing.latest_step(checkpoint_manager)
  return RampupBatchManager(config, current_step)
