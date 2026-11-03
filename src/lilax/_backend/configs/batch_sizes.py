"""Derive training, evaluation, and ramp-up batch sizes."""

import math


def derive_batch_sizes(config):
  """Set batch sizes after the target device count has been resolved."""
  def calculate_global_batch_sizes(per_device_batch_size, expansion_factor, num_devices, grad_accum_steps):
    """Helper to calculate global and micro batch sizes for training and loading."""
    if per_device_batch_size < 1.0:
      micro_batch_to_load = num_devices * (expansion_factor if expansion_factor > 0 else 1)
    else:
      micro_batch_to_load = int(num_devices * per_device_batch_size * (expansion_factor if expansion_factor > 0 else 1))
    micro_batch_to_train = int(num_devices * per_device_batch_size)
    global_batch_to_load = int(micro_batch_to_load * grad_accum_steps)
    global_batch_to_train = int(micro_batch_to_train * grad_accum_steps)
    return global_batch_to_load, global_batch_to_train, micro_batch_to_train

  (
      config.global_batch_size_to_load,
      config.global_batch_size_to_train_on,
      config.micro_batch_size_to_train_on,
  ) = calculate_global_batch_sizes(
      config.per_device_batch_size,
      config.expansion_factor_real_data,
      config.num_target_devices,
      config.gradient_accumulation_steps,
  )
  (
      config.global_batch_size_to_load_eval,
      config.global_batch_size_to_eval_on,
      config.micro_batch_size_to_eval_on,
  ) = calculate_global_batch_sizes(
      config.eval_per_device_batch_size,
      config.expansion_factor_real_data,
      config.num_target_devices,
      1,
  )

  if config.enable_rampup_batch_size:
    (
        config.global_batch_size_to_load_start,
        _,
        _,
    ) = calculate_global_batch_sizes(
        config.per_device_batch_size_start,
        config.expansion_factor_real_data,
        config.num_target_devices,
        config.gradient_accumulation_steps,
    )
    (
        config.global_batch_size_to_load_increment,
        _,
        _,
    ) = calculate_global_batch_sizes(
        config.per_device_batch_size_increment,
        config.expansion_factor_real_data,
        config.num_target_devices,
        config.gradient_accumulation_steps,
    )
    diff_batch_size = config.global_batch_size_to_load - config.global_batch_size_to_load_start
    if config.global_batch_size_to_load_increment > 0:
      num_increments = diff_batch_size // config.global_batch_size_to_load_increment
      if num_increments > 0:
        rampup_samples_per_increment = config.global_rampup_samples / num_increments
        config.rampup_samples_per_increment_to_load = rampup_samples_per_increment
        total_rampup_steps = 0
        current_batch_size = config.global_batch_size_to_load_start
        for _ in range(int(num_increments)):
          steps_for_this_stage = (
              math.ceil(rampup_samples_per_increment / current_batch_size) if current_batch_size > 0 else 0
          )
          total_rampup_steps += steps_for_this_stage
          current_batch_size += config.global_batch_size_to_load_increment
        config.rampup_end_step = total_rampup_steps
      else:
        config.rampup_end_step = 0
    else:
      config.rampup_end_step = 0
  else:
    config.rampup_end_step = 0
