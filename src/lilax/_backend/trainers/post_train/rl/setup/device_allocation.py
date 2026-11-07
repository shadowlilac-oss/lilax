"""Split devices and configurations between RL trainers and samplers."""

import collections

import jax

from lilax._backend.observability import max_logging


def setup_configs_and_devices(config, *, resolve_config, devices=None):
  """Set up device allocation and configs for training and inference."""
  return setup_configs_and_devices_from_config(config, resolve_config=resolve_config, devices=devices)


def setup_configs_and_devices_from_config(config, *, resolve_config, devices=None):
  """Set up device allocation and configs using independently resolved overrides."""
  if devices is None:
    devices = jax.devices()
  if config.num_trainer_slices == -1 and config.num_samplers_slices == -1:
    max_logging.log("Running on a single slice")
    num_vms = len(devices) // config.chips_per_vm
    trainer_devices = devices
    sampler_devices = devices
    if num_vms >= 2 and config.use_pathways:
      max_logging.log(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
      num_devices = len(devices)
      num_trainer_devices = int(num_devices * config.trainer_devices_fraction)
      num_sampler_devices = int(num_devices * config.sampler_devices_fraction)
      trainer_devices = devices[:num_trainer_devices]
      sampler_devices = devices[num_devices - num_sampler_devices :]
      if config.trainer_devices_fraction != 1.0:
        max_logging.log(f"Using first {len(trainer_devices)} devices as Trainer devices")
      if config.sampler_devices_fraction != 1.0:
        max_logging.log(f"Using last {len(sampler_devices)} devices as Sampler devices")
    trainer_config = config
    sampler_config = config
  elif config.num_trainer_slices > 0 and config.num_samplers_slices > 0:
    max_logging.log("Running with Multislice")
    devices_by_slice = collections.defaultdict(list)
    for device in devices:
      devices_by_slice[device.slice_index].append(device)
    slice_indices = sorted(devices_by_slice.keys())
    if len(slice_indices) < config.num_trainer_slices + config.num_samplers_slices:
      raise ValueError("Not enough slices for trainer and samplers")

    trainer_devices = []
    for index in range(config.num_trainer_slices):
      trainer_devices.extend(devices_by_slice[slice_indices[index]])
    sampler_devices = []
    for index in range(config.num_trainer_slices, config.num_trainer_slices + config.num_samplers_slices):
      sampler_devices.extend(devices_by_slice[slice_indices[index]])

    trainer_devices_per_slice = len(trainer_devices) // config.num_trainer_slices
    trainer_fsdp = trainer_devices_per_slice
    tp = config.ici_tensor_parallelism
    if tp > 1:
      if trainer_devices_per_slice % tp != 0:
        raise ValueError(
            f"trainer_devices_per_slice ({trainer_devices_per_slice}) must be divisible by tensor parallelism ({tp})"
        )
      if config.ici_fsdp_parallelism != -1 and config.ici_fsdp_parallelism * tp != trainer_devices_per_slice:
        raise ValueError(
            f"ici_fsdp_parallelism ({config.ici_fsdp_parallelism}) * ici_tensor_parallelism ({tp}) must equal "
            f"devices_per_slice ({trainer_devices_per_slice})"
        )
      trainer_fsdp = trainer_devices_per_slice // tp

    trainer_kwargs = {
        "num_slices": config.num_trainer_slices,
        "ici_fsdp_parallelism": trainer_fsdp,
        "ici_tensor_parallelism": tp,
        "dcn_data_parallelism": config.num_trainer_slices,
    }
    sampler_kwargs = {
        "num_slices": config.num_samplers_slices,
        "ici_fsdp_parallelism": len(sampler_devices) // config.num_samplers_slices,
        "ici_tensor_parallelism": -1,
        "dcn_data_parallelism": config.num_samplers_slices,
    }
    trainer_config = resolve_config(trainer_kwargs)
    sampler_config = resolve_config(sampler_kwargs)
  else:
    raise ValueError("num_trainer_slices and num_samplers_slices should be both -1 or positive")

  return trainer_config, sampler_config, trainer_devices, sampler_devices
