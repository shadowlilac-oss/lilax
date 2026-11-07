"""JAX distributed initialization for GPU backends."""

import os

import jax

from lilax._backend.observability import max_logging


def initialize_jax_for_gpu(raw_keys):
  """Jax distributed initialize for GPUs."""
  if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
    coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
    if os.environ.get("LILAX_JAXCOORD") == "1":
      visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
      devices = list(range(len(visible_devices.split(",")))) if visible_devices and visible_devices != "-1" else None
      jax.distributed.initialize(
          coordinator_address=f"{coordinator_ip}:{coordinator_port}",
          num_processes=int(os.environ["NNODES"]),
          process_id=int(os.environ["NODE_RANK"]),
          initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
          local_device_ids=devices,
          cluster_detection_method="deactivate",
      )
      max_logging.log(f"JAX global devices: {jax.devices()}")
      return
    env_var_list = ["CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS"]
    for env_var in env_var_list:
      devices = os.getenv(env_var)
      if devices is not None:
        max_logging.log(f"Using {env_var} to initialize JAX distributed system: {devices}")
        break
    if devices is None:
      jax.config.update("jax_cuda_visible_devices", "all")
      jax.config.update("jax_rocm_visible_devices", "all")
    if devices is not None:
      try:
        devices = [int(x) for x in devices.split(",")]
      except (ValueError, TypeError) as e:
        max_logging.log(f"Error parsing {env_var}: {e}")
        devices = None
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{coordinator_port}",
        num_processes=int(os.getenv("NNODES")),
        process_id=int(os.getenv("NODE_RANK")),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
        local_device_ids=devices,
    )
    max_logging.log(f"JAX global devices: {jax.devices()}")
