"""Backend dispatch for initializing the JAX distributed system."""

import os

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization

from lilax._backend.utils import elastic_utils as elastic
from lilax._backend.observability import max_logging
from lilax._backend.distributed.jax_distributed.backend import is_cpu_backend, is_gpu_backend
from lilax._backend.distributed.jax_distributed.cpu import initialize_jax_for_cpu
from lilax._backend.distributed.jax_distributed.gpu import initialize_jax_for_gpu
from lilax._backend.distributed.jax_distributed.jaxcoord import initialize_jaxcoord
from lilax._backend.distributed.jax_distributed.tpu import initialize_jax_for_tpu_with_emergency_checkpointing

initialize_multi_tier_checkpointing = initialization.initialize_multi_tier_checkpointing


def maybe_initialize_jax_distributed_system(raw_keys):
  """Initialize the JAX distributed system for the configured backend."""
  if is_cpu_backend(raw_keys):
    jax.config.update("jax_platforms", "cpu")
  if raw_keys["skip_jax_distributed_system"]:
    max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
    return
  if raw_keys["enable_single_controller"]:
    max_logging.log("Skipping jax distributed system since its not needed for single controller.")
    if raw_keys["enable_multi_tier_checkpointing"]:
      max_logging.log("Initializing multi-tier checkpointing for single controller...")
      mtc_init_kwargs = elastic.single_controller_mtc_init_kwargs(raw_keys)
      initialize_multi_tier_checkpointing(
          local_checkpoint_directory=raw_keys["local_checkpoint_directory"],
          backup_interval_minutes=raw_keys["multi_tier_checkpointing_backup_interval_minutes"],
          backup_interval_steps=raw_keys["multi_tier_checkpointing_backup_interval_steps"],
          run_name=raw_keys["run_name"],
          jax_initialization_timeout_seconds=raw_keys["jax_distributed_initialization_timeout"],
          use_colocated_python=True,
          **mtc_init_kwargs,
      )
    return
  if jax.distributed.is_initialized():
    max_logging.log("Jax distributed system is already initialized.")
    return
  if raw_keys["inference_benchmark_test"] or raw_keys["compile_topology"]:
    max_logging.log("Skipping jax distributed system initialization.")
    return
  if os.environ.get("LILAX_JAXCOORD") == "1" and not (
      raw_keys["hardware"] == "tpu" and (raw_keys["enable_multi_tier_checkpointing"] or raw_keys["enable_emergency_checkpoint"])
  ):
    initialize_jaxcoord(raw_keys)
    return
  if is_gpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for GPU backend...")
    initialize_jax_for_gpu(raw_keys)
    max_logging.log("Jax distributed system initialized on GPU!")
    return
  if is_cpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for CPU backend...")
    initialize_jax_for_cpu(raw_keys)
    max_logging.log("Jax distributed system initialized on CPUs!")
    return
  if raw_keys["hardware"] == "gpu_multiprocess":
    max_logging.log("Attempting to initialize the jax distributed system for gpu_multiprocess hardware...")
    if not raw_keys["enable_emergency_checkpoint"]:
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
    else:
      max_logging.log("Initializing jax distributed to support local checkpointing with GPUs...")
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
      ocp.multihost.initialize_runtime_to_distributed_ids()
      ocp.multihost.initialize_distributed_to_device_ids()
      max_logging.log("Jax distributed system initialized!")
    return
  max_logging.log("Attempting to initialize the jax distributed system for TPU backend...")
  if raw_keys["enable_multi_tier_checkpointing"]:
    initialize_multi_tier_checkpointing(
        local_checkpoint_directory=raw_keys["local_checkpoint_directory"],
        backup_interval_minutes=raw_keys["multi_tier_checkpointing_backup_interval_minutes"],
        backup_interval_steps=raw_keys["multi_tier_checkpointing_backup_interval_steps"],
        run_name=raw_keys["run_name"],
        jax_initialization_timeout_seconds=raw_keys["jax_distributed_initialization_timeout"],
        data_parallelism=raw_keys["mtc_data_parallelism"],
        num_slices=raw_keys["num_slices"],
    )
    max_logging.log("Jax distributed system initialized on TPUs for multi-tier checkpointing!")
  elif raw_keys["enable_checkpointing"] and raw_keys["compile_topology_num_slices"] == -1:
    if not raw_keys["enable_emergency_checkpoint"]:
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
    else:
      initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys)
    max_logging.log("Jax distributed system initialized on TPUs!")
