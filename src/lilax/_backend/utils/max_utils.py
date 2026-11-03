"""Common Max Utils needed by multiple modules.
All the functions include MaxText modules, such as Pyconfig, should be moved to MaxText utils file.
"""

import collections
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
import functools
from functools import partial
import os
import re
import socket
import subprocess
import time
from typing import Any

from etils import epath
import flax
from flax.core.spmd import get_logical_axis_rules
import jax
from jax.experimental import mesh_utils
from jax.experimental.xla_metadata import set_xla_metadata
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization
from packaging.version import Version
import psutil
from tensorboardX import writer

from lilax._backend.common.cloud import is_decoupled
from lilax._backend.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN, ReorderStrategy
from lilax._backend.utils import elastic_utils
from lilax._backend.utils import max_logging
from lilax._backend.utils.diagnostics.pytree import (
    _bytes_of,
    calculate_bytes_from_pytree,
    calculate_num_params_from_pytree,
    calculate_total_params_per_chip,
    print_compiled_memory_stats,
    print_cpu_ram_stats,
    print_mem_stats,
    print_model_vars,
    print_pytree_shape,
    summarize_pytree_data,
    summarize_size_from_pytree,
)

HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"

initialize_multi_tier_checkpointing = initialization.initialize_multi_tier_checkpointing
HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"


def permute_to_match_maxtext_rope(arr):
  """Permutes the Huggingface Rope to match the MaxText logic."""
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  evens, odds = np.split(arr, 2, axis=arr.ndim - 1)  # pylint: disable=W0632
  x = np.empty_like(arr)
  x[..., ::2] = evens
  x[..., 1::2] = odds
  return x


def get_project():
  if is_decoupled():
    return os.environ.get("LOCAL_GCLOUD_PROJECT", "local-maxtext-project")
  try:
    completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
    project_outputs = completed_command.stdout.decode().strip().split("\n")
    if len(project_outputs) < 1 or project_outputs[-1] == "":
      max_logging.log("You must specify config.vertex_tensorboard_project or set 'gcloud config set project <project>'")
      return None
    return project_outputs[-1]
  except (FileNotFoundError, subprocess.CalledProcessError) as ex:
    max_logging.log(f"Unable to retrieve gcloud project (decoupled={is_decoupled()}): {ex}")
    return None


def parse_libtpu_flags_to_dict(flags_str: str) -> dict:
  """Parses a string of XLA flags into a dictionary of compilation options.
  This function is only for compilation usage.
  """
  if not flags_str or not flags_str.strip():
    return {}
  cleaned_str = flags_str.replace("\\", " ")
  tokens = cleaned_str.split()
  options_dict = {}
  token_pattern = re.compile(r"^--([a-zA-Z0-9_]+)=(.+)$")
  for token in tokens:
    match = token_pattern.match(token)
    if not match:
      raise ValueError(f"Invalid flag format detected: '{token}'. Expected format: '--key=value'")
    key, value = match.groups()
    if key in options_dict:
      raise ValueError(f"Duplicate flag detected: '--{key}'")
    options_dict[key] = value
  return options_dict


def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind=memory_kind), t)


def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(lambda t: t.astype(dst) if t.dtype == src else t, nest)


def find_nans_and_infs(pytree):
  def finder(x):
    return jnp.any(jnp.isinf(x) | jnp.isnan(x))

  bad_pytree = jax.tree_util.tree_map(finder, pytree)
  return jax.tree_util.tree_flatten(bad_pytree)


def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), x, initializer=0.0))


def device_space():
  """Version guard for jax.memory.Space.Device."""
  if Version(jax.__version__) >= Version("0.7.1"):
    return jax.memory.Space.Device
  else:
    return jax._src.sharding_impls.TransferToMemoryKind("device")


def initialize_summary_writer(tensorboard_dir, run_name, enable_tensorboard=True):
  """Return a tensorboardX SummaryWriter, or None when disabled."""
  if jax.process_index() != 0:
    return None
  if not enable_tensorboard:
    max_logging.log("TensorBoard disabled.")
    return None
  if is_decoupled():
    return None
  if not tensorboard_dir or not run_name:
    max_logging.log("tensorboard_dir or run_name missing; using default SummaryWriter directory.")
    return writer.SummaryWriter()
  summary_writer_path = os.path.join(tensorboard_dir, run_name)
  return writer.SummaryWriter(summary_writer_path)


def close_summary_writer(summary_writer):
  if jax.process_index() == 0 and summary_writer is not None:
    summary_writer.close()


def add_text_to_summary_writer(key, value, summary_writer):
  """Writes given key-value pair to tensorboard as text/summary."""
  if jax.process_index() == 0 and summary_writer is not None:
    summary_writer.add_text(key, value)


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
      mtc_init_kwargs = elastic_utils.single_controller_mtc_init_kwargs(raw_keys)
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


def initialize_jax_for_cpu(raw_keys):
  """Jax distributed initialize for CPUs. Includes retries until the coordinator is ready."""
  coordinator_ip_address = get_coordinator_ip_address()
  coordinator_address = coordinator_ip_address + ":1234"
  job_index = int(os.environ.get("JOB_INDEX"))
  job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
  processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
  pid = job_index * processes_in_job + job_completion_index
  max_logging.log(f" Jax process id is {pid} ")
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      process_id=pid,
      num_processes=int(os.environ.get("JAX_PROCESS_COUNT")),
      initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
  )


def initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys):
  """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
  The information required to initialize JAX distributed runtime will be written by GKE to
  the local checkpoint directory. This function retrieves that information and initializes
  JAX distributed runtime.
  """
  process_id, coordinator_address = _retrieve_jax_init_info(raw_keys)
  if process_id != "" and coordinator_address != "":
    max_logging.log(
        f"Using {process_id} as the process_id and {coordinator_address} as the"
        " coordinator_address to initialize JAX distributed runtime..."
    )
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=int(process_id),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )
    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()


def _retrieve_jax_init_info(raw_keys):
  """Retrieve JAX init info from a local file."""
  JAX_INIT_INFO_FILE = "jax-init-info.txt"
  local_jax_init_info_file = epath.Path(raw_keys["local_checkpoint_directory"]) / JAX_INIT_INFO_FILE
  for i in range(900):
    if local_jax_init_info_file.exists():
      return local_jax_init_info_file.read_text().split("\n")[:2]
    max_logging.log(f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, sleeping for 1 second before retrying...")
    time.sleep(1)
  max_logging.log(
      f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds," "returning empty process id and coordinator address."
  )
  return "", ""


def get_num_slices(raw_keys, config=None):
  """Calculate num_slices based on number of devices."""
  if raw_keys.get("num_slices", -1) != -1:
    max_logging.log(f"Using num_slices={raw_keys['num_slices']} per user request.")
    return raw_keys["num_slices"]
  if getattr(raw_keys, "hardware", None) == "cpu":
    max_logging.log(" Setting num_slices=1 for CPU hardware type")
    return 1
  if int(raw_keys["compile_topology_num_slices"]) > 0:
    return raw_keys["compile_topology_num_slices"]
  else:
    try:
      return len(elastic_utils.live_slice_indices(config))
    except (ValueError, AttributeError):
      return 1


def is_cpu_backend(raw_keys):
  """Determine whether Maxtext is intended to run on a CPU backend."""
  return raw_keys.get("hardware") == "cpu"


def is_gpu_backend(raw_keys):
  """Determine whether Maxtext is intended to run on a GPU backend."""
  return raw_keys.get("hardware") == "gpu"


def model_loop_unroll_context(config):
  """Returns a context that marks the model's loops traced inside it for double buffering.

  `_xla_loop_unroll_strategy` is only read by XLA:GPU (https://github.com/openxla/xla/pull/43468);
  enable it with JAX_OPTIMIZATION_LEVEL=O1 and
  XLA_FLAGS="--xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_MANUAL_UNROLL". The other
  unrolling modes ignore it. Enter it around the model's forward and backward pass only, so that it
  scopes the model's scans and not an enclosing loop such as the gradient accumulation scan.
  On TPU/CPU it is a no-op: `set_xla_metadata` pre-populates frontend attributes on every traced op,
  causing XLA's CallInliner to drop `_xla_compute_type` from `compute_on("device")` calls and
  XLA:TPU to auto-offload those collectives to SparseCore.
  """
  if getattr(config, "hardware", "tpu") in ("gpu", "gpu_multiprocess"):
    return set_xla_metadata(_xla_loop_unroll_strategy="double-buffer")
  return nullcontext()


def get_coordinator_ip_address():
  """Get coordinator IP Address with retries"""
  coordinator_address = ""
  coordinator_ip_address = ""
  if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        coordinator_ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        max_logging.log(
            f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying..."
        )
        lookup_attempt += 1
        time.sleep(5)
  max_logging.log(f"Coordinator IP address: {coordinator_ip_address}")
  return coordinator_ip_address


def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert (
        parallelism_vals.count(-1) == 1
    ), f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert (
        determined_val >= 1 and determined_val.is_integer()
    ), f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"
  assert np.prod(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match"
      f" the product of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
  )

  return parallelism_vals


def reshape_mesh_to_rings(a, strategy):
  """Reshape device mesh to rings for 64x4 or 32x8 mesh shape"""
  b = []
  if strategy == HYBRID_RING_64X4:
    for i in range(8):
      b.append([])
      for j in range(8):
        a_i = i * 2
        a_j = j * 2
        b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    b = np.array(b)
    b = np.reshape(b, (64, 4))
  elif strategy == HYBRID_RING_32X8:
    for i in range(8):
      b.append([])
      for j in range(4):
        a_i = i * 2
        a_j = j * 4
        b[i].append(
            [
                a[a_i, a_j],
                a[a_i, a_j + 1],
                a[a_i, a_j + 2],
                a[a_i, a_j + 3],
                a[a_i + 1, a_j + 3],
                a[a_i + 1, a_j + 2],
                a[a_i + 1, a_j + 1],
                a[a_i + 1, a_j],
            ]
        )
    b = np.array(b)
    b = np.reshape(b, (32, 8))
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is not implemented.")
  return b


def create_custom_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any],
    custom_strategy: str,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
) -> np.ndarray:
  """Custom device mesh for 64x4 ici parallelism"""
  assert len(devices) % 256 == 0, f"This custom mesh is not valid for {len(devices)} devices"
  attr = "process_index" if process_is_granule else "slice_index"
  if not hasattr(devices[0], attr):
    raise ValueError(f"Device {devices[0]} does not have attribute {attr}. See" " `process_is_granule` option.")
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())] if should_sort_granules_by_key else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(f"Number of slices {len(granules)} must equal the product of " f"dcn_mesh_shape {dcn_mesh_shape}")
  per_granule_meshes = [
      mesh_utils.create_device_mesh(
          [16, 16],
          granule,
          allow_split_physical_axes=False,
      )
      for granule in granules
  ]

  per_granule_meshes = [np.reshape(reshape_mesh_to_rings(x, custom_strategy), mesh_shape) for x in per_granule_meshes]
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh


def is_valid_custom_mesh(ici_parallelism, strategy):
  """Checks if the given strategy and ICI parallelism are valid."""
  if not strategy:
    return False

  valid_strategies = {
      HYBRID_RING_64X4: [1, 4, 64],
      HYBRID_RING_32X8: [1, 8, 32],
  }

  if strategy in valid_strategies:
    if sorted(set(ici_parallelism)) == valid_strategies[strategy]:
      return True
    else:
      raise ValueError(f"Invalid custom_mesh:{strategy} chosen for ICI mesh shape {ici_parallelism}")
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is invalid.")


def optimize_mesh_for_tpu_v6e(mesh, devices):
  """Apply transformations to the mesh to optimize for TPU v6e"""
  if devices[0].device_kind != "TPU v6 lite":
    return mesh
  num_devices = len(devices)
  mesh_is_1d_ring = num_devices in mesh.shape
  if not mesh_is_1d_ring:
    return mesh
  device_coords = [d.coords for d in devices]
  coord_size = len(device_coords[0])
  max_coords = tuple(max(dc[i] for dc in device_coords) for i in range(coord_size))
  min_coords = tuple(min(dc[i] for dc in device_coords) for i in range(coord_size))
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))
  if dims != (2, 4, 1):
    return mesh
  axis_idx = mesh.shape.index(num_devices)
  new_mesh = np.moveaxis(mesh, axis_idx, 0)
  new_mesh[4:] = new_mesh[-1:3:-1]
  new_mesh = np.moveaxis(new_mesh, 0, axis_idx)
  max_logging.log("Optimized the mesh for TPU v6e")
  return new_mesh


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def tie_cotangents(*args):
  """Return inputs unchanged while releasing their cotangents together."""
  @jax.custom_vjp
  def tied(*a):
    return a

  def tied_fwd(*a):
    return a, None

  def tied_bwd(_, cts):
    return jax.lax.optimization_barrier(cts)

  tied.defvjp(tied_fwd, tied_bwd)
  outputs = tied(*args)
  return outputs[0] if len(args) == 1 else outputs


@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.
  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.
  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (targets, z_loss, exp_shifted, sum_exp, log_z)


def _cross_entropy_with_logits_bwd(
    res: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    g: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, None, None]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]
  targets, z_loss, exp_shifted, sum_exp, log_z = res
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  return (jnp.asarray(g_logits, exp_shifted.dtype), None, None)


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_util.tree_map(delete_leaf, p)


def print_system_information():
  """Print system information of the current environment.
  Note that this will initialize the JAX backend."""
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.extend.backend.get_backend().platform_version}")


def unpermute_from_match_maxtext_rope(arr, model_size):
  """
  Function to get the RoPE values in correct ordering
  """
  if model_size[:8] != "llama3.1":
    return arr
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


@partial(jax.jit, static_argnames=("cp_size", "seq_dim", "to_contiguous"))
def reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False):
  """Reorder a sequence for causal load balancing or restore contiguous order."""
  if tensor is None:
    return tensor
  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)
  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")
  seq_dim = seq_dim % tensor.ndim
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(*ori_tensor_shape[:seq_dim], 2 * cp_size, group_size, *ori_tensor_shape[seq_dim + 1 :])
  swapped = jnp.swapaxes(reshaped, 0, seq_dim)
  if not to_contiguous:
    first_half, second_half = jnp.split(swapped, 2, axis=0)
    second_half_reversed = second_half[::-1, ...]
    stacked = jnp.stack([first_half, second_half_reversed], axis=1)
    permuted = stacked.reshape(2 * cp_size, *swapped.shape[1:])
  else:
    reshaped_swapped = swapped.reshape(cp_size, 2, *swapped.shape[1:])
    first_half = reshaped_swapped[:, 0, ...]
    second_half_reversed = reshaped_swapped[:, 1, ...]
    second_half = second_half_reversed[::-1, ...]
    permuted = jnp.concatenate([first_half, second_half], axis=0)
  reordered = jnp.swapaxes(permuted, 0, seq_dim)
  return reordered.reshape(ori_tensor_shape)


@partial(jax.jit, static_argnums=(1, 2, 3))
def reorder_causal_load_balanced(batch, cp_size, reorder_strategy, hardware="tpu"):
  """Reorder batch sequences using the appropriate hardware backend."""
  reorder_keys = {
      "inputs", "targets", "inputs_position", "targets_position", "inputs_segmentation",
      "targets_segmentation", "corruption_mask", "targets_loss_mask",
  }
  if hardware in ("gpu", "gpu_multiprocess"):
    from transformer_engine.jax.attention import ReorderStrategy as TE_ReorderStrategy
    from transformer_engine.jax.attention import reorder_causal_load_balancing

    reorder_strategy_map = {
        ReorderStrategy.DUAL_CHUNK_SWAP: TE_ReorderStrategy.DualChunkSwap,
        ReorderStrategy.STRIPED: TE_ReorderStrategy.Striped,
    }
    return {
        key: reorder_causal_load_balancing(value, reorder_strategy_map[reorder_strategy], cp_size=cp_size, seq_dim=1)
        if key in reorder_keys else value
        for key, value in batch.items()
    }
  if reorder_strategy == ReorderStrategy.STRIPED:
    raise ValueError(
        f"STRIPED reorder strategy requires Transformer Engine and is only supported on GPU, got hardware={hardware!r}."
    )
  return {key: reorder_sequence(value, cp_size=cp_size) if key in reorder_keys else value for key, value in batch.items()}


def reorder_mask_load_balancing(tensor, cp_size: int, seq_dim: int):
  """Reorder a NumPy mask for causal load balancing."""
  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)
  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(*ori_tensor_shape[:seq_dim], 2 * cp_size, group_size, *ori_tensor_shape[seq_dim + 1 :])
  first_half = np.arange(cp_size)
  second_half = np.arange(2 * cp_size - 1, cp_size - 1, -1)
  src_indices = np.stack([first_half, second_half], axis=1).reshape(-1)
  reordered = np.take(reshaped, src_indices, axis=seq_dim)
  return reordered.reshape(ori_tensor_shape)


def reordered_cp_size(config, mesh) -> int:
  """The cp_size the input pipeline reordered the batch with; 1 if it did not."""
  if mesh is None or not getattr(config, "context_parallel_load_balance", False):
    return 1
  return mesh.shape.get(config.context_sharding, 1)


@staticmethod
def get_batch_seq_len_for_mode(config, model_mode):
  """Resolves the batch size and sequence length based on the model's operational mode."""
  if model_mode == MODEL_MODE_PREFILL:
    batch_size = 1
    seq_len = config.max_prefill_predict_length
  elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = 1
  elif model_mode == MODEL_MODE_TRAIN:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = config.max_target_length
  else:
    raise ValueError(f"Unknown model_mode: {model_mode}")
  return batch_size, seq_len


def print_non_trivial_mesh_axis(mesh):
  """Print mesh axis if its axis size is larger than one."""
  for mesh_axis, axis_size in mesh.shape.items():
    if axis_size > 1:
      print(f"{mesh_axis}: {axis_size}", flush=True)


def bootstrap_transformer_engine_cgemm(config):
  """Initialize communicators for Transformer Engine collective GEMM."""
  import transformer_engine.jax.cpp_extensions as tex

  tsp_size = config.ici_tensor_sequence_parallelism * config.dcn_tensor_sequence_parallelism
  tex.collective_gemm_bootstrap(jax.device_count(), jax.local_device_count(), jax.process_index(), tsp_size)


@contextmanager
def maybe_get_transformer_engine_context(config):
  """Enter the Transformer Engine mesh context on GPUs."""
  if config.hardware in ["gpu", "gpu_multiprocess"]:
    with transformer_engine_context():
      yield
  else:
    with nullcontext():
      yield


@contextmanager
def transformer_engine_context():
  """Provide MaxText mesh resources to Transformer Engine."""
  from transformer_engine.jax.sharding import global_shard_guard, MeshResource

  mesh_resource = MeshResource(
      dp_resource="data",
      tp_resource="tensor",
      tpsp_resource="tensor_sequence",
      fsdp_resource="fsdp",
      pp_resource=None,
      cp_resource="context",
      ep_resource="expert",
  )
  with global_shard_guard(mesh_resource):
    yield


_te_moe_bootstrap_signature = None


def get_te_moe_recv_capacity_per_rank():
  """Return the receive capacity used by the process-local TE bootstrap."""
  if _te_moe_bootstrap_signature is None:
    raise RuntimeError("TE MoE EP has not been bootstrapped yet.")
  return _te_moe_bootstrap_signature[5]


def maybe_bootstrap_te_moe(config, mesh, shaped_batch):
  """Eagerly initialize Transformer Engine NCCL EP for fused MoE."""
  if not getattr(config, "te_moe_block", False):
    return
  if jax.local_device_count() != 1:
    raise ValueError(
        "te_moe_block=True requires one local device per process. Run MaxText with "
        "`test-maxtext.sh --multiprocess` or an equivalent one-GPU-per-process launcher."
    )
  try:
    from transformer_engine.jax.ep import ep_bootstrap
    from transformer_engine.jax.moe import get_moe_recv_capacity_per_rank, record_ep_bootstrap_signature_for_moe
  except ImportError as exc:
    raise ImportError("te_moe_block=True requires TransformerEngine with JAX EP MoE support.") from exc

  ep_axis = "expert"
  fsdp_axis = "fsdp"
  ep_size = mesh.shape.get(ep_axis, 1)
  fsdp_size = mesh.shape.get(fsdp_axis, 1)
  loaded_batch_size, sequence_length = shaped_batch["inputs"].shape[:2]
  ga_steps = config.gradient_accumulation_steps
  if loaded_batch_size % ga_steps != 0:
    raise ValueError(f"TE MoE EP loaded batch size ({loaded_batch_size}) must be divisible by GA steps ({ga_steps}).")
  batch_size = max(loaded_batch_size // ga_steps, config.micro_batch_size_to_train_on)
  if config.eval_interval > 0:
    batch_size = max(batch_size, config.micro_batch_size_to_eval_on)
  if batch_size <= 0 or batch_size % (fsdp_size * ep_size) != 0:
    raise ValueError(
        f"TE MoE EP per-call batch size ({batch_size}) must be positive and divisible by FSDP * EP ({fsdp_size * ep_size})."
    )
  if config.num_experts % ep_size != 0:
    raise ValueError(f"num_experts={config.num_experts} must be divisible by EP size={ep_size}.")

  max_tokens_per_rank = (batch_size // (fsdp_size * ep_size)) * sequence_length
  worst_case_recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
      num_experts=config.num_experts,
      num_experts_per_tok=config.num_experts_per_tok,
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
  )
  recv_capacity_factor = None if config.ragged_buffer_factor <= 0 else config.ragged_buffer_factor
  recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
      num_experts=config.num_experts,
      num_experts_per_tok=config.num_experts_per_tok,
      max_tokens_per_rank=max_tokens_per_rank,
      ep_size=ep_size,
      recv_capacity_factor=recv_capacity_factor,
  )
  drop_on_overflow = recv_capacity_per_rank < worst_case_recv_capacity_per_rank
  hidden_dim = config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim
  signature = (
      jax.process_count(), jax.process_index(), ep_size, config.num_experts,
      max_tokens_per_rank, recv_capacity_per_rank, hidden_dim, drop_on_overflow,
  )
  global _te_moe_bootstrap_signature
  if _te_moe_bootstrap_signature == signature:
    return
  if _te_moe_bootstrap_signature is not None:
    raise ValueError(
        f"TE MoE EP was already bootstrapped with {_te_moe_bootstrap_signature}, " f"but this run needs {signature}."
    )
  with jax.set_mesh(mesh), mesh:
    max_logging.log(
        "Bootstrapping TE MoE EP: "
        f"world={jax.process_count()} rank={jax.process_index()} ep={ep_size} "
        f"num_experts={config.num_experts} max_tokens_per_rank={max_tokens_per_rank} "
        f"recv_capacity_per_rank={recv_capacity_per_rank} hidden_dim={hidden_dim} "
        f"recv_capacity_factor={recv_capacity_factor} drop_on_overflow={drop_on_overflow}"
    )
    ep_bootstrap(
        world_size=jax.process_count(),
        rank=jax.process_index(),
        num_experts=config.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        max_token_dtype=config.dtype,
        drop_on_overflow=drop_on_overflow,
    )
    record_ep_bootstrap_signature_for_moe(
        num_experts=config.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=hidden_dim,
        ep_size=ep_size,
    )
  _te_moe_bootstrap_signature = signature


def unscan_train_state_params(params, sharding, mesh, scan_axis, layer_groups):
  """
  Unrolls scanned parameter groups into per-layer entries.

  Args:
    train_state: training state with scanned `params`
    mesh: the mesh to use for sharding output
    scan_axis: axis along which scanning was applied (usually 0)
    layer_groups: list of tuples like:
      [("dense_layers", 4), ("moe_layers", 12)]
  """
  params_copy = params.unfreeze() if hasattr(params, "unfreeze") else params
  decoder = params_copy["params"]["decoder"]
  sharding_decoder = sharding["params"]["decoder"]

  # Helper function to remove the scan axis from a PartitionSpec
  def strip_scan_axis(pspec: P) -> P:
    """Removes the element at `scan_axis` from a PartitionSpec tuple."""
    spec_tuple = tuple(pspec)
    return P(*(spec_tuple[:scan_axis] + spec_tuple[scan_axis + 1 :]))

  for layer_name, num_layers in layer_groups:
    scanned_layers = decoder[layer_name]
    scanned_sharding = sharding_decoder[layer_name]

    # 1. Compute the target sharding for a single, unscanned layer.
    # This is done once per layer group, with no expensive compilation.
    unscanned_sharding_spec = jax.tree_util.tree_map(
        strip_scan_axis, jax.tree_util.tree_map(lambda x: x.spec, scanned_sharding)
    )
    unscanned_sharding = jax.tree_util.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), unscanned_sharding_spec)

    # 2. Create a list of PyTrees, one for each layer, by slicing the original.
    # This is more direct than repeatedly calling a JIT'd function.
    layer_pytrees = [
        jax.tree_util.tree_map(functools.partial(jnp.take, indices=i, axis=scan_axis), scanned_layers)
        for i in range(num_layers)
    ]

    # 3. Reshard each layer's PyTree and assign it to the new key.
    for i, layer_params in enumerate(layer_pytrees):
      resharded_params = jax.device_put(layer_params, unscanned_sharding)
      decoder[f"{layer_name}_{i}"] = resharded_params

    # 4. Clean up the original scanned parameter to free up memory.
    del decoder[layer_name]


def rescan_train_state_params(params, source_shardings, scan_axis, layer_groups):
  decoder = params["params"]["decoder"]
  sharding = source_shardings["params"]["decoder"]

  for layer_name, num_layers in layer_groups:

    def stack_layers(*layers):
      return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=scan_axis), *layers)

    compiled_stack = jax.jit(stack_layers, out_shardings=sharding[layer_name])
    layer_list = [decoder.pop(f"{layer_name}_{i}") for i in range(num_layers)]
    scanned = compiled_stack(*layer_list)
    decoder[layer_name] = scanned


def maybe_pad(inputs, tile_size):
  """Pads the inputs leading dimension to be divisible by tile_size."""
  inputs_dim = inputs.shape[0]
  padding_amount = 0
  if inputs_dim % tile_size:
    padding_amount = tile_size - inputs_dim % tile_size
    inputs = jax.lax.pad(
        inputs,
        jnp.array(0.0, dtype=inputs.dtype),
        [(0, padding_amount, 0)] + [(0, 0, 0)] * (inputs.ndim - 1),
    )
  return inputs, padding_amount


def is_eval(config) -> bool:
  """Return whether evaluation-specific kernel tile sizes should be used."""
  eval_rules = getattr(config, "logical_axis_rules_for_eval", None)
  train_rules = getattr(config, "logical_axis_rules", None)
  return bool(eval_rules and eval_rules != train_rules and get_logical_axis_rules() == eval_rules)
