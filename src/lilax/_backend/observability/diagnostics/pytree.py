"""Pytree size and memory diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from lilax._backend.observability import max_logging


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def calculate_total_params_per_chip(params):
  """Calculate total params per chip."""

  def calculate_leaf_params_per_chip(arr):
    shard = arr.addressable_shards[0]
    return np.prod(shard.data.shape)

  params_sizes_per_chip = jax.tree_util.tree_map(calculate_leaf_params_per_chip, params)
  total_parameters_per_chip = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes_per_chip)
  return total_parameters_per_chip


def _bytes_of(x):
  """Return the number of bytes used by a single leaf in a pytree.
  Handles concrete arrays (NumPy/JAX), abstract shapes, scalars, and None.
  Unknown types default to 0.
  """
  if isinstance(x, jax.ShapeDtypeStruct):
    return int(np.prod(x.shape)) * int(jnp.dtype(x.dtype).itemsize)
  if hasattr(x, "nbytes"):
    return int(x.nbytes)
  if isinstance(x, (int, float, bool)):
    return int(np.array(x).nbytes)
  if x is not None:
    max_logging.log(f"Unsupported leaf type in calculate_bytes_from_pytree: {type(x)}")
  return 0


def calculate_bytes_from_pytree(params):
  """Return the total memory footprint (in bytes) of all leaves in a pytree.

  Each leaf is measured using `_bytes_of`. Non-array or unsupported types
  contribute 0 unless they are scalars.
  """
  return sum(map(_bytes_of, jax.tree_util.tree_leaves(params)))


def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes / num_params


def print_pytree_shape(print_str, ptree):
  print("\n")
  print(print_str)
  print(jax.tree_util.tree_map(lambda x: x.shape, ptree))


def print_model_vars(print_str, model_vars):
  for k in model_vars:
    print(f"{print_str} key{k}:")
    print(f"\t {model_vars[k]}")


def summarize_pytree_data(params, name="Params", raw=False):
  """Generate basic metrics of a given Pytree."""
  num_params, total_param_size, avg_param_size = summarize_size_from_pytree(params)
  if not raw:
    num_params_in_billions = num_params / 1e9
    total_param_size_in_gb = total_param_size / 1e9
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params_in_billions:.3f} billion \n"
        f"\tTotal memory usage: {total_param_size_in_gb:.3f} GB \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  else:
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params:.3f} \n"
        f"\tTotal memory usage: {total_param_size:.3f} bytes \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  return num_params, total_param_size, avg_param_size


def print_mem_stats(label: str):
  max_logging.log(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      max_logging.log(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tMemstats unavailable, error: {ex}")


def print_cpu_ram_stats(label: str):
  """Print stats of CPU RAM usage/availability."""
  max_logging.log(f"\nRAMstats: {label}:")
  try:
    ram = psutil.virtual_memory()
    total = round(ram.total / 2**30, 2)
    available = round(ram.available / 2**30, 2)
    used = round(ram.used / 2**30, 2)
    max_logging.log(f"\tUsing (GB) {used} / {total} ({used/total:%}) -->  Available:{available}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tRAM stats unavailable, error: {ex}")


def print_compiled_memory_stats(compiled_stats, prefix: str = ""):
  """Prints a summary of the compiled memory statistics."""
  if compiled_stats is None:
    return
  prefix_str = f"[{prefix}] " if prefix else ""

  def bytes_to_gb(num_bytes):
    return num_bytes / (1024**3)

  output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
  temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
  argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
  alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
  host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
  total_gb = output_gb + temp_gb + argument_gb - alias_gb
  max_logging.log(
      f"{prefix_str}Total estimated memory size: {total_gb:.1f} GB, estimated output"
      f" size: {output_gb:.1f} GB, estimated temp size: {temp_gb:.1f} GB, "
      f"estimated argument size: {argument_gb:.1f} GB, Estimated host temp"
      f" size: {host_temp_gb:.1f} GB."
  )
  max_logging.log(f"{prefix_str}Note that compiler could over-estimate the HBM usage.")
