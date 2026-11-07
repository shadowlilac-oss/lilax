"""Reshard MaxText parameter checkpoints on a caller-configured mesh."""

import math
import os
import time
from typing import Sequence

from etils import epath
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from orbax.checkpoint import v1 as ocp

from lilax._backend.checkpoint_conversion.checkpoint_cpu_bootstrap import run_on_cpu
from lilax._backend.checkpoint_conversion.utils.utils import print_peak_memory
from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.common.checkpoints import flat_moe
from lilax._backend.common import train_state_nnx
from lilax._backend.common.common_types import MODEL_MODE_TRAIN
from lilax._backend.layers.quantization.configure import configure_quantization
from lilax._backend.distributed import device_mesh, sharding
from lilax._backend.observability import max_logging
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.observability.diagnostics import system_info
from lilax._backend.models.creation import model_factory, nnx_abstract_model


def create_abstract_params(config, mesh: Mesh) -> nnx.State:
  if not (config.quantization and config.use_qwix_quantization):
    _, abstract_model = nnx_abstract_model.create_nnx_abstract_model(config, mesh=mesh, model_mode=MODEL_MODE_TRAIN)
  else:
    def _create_model():
      rngs = maxtext_utils_nnx.create_nnx_rngs(config, model_mode=MODEL_MODE_TRAIN)
      quant = configure_quantization(config)
      return model_factory.get_transformer_model(config, mesh, quant, model_mode=MODEL_MODE_TRAIN, rngs=rngs)

    with logical_axis_rules(config.logical_axis_rules):
      graphdef, abstract_state = nnx.split(nnx.eval_shape(_create_model))
      named_shardings = sharding.nnx_construct_named_sharding(abstract_state, mesh)
      abstract_state = jax.tree.map(
          lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), abstract_state, named_shardings
      )
    abstract_model = nnx.merge(graphdef, abstract_state)
  return checkpointing._abstract_params(nnx.state(abstract_model))


def _stored_weights_metadata(load_parameters_path):
  path_str = str(load_parameters_path).rstrip("/")
  root = epath.Path(path_str.removesuffix("/items") if path_str != "items" else ".")
  checkpointable_name = "items" if (root / "items").exists() else None
  try:
    stored = ocp.metadata(root, checkpointable_name=checkpointable_name).metadata
  except Exception as e:
    max_logging.log(f"Could not read checkpoint metadata ({e}); loading straight into the target sharding.")
    return None
  weights = stored.get("params") if isinstance(stored, dict) else None
  if isinstance(weights, dict) and isinstance(weights.get("params"), dict):
    weights = weights["params"]
  return weights if isinstance(weights, dict) else None


def _lookup(tree, path):
  for key in path:
    if not isinstance(tree, dict):
      return None
    tree = tree.get(key, tree.get(str(key)))
  return tree


def chunk_aligned_sharding(
    shape: Sequence[int], chunk_shape: Sequence[int] | None, devices: Sequence[jax.Device]
) -> NamedSharding | None:
  if chunk_shape is None or len(chunk_shape) != len(shape):
    return None
  grid = [s // c if c and s % c == 0 else 1 for s, c in zip(shape, chunk_shape)]
  splits = [1] * len(shape)
  budget = len(devices)
  for axis in sorted(range(len(shape)), key=lambda a: -grid[a]):
    splits[axis] = max(d for d in range(1, grid[axis] + 1) if grid[axis] % d == 0 and d <= budget)
    budget //= splits[axis]
  split_axes = [a for a in range(len(shape)) if splits[a] > 1]
  if not split_axes:
    return NamedSharding(Mesh(np.asarray(devices[:1]), ("chunks",)), PartitionSpec())
  names = tuple(f"chunks_{a}" for a in split_axes)
  mesh_devices = np.asarray(devices[: math.prod(splits)]).reshape([splits[a] for a in split_axes])
  spec = PartitionSpec(*(f"chunks_{a}" if splits[a] > 1 else None for a in range(len(shape))))
  return NamedSharding(Mesh(mesh_devices, names), spec)


def _reshard_on_host(x: jax.Array, target_sharding: NamedSharding) -> jax.Array:
  if x.sharding == target_sharding:
    return x
  host = np.asarray(x)
  x.delete()
  return jax.device_put(host, target_sharding)


def _to_flat_moe_on_host(x: jax.Array, flat_target: jax.ShapeDtypeStruct) -> jax.Array:
  host = np.asarray(x)
  x.delete()
  num_experts, rows = host.shape[0], host.shape[-2]
  src = np.moveaxis(host, -2, 1)

  def _block(index):
    start, stop, _ = index[0].indices(num_experts * rows)
    parts = []
    while start < stop:
      expert, lo = divmod(start, rows)
      hi = min(rows, lo + stop - start)
      parts.append(src[(expert, slice(lo, hi)) + tuple(index[1:])])
      start += hi - lo
    return np.concatenate(parts).astype(flat_target.dtype, copy=False)

  return jax.make_array_from_callback(flat_target.shape, flat_target.sharding, _block)


def without_replicas(target_sharding: NamedSharding) -> NamedSharding:
  mesh, spec = target_sharding.mesh, target_sharding.spec
  used = {name for entry in spec if entry is not None for name in ((entry,) if isinstance(entry, str) else entry)}
  replicated = [name for name in mesh.axis_names if name not in used and mesh.shape[name] > 1]
  if not replicated:
    return target_sharding
  index = tuple(slice(0, 1) if name in replicated else slice(None) for name in mesh.axis_names)
  return NamedSharding(Mesh(mesh.devices[index], mesh.axis_names), spec)


def load_resharded_params(config, mesh: Mesh) -> nnx.State:
  target = create_abstract_params(config, mesh)
  load_kwargs = {"use_ocdbt": config.checkpoint_storage_use_ocdbt, "use_zarr3": config.checkpoint_storage_use_zarr3}
  stored = _stored_weights_metadata(config.load_parameters_path)
  if stored is None:
    return checkpointing.load_params_from_path(
        config.load_parameters_path, target, config.checkpoint_storage_concurrent_gb,
        convert_flat_moe_weights=bool(config.moe_flat_fsdp_weights), **load_kwargs,
    )
  target_weights = nnx.traversals.flatten_mapping(target.to_pure_dict())
  flat_moe_plan = {}
  if config.moe_flat_fsdp_weights:
    plan = flat_moe._flat_moe_restore_plan(nnx.traversals.unflatten_mapping(target_weights), stored)
    flat_moe_plan = {tuple(path): value for path, value in plan.items()}
  devices = list(mesh.devices.flat)
  staging, num_aligned = {}, 0
  for path, want in target_weights.items():
    if path in flat_moe_plan:
      want = flat_moe_plan[path][1]
    meta = _lookup(stored, path)
    staging_sharding = None
    if meta is not None and tuple(getattr(meta, "shape", ()) or ()) == tuple(want.shape):
      storage = getattr(meta, "storage_metadata", None)
      staging_sharding = chunk_aligned_sharding(want.shape, getattr(storage, "chunk_shape", None), devices)
    num_aligned += staging_sharding is not None
    dtype = getattr(meta, "dtype", None) or want.dtype
    staging[path] = jax.ShapeDtypeStruct(want.shape, dtype, sharding=staging_sharding or want.sharding)
  max_logging.log(f"Restoring {num_aligned}/{len(staging)} weights in a chunk-aligned staging sharding.")
  staging_state = jax.tree.map(lambda x: x, target)
  nnx.replace_by_pure_dict(staging_state, nnx.traversals.unflatten_mapping(staging))
  start = time.time()
  restored = checkpointing.load_params_from_path(
      config.load_parameters_path, staging_state, config.checkpoint_storage_concurrent_gb,
      convert_flat_moe_weights=False, **load_kwargs,
  )
  max_logging.log(f"Elapse for staged checkpoint restore: {(time.time() - start) / 60:.2f} min")
  start = time.time()
  restored_weights = nnx.traversals.flatten_mapping(restored.to_pure_dict())
  del restored
  resharded = {}
  for path, want in target_weights.items():
    x = restored_weights.pop(path)
    target_sharding = without_replicas(want.sharding)
    if path in flat_moe_plan:
      flat_target = jax.ShapeDtypeStruct(want.shape, x.dtype, sharding=target_sharding)
      resharded[path] = _to_flat_moe_on_host(x, flat_target)
    else:
      resharded[path] = _reshard_on_host(x, target_sharding)
  max_logging.log(f"Elapse for in-memory reshard to the target: {(time.time() - start) / 60:.2f} min")
  params = jax.tree.map(lambda x: x, target)
  nnx.replace_by_pure_dict(params, nnx.traversals.unflatten_mapping(resharded))
  return params


def save_params(config, params: nnx.State, step: int = 0) -> str:
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.base_output_directory, enable_checkpointing=True, use_async=False, save_interval_steps=1,
      use_ocdbt=config.checkpoint_storage_use_ocdbt, use_zarr3=config.checkpoint_storage_use_zarr3,
      enable_continuous_checkpointing=True, checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
  )
  if checkpoint_manager is None:
    raise RuntimeError("Failed to create Orbax checkpoint manager.")
  state = train_state_nnx.to_linen_checkpoint_dict({"model": params.to_pure_dict(), "optimizer": {"step": step}})
  if not checkpointing.save_checkpoint(checkpoint_manager, step, state):
    raise RuntimeError(f"Checkpoint for step {step} was not saved; does it already exist in the output directory?")
  checkpointing.wait_until_finished(checkpoint_manager)
  return os.path.join(config.base_output_directory, str(step), "items")


def reshard_checkpoint(config, simulated_cpu_devices_count=16):
  return run_on_cpu("reshard_checkpoint", config, simulated_cpu_devices_count=simulated_cpu_devices_count)


def _reshard_checkpoint(config):
  system_info.print_system_information()
  max_logging.log(
      f"Reshard checkpoint with weight dtype: {config.weight_dtype},"
      f" moe_flat_fsdp_weights: {config.moe_flat_fsdp_weights}"
  )
  mesh = Mesh(device_mesh.create_device_mesh(config), config.mesh_axes)
  max_logging.log(f"Target mesh: {dict(mesh.shape)}")
  start = time.time()
  params = load_resharded_params(config, mesh)
  max_logging.log(f"Elapse for checkpoint load (with reshard): {(time.time() - start) / 60:.2f} min")
  start = time.time()
  path = save_params(config, params)
  max_logging.log(f"Saved checkpoint: {path}")
  max_logging.log(f"Elapse for checkpoint save: {(time.time() - start) / 60:.2f} min")
  print_peak_memory()
  return path
