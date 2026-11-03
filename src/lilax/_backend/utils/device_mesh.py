"""Device mesh construction from parallelism config."""

import jax
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh
import numpy as np

from lilax._backend.common.common_types import ShardMode
from lilax._backend.experimental.lineage import lineage_adapter
from lilax._backend.utils import elastic_utils, max_logging, max_utils


def create_device_mesh(config, devices=None):
  if devices is None:
    devices = jax.devices()

  if getattr(config, "elastic_enabled", False):
    devices = elastic_utils.live_devices(config)
    num_slices = len(elastic_utils.live_slice_indices(config))
  else:
    num_slices = config.num_slices

  if config.subslice_shape and config.enable_single_controller and num_slices == 1:
    max_logging.log(f"Trying to create a subslice with shape: {config.subslice_shape}")
    subslice_shape = tuple(int(x) for x in config.subslice_shape.split(","))
    device_coords = [device.coords for device in devices]
    device_coords_np = np.array(device_coords)
    min_coords = device_coords_np.min(axis=0)
    subslice_devices = []
    for device in devices:
      coords = device.coords
      if all(min_coords[i] <= coords[i] < min_coords[i] + subslice_shape[i] for i in range(len(subslice_shape))):
        subslice_devices.append(device)
    devices = subslice_devices

  num_devices = len(devices)
  num_slices = 1 if getattr(config, "inference_benchmark_test", False) else num_slices
  num_devices_per_slice = num_devices // num_slices
  ici_parallelism = config.ici_parallelism.copy()
  ici_parallelism = max_utils.fill_unspecified_mesh_axes(ici_parallelism, num_devices_per_slice, "ICI")
  allow_split_physical_axes = config.allow_split_physical_axes if config.allow_split_physical_axes else False

  if getattr(config, "use_lineage", False):
    dcn_parallelism = (
        max_utils.fill_unspecified_mesh_axes(config.dcn_parallelism.copy(), num_slices, "DCN") if num_slices > 1 else None
    )
    mesh = lineage_adapter.create_device_mesh(
        ici_parallelism, devices, dcn_parallelism=dcn_parallelism, allow_split_physical_axes=allow_split_physical_axes
    )
  elif num_slices > 1:
    dcn_parallelism = config.dcn_parallelism.copy()
    dcn_parallelism = max_utils.fill_unspecified_mesh_axes(dcn_parallelism, num_slices, "DCN")
    if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = max_utils.create_custom_device_mesh(ici_parallelism, dcn_parallelism, devices, config.custom_mesh)
    else:
      mesh = mesh_utils.create_hybrid_device_mesh(
          ici_parallelism, dcn_parallelism, devices, allow_split_physical_axes=allow_split_physical_axes
      )
  elif allow_split_physical_axes:
    if max_utils.is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = mesh_utils.create_device_mesh([16, 16], devices, contiguous_submeshes=False, allow_split_physical_axes=False)
      mesh = max_utils.reshape_mesh_to_rings(mesh, config.custom_mesh)
      mesh = np.reshape(mesh, ici_parallelism)
    else:
      mesh = mesh_utils.create_device_mesh(
          ici_parallelism, devices, contiguous_submeshes=False, allow_split_physical_axes=allow_split_physical_axes
      )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)
    if config.optimize_mesh_for_tpu_v6e:
      mesh = max_utils.optimize_mesh_for_tpu_v6e(mesh, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")
  return mesh


def get_mesh_from_config(config, devices=None):
  devices_array = create_device_mesh(config, devices)
  if config.shard_mode == ShardMode.EXPLICIT:
    axis_types = tuple([AxisType.Explicit] * len(config.mesh_axes))
  else:
    axis_types = tuple([AxisType.Auto] * len(config.mesh_axes))
  return Mesh(devices_array, config.mesh_axes, axis_types=axis_types)
