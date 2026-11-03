"""Device mesh ring ordering optimization for TPU v6e."""

import numpy as np

from lilax._backend.utils import max_logging


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
