"""Report of mesh axes larger than one."""


def print_non_trivial_mesh_axis(mesh):
  """Print mesh axis if its axis size is larger than one."""
  for mesh_axis, axis_size in mesh.shape.items():
    if axis_size > 1:
      print(f"{mesh_axis}: {axis_size}", flush=True)
