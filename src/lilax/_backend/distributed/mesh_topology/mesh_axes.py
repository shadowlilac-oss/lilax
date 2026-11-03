"""Resolution of unspecified DCN/ICI parallelism mesh axes."""

import numpy as np


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
