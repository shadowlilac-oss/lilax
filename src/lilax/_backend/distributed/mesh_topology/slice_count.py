"""Resolution of the number of slices in the device topology."""

from lilax._backend.utils import elastic_utils
from lilax._backend.observability import max_logging


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
