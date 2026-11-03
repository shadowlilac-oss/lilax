"""Report parameter shardings: logical definitions against physical placement."""

from flax import nnx
import jax

from lilax._backend.distributed import sharding
from lilax._backend.utils import max_logging


def print_shardings_params(params, params_sharding, mesh, logical_annotations=None):
  """Print state shardings comparing logical definition and physical result."""
  if not isinstance(params, nnx.State):
    if not hasattr(params, "params"):
      params = {"params": params}
    if not hasattr(params_sharding, "params"):
      params_sharding = {"params": params_sharding}
    if logical_annotations and not hasattr(logical_annotations, "params"):
      logical_annotations = {"params": logical_annotations}
  leaves_params, _ = jax.tree_util.tree_flatten_with_path(params)
  leaves_sharding, _ = jax.tree_util.tree_flatten_with_path(params_sharding)
  if logical_annotations is not None:
    leaves_logical, _ = jax.tree_util.tree_flatten_with_path(logical_annotations)
    for (path, leaf_val), (_, leaf_sharding), (_, leaf_logical_val) in zip(
        leaves_params, leaves_sharding, leaves_logical
    ):
      path_str = "/".join(str(part.key if hasattr(part, "key") else part.name) for part in path)
      shape = jax.typeof(leaf_val)
      pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
      pspec_str = str(tuple(pspec))
      logical_str = str(leaf_logical_val)
      message = (
          f" {path_str}\n" f"    Shape:     {shape}\n" f"    Logical:   {logical_str}\n" f"    Physical:  {pspec_str}"
      )
      max_logging.info(message)
  else:
    for (path, leaf_val), (_, leaf_sharding) in zip(leaves_params, leaves_sharding):
      path_str = "/".join(str(part.key if hasattr(part, "key") else part.name) for part in path)
      shape = jax.typeof(leaf_val)
      pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
      pspec_str = str(tuple(pspec))
      message = f" {path_str}\n" f"    Shape:     {shape}\n" f"    Physical:  {pspec_str}"
      max_logging.info(message)
  print(flush=True)
