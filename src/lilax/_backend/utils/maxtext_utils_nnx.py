"""Utils for MaxText NNX."""

from functools import partial
from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from lilax._backend.observability import max_logging
from lilax._backend.configs import pyconfig
from lilax._backend.common.common_types import MODEL_MODE_TRAIN


def create_nnx_rngs(
    config: pyconfig.HyperParameters, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None
) -> nnx.Rngs:
  """Create NNX Rngs."""
  if rng_key is None:
    rng_key = jax.random.PRNGKey(config.init_weights_seed)

  if model_mode == MODEL_MODE_TRAIN:
    return nnx.Rngs(
        params=jax.random.fold_in(rng_key, 0), dropout=jax.random.fold_in(rng_key, 1), aqt=jax.random.fold_in(rng_key, 2)
    )
  return nnx.Rngs(params=rng_key)


def nnx_extract_named_sharding(abstract_state: nnx.State) -> nnx.State:
  """Get named sharding from NNX abstract state."""
  return jax.tree.map(
      lambda x: x.sharding,
      abstract_state,
      is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
  )


def get_partition_spec_nnx(named_sharding: nnx.State) -> nnx.State:
  """Get mesh partition spec from named sharding."""
  return jax.tree.map(
      lambda x: x.spec,
      named_sharding,
      is_leaf=lambda x: isinstance(x, NamedSharding),
  )


def set_named_sharding_nnx(abstract_state: nnx.State, named_sharding: nnx.State) -> nnx.State:
  """Set named sharding to NNX abstract state."""
  return jax.tree.map(lambda x, y: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=y), abstract_state, named_sharding)


def move_memory_to_host(path: tuple[str, ...], x: NamedSharding) -> NamedSharding:
  """Change the memory kind of the named sharding to pinned host."""
  max_logging.log(f"max_utils.py: Moving {path} to host")
  return x.with_memory_kind(kind="pinned_host")


def move_memory_to_device(path: tuple[str, ...], x: NamedSharding) -> NamedSharding:
  """Change the memory kind of the named sharding to device."""
  max_logging.log(f"max_utils.py: Moving {path} to device")
  return x.with_memory_kind(kind="device")


def create_nnx_sharded_model(
    abstract_model: nnx.Module,
    init_fn: Callable,
    mesh: Mesh | None = None,
    named_sharding: nnx.State | None = None,
) -> nnx.Module:
  """Create the model with the given sharding."""
  graphdef, abstract_state = nnx.split(abstract_model)
  if named_sharding is None:
    named_sharding = nnx_extract_named_sharding(abstract_state)

  if mesh is None:
    mesh = abstract_model.mesh

  @partial(jax.jit, out_shardings=named_sharding)
  def create_sharded_state():
    model = init_fn()
    return jax.lax.with_sharding_constraint(nnx.state(model), named_sharding)

  with jax.set_mesh(mesh):
    sharded_state = create_sharded_state()
  return nnx.merge(graphdef, sharded_state)


def nnx_ensure_scan_leading_axis(tree, length):
  """Broadcasts scalar-like variables to have a leading scan axis."""

  def _op(x):
    is_var = isinstance(x, nnx.Variable)
    val = x.get_value() if is_var else x
    if hasattr(val, "shape") and len(val.shape) == 0:
      new_val = jax.numpy.broadcast_to(val, (length,))
      return x.replace(value=new_val) if is_var else new_val
    return x

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


def nnx_update_sharding_meta(variable, transform_fn):
  """Generic helper to apply a list transformation to all sharding-related metadata."""
  if not (hasattr(variable, "get_metadata") and hasattr(variable, "replace")):
    return variable

  meta = variable.get_metadata()
  updates = {}

  for key in ["sharding", "out_sharding", "sharding_names"]:
    if (val := meta.get(key)) and isinstance(val, (P, tuple, list)):
      new_list = list(val)
      transformed = transform_fn(new_list)
      updates[key] = P(*transformed) if isinstance(val, P) else tuple(transformed)

  if updates:
    return variable.replace(**updates)
  return variable


def nnx_remove_scan_axis(tree, name="layers"):
  """Removes the given scan axis from the PartitionSpec."""

  def _op(x):
    if not isinstance(x, nnx.Variable):
      return x

    axis_name = x.get_metadata().get(nnx.PARTITION_NAME, name)

    def remove_fn(l):
      removed = axis_name in l
      if removed:
        l.remove(axis_name)
      if len(l) > x.get_value().ndim:
        if removed:
          raise ValueError(
              f"Sharding names {l} still exceed value rank {x.get_value().ndim} after removing scan axis "
              f"{axis_name!r}; the partition metadata is inconsistent."
          )
        raise ValueError(
            f"Scan axis {axis_name!r} not found in sharding names {l} for a rank-{x.get_value().ndim} value; "
            "the partition metadata is inconsistent."
        )
      return l

    return nnx_update_sharding_meta(x, remove_fn)

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


def nnx_add_and_sync_scan_axis(tree, name="layers", pos=0):
  """Restores the scan axis on each variable's value and sharding metadata."""

  def _op(x):
    if not isinstance(x, nnx.Variable):
      return x

    axis_name = x.get_metadata().get(nnx.PARTITION_NAME, name)
    target = x.get_metadata().get("param_scan_axis", pos)

    val = x.get_value()
    if target != 0 and hasattr(val, "ndim") and val.ndim > target:
      x = x.replace(value=jnp.moveaxis(val, 0, target))

    def add_fn(l):
      if axis_name not in l:
        while len(l) < x.get_value().ndim - 1:
          l.append(None)
        l.insert(target, axis_name)
      else:
        while len(l) < x.get_value().ndim:
          l.append(None)
      return l

    return nnx_update_sharding_meta(x, add_fn)

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
