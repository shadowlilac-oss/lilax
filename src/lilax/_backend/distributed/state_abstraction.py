"""Abstract shaped state and mesh shardings for Linen and NNX train states."""

import functools

from flax import nnx
from flax.core.meta import get_partition_spec
from flax.core.spmd import logical_axis_rules as axis_rules
import jax
from jax.sharding import NamedSharding

from lilax._backend.common import state_initialization
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.distributed import sharding
from lilax._backend.pytree import partitioning


def get_abstract_state(config, mesh, init_state_fn, is_training=True):
  if isinstance(init_state_fn, functools.partial) and init_state_fn.func is state_initialization.init_initial_state:
    with axis_rules(config.logical_axis_rules):
      abstract_state = jax.eval_shape(init_state_fn)
      logical_annotations = get_partition_spec(abstract_state)
      state_mesh_annotations = sharding.logical_to_mesh(logical_annotations, mesh)
      state_mesh_shardings = sharding.logical_to_mesh_sharding(logical_annotations, mesh)
      unboxed_abstract_state = partitioning.unbox_logicallypartioned(abstract_state)
    return unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings
  return get_abstract_state_nnx(config, mesh, init_state_fn, is_training)


def _extract_primary_sharding(s):
  if isinstance(s, (jax.sharding.Sharding, jax.sharding.PartitionSpec)):
    return s
  if hasattr(s, "qvalue"):
    return _extract_primary_sharding(s.qvalue)
  leaves = jax.tree.leaves(s)
  return leaves[0] if leaves else s


def _make_abstract_leaf(leaf_a, leaf_s):
  leaf_s = _extract_primary_sharding(leaf_s)
  if hasattr(leaf_s, "spec") and len(leaf_s.spec) > len(leaf_a.shape):
    if "local_layers" in leaf_s.spec and len(leaf_s.spec) - 1 == len(leaf_a.shape):
      spec_tuple = tuple(axis for axis in leaf_s.spec if axis != "local_layers")
    else:
      spec_tuple = leaf_s.spec[: len(leaf_a.shape)]
    leaf_s = jax.sharding.NamedSharding(leaf_s.mesh, jax.sharding.PartitionSpec(*spec_tuple))
  return jax.ShapeDtypeStruct(leaf_a.shape, leaf_a.dtype, sharding=leaf_s)


def _to_abstract_var(a_var, s_var):
  a_val = a_var.get_value()
  s_val = getattr(s_var, "sharding", None) if isinstance(s_var, nnx.Variable) else s_var
  if s_val is None and isinstance(s_var, nnx.Variable):
    s_val = s_var.get_value()
  if type(a_val) in (jax.Array, jax.ShapeDtypeStruct) or (hasattr(a_val, "shape") and not hasattr(a_val, "qvalue")):
    new_val = _make_abstract_leaf(a_val, s_val)
  else:
    s_tree = (
        jax.tree.map(lambda _: s_val, a_val)
        if isinstance(s_val, (jax.sharding.Sharding, jax.sharding.PartitionSpec)) else s_val
    )
    new_val = jax.tree.map(
        _make_abstract_leaf, a_val, s_tree, is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype")
    )
  return a_var.replace(value=new_val)


def get_abstract_state_nnx(config, mesh, nnx_init_trainstate_fn, is_training=True):
  assert nnx_init_trainstate_fn is not None, "get_abstract_state_nnx: init function must be given."
  with axis_rules(config.logical_axis_rules):
    abs_model = nnx.eval_shape(nnx_init_trainstate_fn)
    _, abs_var_state = nnx.split(abs_model)
    named_sharding_state = sharding.nnx_construct_named_sharding(abs_var_state, mesh)
    abstract_state = jax.tree.map(
        _to_abstract_var, abs_var_state, named_sharding_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )

  state_mesh_shardings = maxtext_utils_nnx.nnx_extract_named_sharding(abstract_state)
  if is_training and config.shard_optimizer_over_data:
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        functools.partial(sharding.add_data_to_sharding, mesh), abstract_state.optimizer, state_mesh_shardings.optimizer
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.optimizer_memory_host_offload:
    optimizer_sharding = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host, state_mesh_shardings.optimizer,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )
    state_mesh_shardings.optimizer = optimizer_sharding
  if is_training and config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."
    _, state_params, _ = nnx.split(state_mesh_shardings, nnx.Param, ...)
    state_params = jax.tree_util.tree_map_with_path(
        maxtext_utils_nnx.move_memory_to_host, state_params, is_leaf=lambda x: isinstance(x, NamedSharding)
    )
    nnx.update(state_mesh_shardings, state_params)
  abstract_sharded_state = maxtext_utils_nnx.set_named_sharding_nnx(abstract_state, state_mesh_shardings)
  state_mesh_annotations = maxtext_utils_nnx.get_partition_spec_nnx(state_mesh_shardings)
  return abstract_sharded_state, state_mesh_annotations, state_mesh_shardings
