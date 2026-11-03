"""Utilities for MaxText Muon dimension numbers and shardings."""

from typing import Tuple

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax

from lilax._backend.optimizers.muon import ShardedMuonDimensionNumbers as smdn
from lilax._backend.utils import maxtext_utils, muon_utils
from lilax._backend.utils import sharding as sharding_lib


def _get_leaf_value(leaf):
  if hasattr(leaf, "get_value"):
    return leaf.get_value()
  return leaf


def _extract_sharding(leaf):
  val = _get_leaf_value(leaf)
  if isinstance(val, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)):
    return val
  sharding = getattr(leaf, "sharding", None)
  if isinstance(sharding, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)):
    return sharding
  return None


def get_maxtext_muon_weight_dimension_numbers(model, config=None, mesh=None, verbose=False):
  if mesh is None:
    if config is None:
      raise ValueError("Either mesh or config must be provided to get_maxtext_muon_weight_dimension_numbers.")
    mesh = maxtext_utils.get_mesh_from_config(config)
  if config is not None:
    logical_rules = getattr(config, "logical_axis_rules", ())
  else:
    logical_rules = ()

  with jax.set_mesh(mesh), logical_axis_rules(logical_rules):
    _, abstract_param, _ = nnx.split(model, nnx.Param, ...)
    named_sharding_state = sharding_lib.nnx_construct_named_sharding(abstract_param, mesh)
    abstract_dict = nnx.to_pure_dict(abstract_param)
    named_sharding_dict = nnx.to_pure_dict(named_sharding_state)

    def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf, abs_leaf):
      path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
      abs_val = _get_leaf_value(abs_leaf)
      val_shape = getattr(abs_val, "shape", None)
      dim_num = muon_utils.transform_logic(path_strings, shape=val_shape)
      if dim_num is not None:
        sharding = _extract_sharding(leaf)
        return smdn(reduction_axis=dim_num.reduction_axis, output_axis=dim_num.output_axis, sharding=sharding)
      return None

    muon_weight_dimension_numbers = jax.tree.map_with_path(apply_transform_nnx, named_sharding_dict, abstract_dict)
    muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)
    if verbose:
      _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
    return muon_weight_dimension_numbers


get_sharded_muon_weight_dimension_numbers = get_maxtext_muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  return muon_utils._print_structure_debug(abstract_param, muon_weight_dimension_numbers)
