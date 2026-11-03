"""Create abstract, sharded NNX models together with their concrete model factories."""

from typing import Callable

from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax

from lilax._backend.common.common_types import MODEL_MODE_TRAIN
from lilax._backend.utils import sharding
from lilax._backend.utils.model_creation.m3_default_sharding import _apply_m3_default_named_sharding
from lilax._backend.utils.model_creation.model_factory import get_nnx_create_model_fn


def create_nnx_abstract_model(
    config, mesh=None, devices=None, model_mode=MODEL_MODE_TRAIN, rng_key=None, *, quant_mode_str: str = "train"
) -> tuple[Callable, nnx.Module]:
  """Create a model factory and an abstract, sharded NNX model."""
  with logical_axis_rules(config.logical_axis_rules):
    create_model_fn = get_nnx_create_model_fn(config, mesh, devices, model_mode, rng_key, quant_mode_str=quant_mode_str)
    abs_model = nnx.eval_shape(create_model_fn)
    if mesh is None:
      mesh = abs_model.mesh
    graphdef, abs_var_state = nnx.split(abs_model)
    named_sharding_state = sharding.nnx_construct_named_sharding(abs_var_state, mesh)
    if getattr(config, "use_m3_model", False) and mesh is not None and getattr(mesh, "size", 1) > 1:
      named_sharding_state = _apply_m3_default_named_sharding(abs_var_state, named_sharding_state, mesh)
    abstract_state = jax.tree.map(
        lambda abstract, named: jax.ShapeDtypeStruct(abstract.shape, abstract.dtype, sharding=named),
        abs_var_state,
        named_sharding_state,
    )
    return create_model_fn, nnx.merge(graphdef, abstract_state)
