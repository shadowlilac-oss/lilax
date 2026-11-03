"""Initialize, reshard, and load MaxEngine NNX parameter state."""

import jax
from flax import nnx
from flax.core.spmd import logical_axis_rules

from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from lilax._backend.utils import max_logging
from lilax._backend.utils import maxtext_utils_nnx
from lilax._backend.utils import model_creation_utils
from lilax._backend.utils import sharding


def initialize_model_state(engine):
  with logical_axis_rules(engine.config.logical_axis_rules):
    _, full_abs = nnx.split(engine.model)
    full_sharding = sharding.nnx_construct_named_sharding(full_abs, engine._mesh)
    concrete_model = maxtext_utils_nnx.create_nnx_sharded_model(
        engine.model,
        engine._create_model_fn,
        mesh=engine._mesh,
        named_sharding=full_sharding,
    )
    graphdef, _, _, rest_state = nnx.split(concrete_model, nnx.Param, nnx.Cache, ...)
    engine.graphdef = graphdef
    del concrete_model
  return rest_state


def reshard_params(engine, params):
  print("Resharding given NNX params")
  _, params_abs, _ = nnx.split(engine.model, nnx.Param, ...)
  with logical_axis_rules(engine.config.logical_axis_rules):
    target_shardings = sharding.nnx_construct_named_sharding(params_abs, engine._mesh)
    params_state = jax.device_put(params, target_shardings)
  return params_state


def load_checkpoint(engine, rest_state):
  max_logging.log("Loading NNX params via from_pretrained")
  with engine._mesh:
    nnx_model = model_creation_utils.from_pretrained(
        engine.config,
        mesh=engine._mesh,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        quant_mode_str=engine._nnx_quant_mode_str,
    )
  _, params_state, _, loaded_rest_state = nnx.split(nnx_model, nnx.Param, nnx.Cache, ...)
  loaded_rest_dict = nnx.to_pure_dict(loaded_rest_state)
  rest_dict = nnx.to_pure_dict(rest_state)

  def _overlay(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
      for k, v in dst.items():
        if k in src:
          dst[k] = _overlay(v, src[k])
      return dst
    if isinstance(dst, dict) or isinstance(src, dict):
      return dst
    return src

  rest_dict = _overlay(rest_dict, loaded_rest_dict)
  nnx.replace_by_pure_dict(rest_state, rest_dict)
  engine._nnx_rest_state = rest_state
  del nnx_model, loaded_rest_state, loaded_rest_dict, rest_dict
  return params_state
