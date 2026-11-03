"""Build a Linen decode TrainState from an NNX pretrained checkpoint."""

from functools import partial

from flax import nnx
import jax

from lilax._backend.common import state_initialization
from lilax._backend.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from lilax._backend.m3.core import checkpoint as m3_checkpoint
from lilax._backend.utils import state_abstraction
from lilax._backend.utils.model_creation.pretrained import from_pretrained


def setup_decode_state_from_nnx(model, config, rng, mesh):
  """Load an NNX checkpoint into a linen decode TrainState."""
  init_state_fn = partial(state_initialization.init_initial_state, model, None, config, False, rng)
  _, state_mesh_annotations, _ = state_abstraction.get_abstract_state(config, mesh, init_state_fn, False)
  nnx_model = from_pretrained(config, mesh=mesh, model_mode=MODEL_MODE_AUTOREGRESSIVE)

  def _state_to_dict(tree):
    if isinstance(tree, nnx.Variable):
      return tree.get_value()
    if hasattr(tree, "items") and not isinstance(tree, jax.Array):
      return {key: _state_to_dict(value) for key, value in tree.items()}
    return tree

  nnx_param_state = nnx.state(nnx_model, nnx.Param)
  raw_params = _state_to_dict(nnx_param_state)
  if config.use_m3_model:
    raw_params = m3_checkpoint.to_legacy_layer_names(raw_params)
  del nnx_model, nnx_param_state

  params = {"params": raw_params}
  state = state_initialization.init_decode_state(model.apply, params)
  return state, state_mesh_annotations
