"""Load and quantize a MaxText parameter checkpoint."""

import jax

from lilax._backend.inference.maxengine import maxengine
from lilax._backend.utils import max_utils


def validate_config(config):
  assert config.load_full_state_path == "", "Operation on full states not supported! Convert to parameter checkpoint first."


def load_and_quantize_checkpoint(config):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  validate_config(config)
  max_utils.print_system_information()
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  _, rng_load_params = jax.random.split(rng)
  return engine.load_params(rng_load_params)
