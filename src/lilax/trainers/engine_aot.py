"""Ahead-of-time compilation for the training engine."""

import os

import jax

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.training_engine import maxtext_engine_compile
from lilax.configuration import MaxTextConfig


def compile_engine(config: MaxTextConfig):
  """Compile each training engine kernel for the configured target topology."""
  if not isinstance(config, MaxTextConfig):
    raise TypeError("compile_engine requires an unresolved MaxTextConfig so HLO defaults can be set before initialization.")
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  if config.dump_hlo:
    defaults = {
        key: value for key, value in maxtext_engine_compile.HLO_DUMP_DEFAULTS.items()
        if key not in config._explicit_keys
    }
    config = config.model_copy(update=defaults)
  return maxtext_engine_compile.compile_engine(resolve_runtime_config(config))
