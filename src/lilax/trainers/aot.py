"""Ahead-of-time compilation for pretraining."""

import os

import jax

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.trainers.pre_train import train_compile
from lilax.configuration import MaxTextConfig


def compile_train_step(config: MaxTextConfig):
  """Compile the fused pretraining step for the configured target topology."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  return train_compile.compile_train_step(resolve_runtime_config(config))


def is_oom(config: MaxTextConfig) -> bool:
  """Report whether compilation of the fused pretraining step runs out of memory."""
  return train_compile.is_oom(resolve_runtime_config(config))
