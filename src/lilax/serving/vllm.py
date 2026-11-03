"""Native vLLM inference operations."""

import os

import jax

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.vllm_decode import (
    build_chat_messages,
    create_vllm_llm as _create_vllm_llm,
    decode_with_tunix as _decode_with_tunix,
    decode_with_vllm as _decode_with_vllm,
)
from lilax._backend.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from lilax._backend.integration.vllm import maxtext_vllm_adapter


def _prepare_vllm():
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["NEW_MODEL_DESIGN"] = "1"
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )


def create_vllm_llm(config, **overrides):
  """Create a vLLM engine from a public config after adapter registration."""
  _prepare_vllm()
  config = resolve_runtime_config(config)
  maxtext_vllm_adapter.register(config)
  return _create_vllm_llm(config, **overrides)


def decode_with_vllm(config):
  """Decode with a public MaxText configuration."""
  _prepare_vllm()
  config = resolve_runtime_config(config)
  maxtext_vllm_adapter.register(config)
  return _decode_with_vllm(config)


def decode_with_tunix(model):
  """Decode using the native model and mesh bound to a MaxText model."""
  _prepare_vllm()
  maxtext_vllm_adapter.register(model.runtime.config)
  native_model = model.native_model
  if isinstance(native_model, TunixMaxTextAdapter):
    native_model = native_model.base
  return _decode_with_tunix(model.runtime.config, native_model, model.runtime.mesh)


__all__ = ["build_chat_messages", "create_vllm_llm", "decode_with_tunix", "decode_with_vllm"]
