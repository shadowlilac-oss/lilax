"""Native vLLM inference operations."""

import os

import jax

from .configuration import VLLMConfig
from ._model_source import serving_model_source, serving_overrides
from lilax._backend.inference.vllm_messages import build_chat_messages


def _prepare_vllm():
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["NEW_MODEL_DESIGN"] = "1"
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )


def create_vllm_llm(model, *, serving_config: VLLMConfig | None = None):
  """Create a vLLM engine from a public config after adapter registration."""
  from lilax._backend.inference.vllm_decode import create_vllm_llm as _create_vllm_llm
  from lilax._backend.integration.vllm import maxtext_vllm_adapter

  _prepare_vllm()
  options = serving_config or VLLMConfig()
  overrides = serving_overrides(options)
  additional_config = overrides.pop("additional_config")
  config, resource = serving_model_source(model, overrides=overrides)
  try:
    maxtext_vllm_adapter.register(config)
    llm = _create_vllm_llm(config, additional_config=additional_config)
  except BaseException:
    if resource is not None:
      resource.cleanup()
    raise
  llm._lilax_checkpoint_directory = resource
  return llm


__all__ = ["build_chat_messages", "create_vllm_llm"]
