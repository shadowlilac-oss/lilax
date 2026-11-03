"""Prepare native model construction from runtime configuration."""

from lilax._backend.utils.model_creation import model_factory
from lilax.runtime import Runtime


def from_config_linen(config, *, runtime=None, model_mode="train", quant_mode_str="train"):
  if runtime is None:
    runtime = Runtime(config).initialize()
  elif config is not runtime.public_config and config is not runtime.config:
    raise ValueError("The configuration must be the configuration used to initialize the runtime.")
  if runtime.mesh is None:
    raise ValueError("Runtime must be initialized before constructing a model.")
  if runtime.config.use_m3_model:
    raise ValueError("The native M3 factory does not construct a Linen model.")
  return model_factory.from_config(
      runtime.config, mesh=runtime.mesh, model_mode=model_mode, quant_mode_str=quant_mode_str
  )


def prepare_from_config(config, *, runtime=None, rngs=None, model_mode="train", quant_mode_str="train"):
  if runtime is None:
    runtime = Runtime(config)
  elif config is not runtime.public_config and config is not runtime.config:
    raise ValueError("The configuration must be the configuration used to initialize the runtime.")
  return runtime, {"rngs": rngs, "model_mode": model_mode, "quant_mode_str": quant_mode_str}
