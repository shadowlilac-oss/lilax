"""Prepare native model construction from runtime configuration."""

from lilax._backend.models.creation import model_factory
from lilax.runtime import Runtime
from lilax.configuration.model_mapping import project_model_config


def prepare_from_config(config, *, runtime=None, rngs=None, model_mode="train", quant_mode_str="train"):
  if runtime is None:
    runtime = Runtime(project_model_config(config))
  elif config is not runtime.public_config and config is not runtime.config:
    raise ValueError("The configuration must be the configuration used to initialize the runtime.")
  return runtime, {"rngs": rngs, "model_mode": model_mode, "quant_mode_str": quant_mode_str}
