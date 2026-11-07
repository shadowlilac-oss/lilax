"""Point a configuration at a converted checkpoint when checkpoint conversion is requested."""

from lilax._backend.checkpoint_conversion.cpu_bootstrap import convert_on_cpu
from lilax._backend.configs import pyconfig


def use_converted_checkpoint_if_possible(config):
  """Return a config loading the converted checkpoint, converting on CPU when none exists yet."""
  if not config.convert_checkpoint_if_possible or config.load_parameters_path:
    return config
  load_parameters_path = convert_on_cpu(config, None, config.base_output_directory)
  pydantic_config = getattr(config, "_pydantic_config", config)
  return pyconfig.HyperParameters(pydantic_config.model_copy(update={"load_parameters_path": load_parameters_path}))
