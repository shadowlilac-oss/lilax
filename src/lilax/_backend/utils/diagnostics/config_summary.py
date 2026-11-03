"""Write run config values to a TensorBoard summary writer."""

import jax

from lilax._backend.utils.tensorboard_logging import summary_writing


def add_config_to_summary_writer(config, summary_writer):
  if jax.process_index() == 0:
    if hasattr(config, "get_keys"):
      config_dict = config.get_keys()
    elif hasattr(config, "model_dump"):
      config_dict = config.model_dump()
    else:
      config_dict = dict(config)
    for key, value in config_dict.items():
      summary_writing.add_text_to_summary_writer(key, str(value), summary_writer)
