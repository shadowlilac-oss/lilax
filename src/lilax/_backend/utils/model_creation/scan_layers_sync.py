"""Reconcile the configured scan_layers setting with the checkpoint being loaded."""

from lilax._backend.common.checkpoints import checkpointing
from lilax._backend.configs import pyconfig
from lilax._backend.utils import max_logging


def verify_and_sync_scan_layers(config):
  """Verify and sync scan_layers based on checkpoint metadata."""
  if not config.load_parameters_path:
    return config

  custom_metadata = checkpointing.load_checkpoint_metadata(config.load_parameters_path)
  saved_scan_layers = custom_metadata.get("scan_layers")
  if not isinstance(saved_scan_layers, bool):
    return config

  if saved_scan_layers == config.scan_layers:
    return config

  pydantic_config = getattr(config, "_pydantic_config", config)
  model_fields_set = getattr(pydantic_config, "model_fields_set", None)
  is_explicit = "scan_layers" in model_fields_set if model_fields_set is not None else True

  if is_explicit:
    if saved_scan_layers != config.scan_layers:
      raise ValueError(
          f"Configuration mismatch: Your run specifies scan_layers={config.scan_layers}, "
          f"but the checkpoint was saved with scan_layers={saved_scan_layers}."
      )
  else:
    max_logging.log(f"Setting scan_layers={saved_scan_layers} loaded from checkpoint metadata.")
    new_pydantic_config = pydantic_config.model_copy(update={"scan_layers": saved_scan_layers})
    if getattr(config, "_pydantic_config", None) is not None:
      config = pyconfig.HyperParameters(new_pydantic_config)
    else:
      config = new_pydantic_config

  return config
