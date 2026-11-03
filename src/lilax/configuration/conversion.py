"""Project a public configuration into an inert CPU conversion configuration."""

from pathlib import Path
from typing import Literal

from lilax.configuration import schema
from lilax.configuration.values import load_config_values


def project_cpu_conversion_config(
    public_config: schema.MaxTextConfig, *, output_directory: str | Path,
    conversion_mode: Literal["automatic", "explicit"] = "automatic",
) -> schema.MaxTextConfig:
  """Build the configuration for an isolated on-the-fly conversion."""
  if conversion_mode == "explicit":
    return public_config.model_copy(update={
        "hardware": "cpu",
        "base_output_directory": str(output_directory),
        "skip_jax_distributed_system": True,
    })
  model_name = public_config.model_name.replace("-Instruct", "")
  values = load_config_values(model_name, _apply_environment=False)
  transfers = {
      "model_name": model_name,
      "hardware": "cpu",
      "base_output_directory": str(output_directory),
      "scan_layers": public_config.scan_layers,
      "use_multimodal": False,
      "skip_jax_distributed_system": True,
      "checkpoint_storage_use_ocdbt": public_config.checkpoint_storage_use_ocdbt,
      "checkpoint_storage_use_zarr3": public_config.checkpoint_storage_use_zarr3,
  }
  if "hf_access_token" in public_config._explicit_keys:
    transfers["hf_access_token"] = public_config.hf_access_token
  values.update(transfers)
  return schema.MaxTextConfig._from_resolved(values, transfers.keys())
