"""LoRA adapter loading and application for MaxEngine."""

import os.path

from lilax._backend.utils import lora_utils


def load_single_adapter(config, abstract_params, adapter_path):
  adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
  adapter_weights_path = os.path.join(adapter_path, "0", "items")

  params, config = lora_utils.load_adapter(config, abstract_params, adapter_config_path, adapter_weights_path)

  if config is None:
    raise ValueError(f"Failed to read lora_config from {adapter_config_path}")

  if params is None:
    raise ValueError(f"Failed to read lora_config from {adapter_config_path}")

  config["adapter_path"] = adapter_weights_path
  return params, config


def apply_adapter(base_params, adapter_config, adapter_params):
  lora_rank = int(adapter_config["r"])
  lora_scale_factor = float(adapter_config["lora_alpha"]) / lora_rank
  lora_utils.apply_lora_on_base_params_nnx(base_params, adapter_params, lora_scale_factor)


def unapply_adapter(base_params, adapter_config, adapter_params):
  lora_rank = int(adapter_config["r"])
  lora_scale_factor = float(adapter_config["lora_alpha"]) / lora_rank
  lora_utils.unapply_lora_from_base_params_nnx(base_params, adapter_params, lora_scale_factor)
