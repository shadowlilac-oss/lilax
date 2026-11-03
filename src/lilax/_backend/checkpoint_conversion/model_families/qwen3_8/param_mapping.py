"""Qwen3.8 text-only parameter mappings and transformation hooks for checkpoint conversion."""

from lilax._backend.checkpoint_conversion.model_families.qwen3_5.param_mapping import (
    QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING,
)


def _qwen3_8_wrap_config(config):
  if "text_config" in config:
    return config
  return {"text_config": config}


def QWEN3_8_MAXTEXT_TO_HF_PARAM_MAPPING(config, maxtext_config, scan_layers=False):
  return QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(
      _qwen3_8_wrap_config(config), maxtext_config, scan_layers=scan_layers, hf_prefix="model")


def QWEN3_8_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, maxtext_config, scan_layers=False, saving_to_hf=False):
  return QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      _qwen3_8_wrap_config(config), maxtext_config, scan_layers, saving_to_hf)
