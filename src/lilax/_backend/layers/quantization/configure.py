"""Selecting the quantization recipe and KV cache quantization from the user configuration."""

import json

from lilax._backend.configs.types import JAX_MX_QUANTIZATIONS
from lilax._backend.common.common_types import Config
from lilax._backend.inference.kvcache import KVQuant
from lilax._backend.layers.quantization.aqt import AqtQuantization
from lilax._backend.layers.quantization.aqt_configs import (
    _get_aqt_fp8_default_config,
    _get_aqt_fp8_quant_config,
    _get_int8_quant_config,
    _get_mixed_precision_quant_config,
)
from lilax._backend.layers.quantization.gpu_fp8 import Fp8Quantization, NANOOFp8Quantization
from lilax._backend.layers.quantization.jax_mx import JaxBlockScaledQuantization
from lilax._backend.layers.quantization.modes import get_quant_mode
from lilax._backend.layers.quantization.qwix_training import QwixQuantization
from lilax._backend.layers.quantization.serve_fp8 import ServeFp8WeightQuantization
from lilax._backend.layers.quantization.transformer_engine import TransformerEngineQuantization


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization in JAX_MX_QUANTIZATIONS:
    return config.quantization
  if config.quantization == "int8":
    return _get_int8_quant_config(config)
  if config.quantization == "intmp":
    assert config.quant_cfg_path, "Must specify quant_cfg for mixed precision quantization"
    with open(config.quant_cfg_path, "rt", encoding="utf8") as config_file:
      mixed_precision_config = json.load(config_file)
    return _get_mixed_precision_quant_config(mixed_precision_config)
  if config.quantization == "fp8":
    return "fp8"
  if config.quantization == "nanoo_fp8":
    return "nanoo_fp8"
  if config.quantization == "aqt_fp8":
    return _get_aqt_fp8_quant_config(config)
  if config.quantization == "aqt_fp8_full":
    return _get_aqt_fp8_default_config(config)
  if config.quantization.startswith("te_"):
    return config.quantization

  raise ValueError(f"Invalid value configured for quantization {config.quantization}.")


def configure_quantization(config: Config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  if config.quantization == "serve_fp8_weight":
    return ServeFp8WeightQuantization(act_calibration_method=config.act_quantization_calibration_method)

  if getattr(config, "use_batch_split_schedule", False) and config.quantization:
    # The older version of batch-split that fully uses qwix quantization.
    if config.quantization == "fp8_full" and not config.use_manual_quantization:
      return QwixQuantization(
          weight_calibration_method=config.weight_quantization_calibration_method,
          act_calibration_method=config.act_quantization_calibration_method,
          bwd_calibration_method=config.bwd_quantization_calibration_method,
      )
    # The pure JAX version of batch-split that uses manual quantization for dot general.
    return None

  if config.quantization and config.use_qwix_quantization:
    return None
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg in JAX_MX_QUANTIZATIONS:
      mode = quant_cfg.removeprefix("jax_")
      if quant_mode_str != "train":
        raise ValueError(f"quantization='jax_{mode}' supports quant_mode 'train' only, got '{quant_mode_str}'.")
      return JaxBlockScaledQuantization(mode=mode, gradient_mode=config.jax_mx_gradient_mode or None)
    elif quant_cfg == "fp8":
      return Fp8Quantization()
    elif quant_cfg == "nanoo_fp8":
      return NANOOFp8Quantization()
    elif isinstance(quant_cfg, str) and quant_cfg.startswith("te_"):
      return TransformerEngineQuantization(config)
    quant_mode = get_quant_mode(quant_mode_str)
    replicate_scale = config.replicate_quant_scale if config.replicate_quant_scale else False
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode, replicate_scale=replicate_scale)
  return None


def configure_kv_quant(config):
  return None if not config.quantize_kvcache else KVQuant(config)
