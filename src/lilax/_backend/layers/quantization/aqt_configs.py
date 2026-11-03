"""AQT dot_general configurations: int8, fp8 and per-layer mixed precision recipes."""

import functools
from dataclasses import dataclass

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import calibration

import jax.numpy as jnp

# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel


def _get_int8_quant_config(config):
  """Get int8 quantization configuration."""
  drhs_bits = None
  drhs_accumulator_dtype = None
  drhs_local_aqt = None
  if config.quantization_local_shard_count != 0:
    drhs_bits = 8
    drhs_accumulator_dtype = jnp.int32
    drhs_local_aqt = aqt_config.LocalAqt(
        contraction_axis_shard_count=config.quantization_local_shard_count  # pyrefly: ignore[unexpected-keyword]
    )  # pyrefly: ignore[unexpected-keyword]
  return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type="jax.uniform",
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
  )


@dataclass(frozen=True)
class ConstantBoundConfig:
  fwd_lhs_bound: float | None = None
  fwd_rhs_bound: float | None = None
  dlhs_lhs_bound: float | None = None
  dlhs_rhs_bound: float | None = None
  drhs_lhs_bound: float | None = None
  drhs_rhs_bound: float | None = None


def _build_const_scale_config(
    aqt_dg: aqt_config.DotGeneral,
    cst_bound_config: ConstantBoundConfig,
) -> aqt_config.DotGeneral:
  """Build a constant scale config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    cst_bound_config: The constant bound config.

  Returns:
    The AQT dot general config with constant scale config.
  """
  if cst_bound_config.fwd_lhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_lhs_bound
    )
  if cst_bound_config.fwd_rhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_rhs_bound
    )
  if cst_bound_config.dlhs_lhs_bound is not None:
    aqt_dg.dlhs.dg_quantizer.lhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_lhs_bound
    )

  if cst_bound_config.dlhs_rhs_bound is not None:
    aqt_dg.dlhs.dg_quantizer.rhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_rhs_bound
    )

  if cst_bound_config.drhs_lhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.lhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_lhs_bound
    )

  if cst_bound_config.drhs_rhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.rhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_rhs_bound
    )

  return aqt_dg


@dataclass
class PerTensorScales:
  fwd_lhs: bool = False
  fwd_rhs: bool = False
  dlhs_lhs: bool = False
  dlhs_rhs: bool = False
  drhs_lhs: bool = False
  drhs_rhs: bool = False


def _build_per_tensor_config(
    aqt_dg: aqt_config.DotGeneral,
    per_tensor_scales: PerTensorScales,
) -> aqt_config.DotGeneral:
  """Build a per tensor config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    per_tensor_scales: The per tensor scales config.

  Returns:
    The AQT dot general config with per tensor config.
  """
  if per_tensor_scales.fwd_lhs:
    aqt_dg.fwd.dg_quantizer.lhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  if per_tensor_scales.fwd_rhs:
    aqt_dg.fwd.dg_quantizer.rhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  if per_tensor_scales.dlhs_lhs:
    aqt_dg.dlhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  if per_tensor_scales.dlhs_rhs:
    aqt_dg.dlhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  if per_tensor_scales.drhs_lhs:
    aqt_dg.drhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  if per_tensor_scales.drhs_rhs:
    aqt_dg.drhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"  # pyrefly: ignore[missing-attribute]
  return aqt_dg


# fp8 training recipe of dynamic scaling with configurable constant_bound_config for static scaling option
def _get_aqt_fp8_default_config(config):
  """Get aqt for 8-bit floating point quantization configuration."""
  aqt_dg = aqt_config.config_v4(
      fwd_bits="e4m3",
      dlhs_bits="e5m2",
      drhs_bits="e5m2",
      use_dummy_static_bound=False,
      fwd_accumulator_dtype=jnp.bfloat16,
      dlhs_accumulator_dtype=jnp.bfloat16,
      drhs_accumulator_dtype=jnp.bfloat16,
      dlhs_use_fwd_quant=False,
      drhs_use_fwd_quant=False,
  )
  constant_bound_config = None

  if len(config.constant_bound_config) == 6:
    (
        fwd_lhs_bound,
        fwd_rhs_bound,
        dlhs_lhs_bound,
        dlhs_rhs_bound,
        drhs_lhs_bound,
        drhs_rhs_bound,
    ) = config.constant_bound_config
    constant_bound_config = ConstantBoundConfig(
        fwd_lhs_bound=fwd_lhs_bound,
        fwd_rhs_bound=fwd_rhs_bound,
        dlhs_lhs_bound=dlhs_lhs_bound,
        dlhs_rhs_bound=dlhs_rhs_bound,
        drhs_lhs_bound=drhs_lhs_bound,
        drhs_rhs_bound=drhs_rhs_bound,
    )
    aqt_dg = _build_const_scale_config(aqt_dg, constant_bound_config)

  aqt_config.set_stochastic_rounding(
      aqt_dg,
      vjp_lhs_stochastic_rounding=False,
      vjp_rhs_stochastic_rounding=False,
      implementation="jax.uniform",
  )

  per_tensor_scales = PerTensorScales(
      fwd_lhs=True,
      fwd_rhs=True,
      dlhs_lhs=True,
      dlhs_rhs=True,
      drhs_lhs=True,
      drhs_rhs=True,
  )
  return _build_per_tensor_config(aqt_dg, per_tensor_scales)


def _get_aqt_fp8_quant_config(config):
  """get aqt for 8-bit floating point quantization configuration"""
  cfg = aqt_config.config_v4(fwd_bits="e4m3", dlhs_bits=None, drhs_bits=None, fwd_accumulator_dtype=jnp.bfloat16)
  return cfg


def _dot_general_make(quant_cfg):
  """Create quantization configs for input matrices to a matmul"""
  lhs_bits = quant_cfg[_A_BITS]
  lhs_scale = quant_cfg[_A_SCALE]
  rhs_bits = quant_cfg[_W_BITS]
  rhs_scale = quant_cfg[_W_SCALE]
  aqt_dg = aqt_config.dot_general_make(lhs_bits=lhs_bits, rhs_bits=rhs_bits)
  if lhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.AbsMaxCalibration, scale=lhs_scale
    )  # pyrefly: ignore[missing-attribute]
  if rhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(  # pyrefly: ignore[missing-attribute]
        calibration.AbsMaxCalibration, scale=rhs_scale
    )  # pyrefly: ignore[missing-attribute]
  return aqt_dg


def _get_default_mp_config(default=None):
  default_config = {_W_BITS: None, _A_BITS: None, _W_SCALE: 1.0, _A_SCALE: 1.0, _TILE_SIZE: -1}
  if default:
    default_config.update(default)
  return default_config


def _get_mixed_precision_quant_config(mixed_precision_config):
  """Set quantization params based on user configuration."""
  ret_config = {}
  default_mp_config = _get_default_mp_config(default=mixed_precision_config.get(DEFAULT, None))
  for layer_name_re, layer_quantization_config in mixed_precision_config.items():
    # Make a copy of default_mp_config to avoid updating original dict
    quant_config = default_mp_config.copy()
    # print(f"Mixed precision config: processing
    # {layer_name_re} - {layer_quantization_config}, default config - {quant_config}")
    if layer_name_re != DEFAULT:
      for k in quant_config:
        quant_config[k] = layer_quantization_config.get(k, default_mp_config[k])
    ret_config[layer_name_re] = [_dot_general_make(quant_config), quant_config["tile_size"]]
  return ret_config
