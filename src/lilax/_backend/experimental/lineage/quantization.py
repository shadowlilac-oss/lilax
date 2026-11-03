"""Per-tensor quantization configs and helpers (qwix `fp8_full`-style)."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from lilax._backend.experimental.lineage import ops, ragged_quantization


def _finfo_max(qtype: Any) -> float:
  return float(jnp.finfo(qtype).max)


def get_static_scale(calibration_method: str, qtype: Any) -> float:
  """Returns the scale of a `fixed,<min>,<max>` calibration for `qtype`."""
  name, *bounds = calibration_method.split(",")
  if name != "fixed" or len(bounds) != 2:
    raise ValueError("Expected a 'fixed,<min>,<max>' calibration method, got" f" {calibration_method!r}.")
  return max(abs(float(b)) for b in bounds) / _finfo_max(qtype)


@dataclasses.dataclass(frozen=True)
class GmmQuantRule:
  """Per-tensor quantization of GMMs and their backward passes."""

  weight_qtype: Any = jnp.float8_e4m3fn
  act_qtype: Any = jnp.float8_e4m3fn
  bwd_qtype: Any = jnp.float8_e5m2
  grad_dtype: Any = jnp.bfloat16
  weight_calibration_method: str = "fixed,-224,224"
  act_calibration_method: str = "fixed,-224,224"
  bwd_calibration_method: str = "absmax"

  def __post_init__(self):
    if self.bwd_calibration_method != "absmax":
      raise ValueError("Only 'absmax' gradient calibration is supported, got" f" {self.bwd_calibration_method!r}.")
    _ = self.weight_scale, self.act_scale

  @property
  def weight_scale(self) -> float:
    return get_static_scale(self.weight_calibration_method, self.weight_qtype)

  @property
  def act_scale(self) -> float:
    return get_static_scale(self.act_calibration_method, self.act_qtype)


@dataclasses.dataclass(frozen=True)
class MlaQuantRule(GmmQuantRule):
  """Quantization of MLA projections and splash attention options."""

  splash_qk_qtype: Any | None = None
  q_proj_bwd_quantized: bool = True
  q_prescaled_log2e: bool = False


@dataclasses.dataclass(frozen=True)
class QuantConfig:
  """Model-wide quantization config; None fields stay unquantized."""

  routed_experts: GmmQuantRule | None = None
  mla: GmmQuantRule | None = None
  combine_qtype: Any | None = None
  combine_per_hop_scaling: bool = False

  def __post_init__(self):
    if self.combine_qtype is not None and self.combine_qtype != jnp.float8_e4m3fn:
      raise ValueError("combine_qtype must be None or jnp.float8_e4m3fn, got" f" {self.combine_qtype!r}.")
    if self.combine_per_hop_scaling and self.combine_qtype is None:
      raise ValueError("combine_per_hop_scaling requires combine_qtype.")


FP8_FULL = QuantConfig(routed_experts=GmmQuantRule(), mla=GmmQuantRule())
FP8_FULL_SPLASH_QK = QuantConfig(routed_experts=GmmQuantRule(), mla=MlaQuantRule(splash_qk_qtype=jnp.float8_e4m3fn))
FP8_FULL_SPLASH_QK_E3 = QuantConfig(
    routed_experts=GmmQuantRule(), mla=MlaQuantRule(splash_qk_qtype=jnp.float8_e4m3fn, q_prescaled_log2e=True)
)
FP8_FULL_COMBINE = dataclasses.replace(FP8_FULL, combine_qtype=jnp.float8_e4m3fn)
FP8_FULL_COMBINE_PER_HOP = dataclasses.replace(FP8_FULL_COMBINE, combine_per_hop_scaling=True)
FP8_FULL_SPLASH_QK_E3_COMBINE_PER_HOP = dataclasses.replace(
    FP8_FULL_SPLASH_QK_E3, combine_qtype=jnp.float8_e4m3fn, combine_per_hop_scaling=True
)


def static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  """Returns clip(x / scale) cast to `qtype`."""
  fmax = _finfo_max(qtype)
  return jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)


def absmax_to_scale(amax: jax.Array, qtype: Any) -> jax.Array:
  """Returns the `[1, 1]` f32 quantization scale for `amax` and `qtype`."""
  fmax = _finfo_max(qtype)
  amax_f32 = amax.astype(jnp.float32)
  scale = jnp.where(amax_f32 > 0, amax_f32 / fmax, 1.0)
  return scale.reshape(1, 1)


def _xla_quantize_with_scale(x: jax.Array, scale: jax.Array, qtype: Any) -> jax.Array:
  fmax = _finfo_max(qtype)
  inv_scale = 1.0 / scale.astype(jnp.float32)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(qtype)


def _can_use_ragged_pallas(x: jax.Array) -> bool:
  return (
      x.ndim in (2, 3)
      and x.shape[0] >= 64
      and x.shape[0] % 64 == 0
      and ((x.ndim == 2 and x.shape[1] % 128 == 0) or (x.ndim == 3 and x.shape[1] % 8 == 0 and x.shape[2] == 128))
  )


def quantize_with_scale(
    x: jax.Array, scale: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None
) -> jax.Array:
  """Returns `clip(x / scale)` cast to `qtype` using `[1, 1]` f32 `scale`."""
  if num_valid_rows is not None and _can_use_ragged_pallas(x):
    return jax.lax.platform_dependent(
        x, scale, jnp.asarray(num_valid_rows, jnp.int32),
        tpu=lambda a, s, n: ragged_quantization.ragged_quantize(a, s, qtype, n),
        default=lambda a, s, _: _xla_quantize_with_scale(a, s, qtype),
    )
  return _xla_quantize_with_scale(x, scale, qtype)


def ragged_absmax(x: jax.Array, num_valid_rows: jax.Array | int | None = None) -> jax.Array:
  """Returns the `[1, 1]` f32 absmax of the first `num_valid_rows` of `x`."""
  abs_x = jnp.abs(x.astype(jnp.float32))
  if num_valid_rows is not None:
    rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    abs_x = jnp.where(rows < num_valid_rows, abs_x, 0.0)
  return jnp.max(abs_x).reshape(1, 1)


def _xla_absmax_quantize(
    x: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None
) -> tuple[jax.Array, jax.Array]:
  scale = absmax_to_scale(ragged_absmax(x, num_valid_rows), qtype)
  xq = _xla_quantize_with_scale(x, scale, qtype)
  return xq, scale


def absmax_quantize(
    x: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None
) -> tuple[jax.Array, jax.Array]:
  """Dynamically quantizes x with one scale from its first `num_valid_rows`."""
  if num_valid_rows is not None and _can_use_ragged_pallas(x):
    return jax.lax.platform_dependent(
        x, jnp.asarray(num_valid_rows, jnp.int32),
        tpu=lambda a, n: ragged_quantization.ragged_absmax_quantize(a, qtype, n),
        default=lambda a, n: _xla_absmax_quantize(a, qtype, n),
    )
  return _xla_absmax_quantize(x, qtype, num_valid_rows)
