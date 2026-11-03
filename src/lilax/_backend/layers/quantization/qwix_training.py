"""Qwix fp8_full dot_general and einsum (github.com/google/qwix) plumbed manually, for training only."""

import functools
from typing import ClassVar, Tuple, Callable
from dataclasses import dataclass

from qwix._src.core import dot_general_qt

import jax
import jax.numpy as jnp

from flax import nnx


@dataclass
class QwixQuantization:
  """Configures Qwix quantization github.com/google/qwix, for training only."""

  # Declared, not inherited: this class sits outside the Quantization hierarchy.
  needs_apply_rngs: ClassVar[bool] = True

  quant_mode = "train"  # needed by external call
  act_calibration_method: str = "absmax"
  weight_calibration_method: str = "absmax"
  bwd_calibration_method: str = "absmax"

  def _get_fp8_full_qwix_config(self) -> dot_general_qt.DotGeneralQtConfig:
    """Returns Qwix dot_general config for fp8_full quantization."""
    return dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,  # activation
        rhs_qtype=jnp.float8_e4m3fn,  # weight
        dlhs_grad_qtype=jnp.float8_e5m2,  # activation gradient
        drhs_grad_qtype=jnp.float8_e5m2,  # weight gradient
        lhs_calibration_method=self.act_calibration_method,
        rhs_calibration_method=self.weight_calibration_method,
        dlhs_grad_calibration_method=self.bwd_calibration_method,
        drhs_grad_calibration_method=self.bwd_calibration_method,
        tile_size=None,
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns Qwix dot_general."""
    return functools.partial(QwixDotGeneral, config=self._get_fp8_full_qwix_config())

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns Qwix einsum."""
    return QwixEinsum(config=self._get_fp8_full_qwix_config())


class QwixDotGeneral(nnx.Module):
  """A callable class for Qwix dot_general."""

  def __init__(self, config: dot_general_qt.DotGeneralQtConfig):
    self.config = config

  def __call__(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding=None,
  ) -> jax.Array:
    return dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, self.config)


class QwixEinsum(nnx.Module):
  """A callable class for Qwix einsum."""

  def __init__(self, config: dot_general_qt.DotGeneralQtConfig):
    self.config = config

  def __call__(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] | None = None,
      out_sharding=None,
  ) -> jax.Array:
    def custom_dot_general(*args, **kwargs):
      return dot_general_qt.dot_general_qt(*args[:3], self.config)  # pyrefly: ignore[bad-argument-count]

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=custom_dot_general,
          out_sharding=out_sharding,
      )
