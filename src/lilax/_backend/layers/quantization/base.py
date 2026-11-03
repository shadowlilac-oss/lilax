"""Base contract of quantization recipes."""

from typing import ClassVar, Tuple
from dataclasses import dataclass

import jax.numpy as jnp

from lilax._backend.common.common_types import DType


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  # Whether this backend's dot_general draws RNGs at apply time, as AQT and NVFP4
  # stochastic rounding do. False lets the Linen->NNX bridge drop its forked Rngs after
  # init; the default makes opting out deliberate.
  needs_apply_rngs: ClassVar[bool] = True

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""
