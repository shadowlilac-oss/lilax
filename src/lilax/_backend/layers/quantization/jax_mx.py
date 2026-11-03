"""OCP MX quantization through `jax.nn.scaled_dot_general`."""

import functools
import math
from typing import ClassVar, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from flax import nnx

from lilax._backend.layers.quantization.base import Quantization


class JaxBlockScaledDotGeneral(nnx.Module):
  """Stateless OCP MX dot_general: `jax.nn.scaled_dot_general` with one config for both operands and one for the gradient.

  `mode` and `gradient_mode` are `jax.nn.get_scaled_dot_general_config` names, "mxfp8" or "mxfp4" (mxfp4 needs
  jax>=0.12.0); the matching MaxText `quantization` value is the same name with a `jax_` prefix.
  """

  def __init__(self, mode: str, gradient_mode: str | None = None):
    gradient_mode = gradient_mode or mode
    # Reject a mode this JAX lacks here, not at trace time.
    for name in (mode, gradient_mode):
      try:
        jax.nn.get_scaled_dot_general_config(name)
      except (AttributeError, ValueError) as e:
        raise ValueError(
            f"quantization=jax_{mode} needs a JAX with scaled_dot_general mode '{name}' (mxfp4: jax>=0.12.0)"
        ) from e
    self.mode = mode
    self.gradient_mode = gradient_mode

  def __call__(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> jax.Array:
    # jax.nn.scaled_dot_general takes no precision argument.
    del precision
    output_dtype = jnp.result_type(lhs, rhs) if preferred_element_type is None else preferred_element_type
    config = jax.nn.get_scaled_dot_general_config(self.mode)
    gradient_config = jax.nn.get_scaled_dot_general_config(self.gradient_mode)
    self._check_contracting_dims(lhs, rhs, dimension_numbers, config.block_size)
    return jax.nn.scaled_dot_general(
        lhs, rhs, dimension_numbers, preferred_element_type=output_dtype, configs=[config, config, gradient_config]
    )

  def _check_contracting_dims(
      self, lhs: jax.Array, rhs: jax.Array, dimension_numbers: jax.lax.DotDimensionNumbers, block: int
  ) -> None:
    """Checks that the contracting dims are a whole number of blocks.

    Three GEMMs are checked here:
    - forward, y = x @ W
    - backward, dx = g @ W^T
    - backward, dW = x^T @ g
    """
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_free = [d for d in range(lhs.ndim) if d not in lhs_contract and d not in lhs_batch]
    rhs_free = [d for d in range(rhs.ndim) if d not in rhs_contract and d not in rhs_batch]
    contracting_sizes = {
        "forward (y = x @ W)": math.prod(lhs.shape[d] for d in lhs_contract),
        "input gradient (dx = g @ W^T)": math.prod(rhs.shape[d] for d in rhs_free),
        "weight gradient (dW = x^T @ g)": math.prod(lhs.shape[d] for d in lhs_free),
    }
    for gemm, k in contracting_sizes.items():
      if k < block or k % block != 0:
        raise ValueError(
            f"quantization=jax_{self.mode}: the {gemm} GEMM contracts over {k} elements, which must be at least"
            f" {block} and a multiple of {block}; lhs {lhs.shape}, rhs {rhs.shape}, dimension_numbers"
            f" {dimension_numbers}."
        )


@dataclass
class JaxBlockScaledQuantization(Quantization):
  """Configures OCP MX quantization through `jax.nn.scaled_dot_general`."""

  # MX scales are the per-block amax of the data; no RNG is used at apply time.
  needs_apply_rngs: ClassVar[bool] = False

  quant_mode = "train"
  mode: str
  gradient_mode: str | None = None

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    del mesh_axes
    return functools.partial(JaxBlockScaledDotGeneral, mode=self.mode, gradient_mode=self.gradient_mode)

  def einsum(self, *args, **kwargs):
    del args, kwargs
    raise ValueError(f"quantization='jax_{self.mode}' does not support einsum.")
