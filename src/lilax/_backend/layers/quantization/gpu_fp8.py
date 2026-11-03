"""Delayed-scaling fp8 recipes and ops for NVIDIA GPUs and AMD MI300/MI325 (NANOO), built on Flax's fp8 ops."""

import functools
from typing import ClassVar, Tuple
from dataclasses import dataclass

import jax.numpy as jnp

from flax.linen import fp8_ops
from flax import nnx
from flax.nnx import variablelib

from lilax._backend.common.common_types import DType
from lilax._backend.layers.quantization.base import Quantization


@dataclass
class Fp8Quantization(Quantization):
  """Configures Fp8 quantization for NVIDIA GPUs"""

  # Flax's fp8 ops scale from amax history, never from make_rng at apply time.
  needs_apply_rngs: ClassVar[bool] = False

  quant_mode = "train"
  # The forward dtype, for callers that quantize an operand themselves rather than through
  # `dot_general_cls` or `einsum`, such as the grouped matmul in MoE.
  quantize_dtype = jnp.float8_e4m3fn

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return Fp8DirectDotGeneralOp

  def einsum(self, dtype: DType = jnp.float32):
    return _Fp8EinsumWrapper(dtype=dtype)


_OVERWRITE_WITH_GRADIENT = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)


def _fp8_dot_inputs(*args, **kwargs):
  """Splits dot_general arguments the way Flax's fp8 ops do: the kernel's dtype is the computation dtype."""
  del kwargs
  assert len(args) == 3
  x, k, dimension_numbers = args
  return jnp.asarray(x, k.dtype), k, dimension_numbers, k.dtype


class _Fp8Stats(nnx.Module):
  """The scales and amax histories of an fp8 op, kept the way Flax's `Fp8DotGeneralBase` keeps them.

  They are `_overwrite_with_gradient` variables: the fp8 ops return their new values as gradients,
  and the train step writes those back instead of applying the optimizer.
  """

  def __init__(self, amax_history_length: int = 1024):
    self.input_amax_history = _OVERWRITE_WITH_GRADIENT(jnp.zeros((amax_history_length,), jnp.float32))
    self.kernel_amax_history = _OVERWRITE_WITH_GRADIENT(jnp.zeros((amax_history_length,), jnp.float32))
    self.output_grad_amax_history = _OVERWRITE_WITH_GRADIENT(jnp.zeros((amax_history_length,), jnp.float32))
    self.input_scale = _OVERWRITE_WITH_GRADIENT(jnp.ones((1,), jnp.float32))
    self.kernel_scale = _OVERWRITE_WITH_GRADIENT(jnp.ones((1,), jnp.float32))
    self.output_grad_scale = _OVERWRITE_WITH_GRADIENT(jnp.ones((1,), jnp.float32))

  def scaled_dot_general(self, compute_dtype: DType):
    """Flax's `fp8_scaled_dot_general` bound to these scales and amax histories."""
    return functools.partial(
        fp8_ops.fp8_scaled_dot_general,
        lhs_scale=self.input_scale[...],
        rhs_scale=self.kernel_scale[...],
        grad_scale=self.output_grad_scale[...],
        lhs_amax_history=self.input_amax_history[...],
        rhs_amax_history=self.kernel_amax_history[...],
        grad_amax_history=self.output_grad_amax_history[...],
        quantize_compute_type=compute_dtype,
    )

  def quantize_inputs(self, compute_dtype: DType, e4m3_dtype: DType, x, k):
    x_qdq = fp8_ops.in_qdq(compute_dtype, e4m3_dtype, x, self.input_scale[...], self.input_amax_history[...])
    k_qdq = fp8_ops.in_qdq(compute_dtype, e4m3_dtype, k, self.kernel_scale[...], self.kernel_amax_history[...])
    return x_qdq, k_qdq

  def quantize_output_grad(self, compute_dtype: DType, e5m2_dtype: DType, y):
    return fp8_ops.out_qdq(compute_dtype, e5m2_dtype, y, self.output_grad_scale[...], self.output_grad_amax_history[...])


class Fp8DirectDotGeneralOp(_Fp8Stats):
  """Flax's `Fp8DirectDotGeneralOp` as an NNX module.

  It keeps Flax's class name: layers store it as "<class name>_0", and checkpoints use that path.
  """

  def __call__(self, *args, **kwargs):
    x, k, dimension_numbers, compute_dtype = _fp8_dot_inputs(*args, **kwargs)
    return self.scaled_dot_general(compute_dtype)(x, k, dimension_numbers, precision=None, preferred_element_type=x.dtype)


class NANOOFp8DotGeneralOp(_Fp8Stats):
  """Flax's `NANOOFp8DotGeneralOp` as an NNX module, using the fnuz fp8 formats of AMD MI300/MI325."""

  def __call__(self, *args, **kwargs):
    x, k, dimension_numbers, compute_dtype = _fp8_dot_inputs(*args, **kwargs)
    x_qdq, k_qdq = self.quantize_inputs(compute_dtype, jnp.float8_e4m3fnuz, x, k)
    y_qdq = fp8_ops.dot_general_with_precision(x_qdq, k_qdq, dimension_numbers)
    return self.quantize_output_grad(compute_dtype, jnp.float8_e5m2fnuz, y_qdq)


class Fp8DirectEinsum(_Fp8Stats):
  """Flax's `Fp8Einsum` as an NNX module. The rhs is the weight and sets the computation dtype."""

  def __call__(self, eqn, lhs, rhs, precision=None, preferred_element_type=None):
    compute_dtype = rhs.dtype
    return jnp.einsum(
        eqn,
        lhs.astype(compute_dtype),
        rhs,
        precision=precision,
        preferred_element_type=preferred_element_type,
        _dot_general=self.scaled_dot_general(compute_dtype),
    )


class _Fp8EinsumWrapper(nnx.Module):
  """Runs `Fp8DirectEinsum` in a given computation dtype."""

  def __init__(self, dtype: DType):
    self.dtype = dtype
    self.fp8_einsum = Fp8DirectEinsum()

  def __call__(self, eqn, lhs, rhs, **kwargs):
    # The fp8 einsum takes its computation dtype from rhs and casts lhs to it.
    return self.fp8_einsum(eqn, lhs, rhs.astype(self.dtype), **kwargs)


class Fp8Einsum(_Fp8Stats):
  """An fp8 einsum op."""

  def __init__(
      self,
      amax_history_length: int = 1024,
      e4m3_dtype: DType = jnp.float8_e4m3fn,
      e5m2_dtype: DType = jnp.float8_e5m2,
      dtype: DType = jnp.float32,
  ):
    super().__init__(amax_history_length)
    self.e4m3_dtype = e4m3_dtype
    self.e5m2_dtype = e5m2_dtype
    self.dtype = dtype

  def __call__(self, eqn, *args, **kwargs):
    assert len(args) == 2
    comp_dtype = self.dtype
    x = jnp.asarray(args[0], comp_dtype)
    k = jnp.asarray(args[1], comp_dtype)
    x_qdq, k_qdq = self.quantize_inputs(comp_dtype, self.e4m3_dtype, x, k)
    y_qdq = jnp.einsum(eqn, x_qdq, k_qdq, _dot_general=fp8_ops.dot_general_with_precision)
    return self.quantize_output_grad(comp_dtype, self.e5m2_dtype, y_qdq)


@dataclass
class NANOOFp8Quantization(Quantization):
  """Configures NANOO Fp8 quantization for AMD MI300/MI325 GPUs"""

  # Same as Fp8Quantization: the NANOO fp8 op never draws at apply time.
  needs_apply_rngs: ClassVar[bool] = False

  quant_mode = "train"
  quantize_dtype = jnp.float8_e4m3fnuz

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return NANOOFp8DotGeneralOp

  def einsum(self, dtype: DType = jnp.float32):
    """Returns an einsum using the NANOO (fnuz) fp8 formats of AMD MI300/MI325."""
    return Fp8Einsum(dtype=dtype, e4m3_dtype=jnp.float8_e4m3fnuz, e5m2_dtype=jnp.float8_e5m2fnuz)
