"""Building and applying a recipe's quantized dot_general and einsum inside NNX layers."""

import functools
from typing import Tuple, Sequence

import flax.linen as nn
from flax import nnx
from lilax._backend.layers import nnx_wrappers

from lilax._backend.common.common_types import DType
from lilax._backend.layers.quantization.base import Quantization


def create_fp8_einsum(quant: Quantization, dtype: DType, rngs: nnx.Rngs) -> nnx.Module:
  """Creates the fp8 einsum an MoE layer registers in `quant_einsums`.

  Its scales and amax histories live in the einsum, so it is built with the layer rather than
  on the first call, which would grow the module graph inside the scanned layer loop.

  Args:
    quant: The fp8 quantization providing the einsum.
    dtype: The computation dtype of the einsum.
    rngs: The `nnx.Rngs` of the parent module, forked as in `make_dot_general`.
  """
  rngs.fork()
  return quant.einsum(dtype=dtype)  # pytype: disable=attribute-error


def make_dot_general(quant, kernel_axes: Tuple[str, ...], rngs: nnx.Rngs) -> tuple[str, nnx.Module]:
  """Builds the quantized dot_general of an NNX layer and the attribute name the layer stores it under.

  The name follows the quantizer's class, e.g. "Fp8DirectDotGeneralOp_0", and checkpoints use it.
  Linen quantizers (AQT, Transformer Engine) are bridged into NNX and create their state on the
  first call; the others are NNX modules and are used as they are.
  """
  dot_general = quant.dot_general_cls(mesh_axes=kernel_axes)()
  name = f"{type(dot_general).__name__}_0"
  if isinstance(dot_general, nn.Module):
    dot_general = nnx_wrappers.ToNNX(dot_general, rngs=rngs)
  else:
    # The bridge forks rngs; do the same so the weights created after this one draw the same values.
    rngs.fork()
  return name, dot_general


def apply_einsum_in_nnx(parent: nnx.Module, op_id: str, einsum, mutable: Sequence[str], *args):
  """Applies a quantized Linen einsum from within an NNX parent module.

  A Linen module cannot be called unbound, which is all an `nnx.Module` can offer it, so the
  einsum is bridged into NNX on first use and the bridged instance is reused afterwards.
  `op_id` must be unique per call site: two call sites sharing an id would also share
  quantization state. Bridging here rather than while building the parent needs the operands,
  so it only suits einsums whose state is shaped after them, such as AQT.

  Args:
    parent: The NNX module hosting the einsum; it must carry an `rngs` attribute.
    op_id: Stable identifier for the call site.
    einsum: The einsum returned by a `Quantization`, either a Linen module or a callable.
    mutable: Variable collections the einsum writes to.
    *args: Arguments forwarded to the einsum.
  """
  linen_einsum = einsum.func if isinstance(einsum, functools.partial) else einsum
  if not isinstance(linen_einsum, nn.Module):
    return einsum(*args)

  # The name must not start with an underscore: NNX treats such attributes as static,
  # which cannot hold the bridged module's variables.
  attr_name = f"quant_einsum_{op_id}"
  wrapper = getattr(parent, attr_name, None)
  if wrapper is None:
    wrapper = nnx_wrappers.ToNNX(linen_einsum, rngs=parent.rngs)
    wrapper.lazy_init(*args)
    setattr(parent, attr_name, wrapper)
  return wrapper(*args, mutable=list(mutable))
