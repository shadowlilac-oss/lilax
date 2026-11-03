"""Qwix providers that route intercepted ops to the GPU fp8 ops, keeping their state on the intercepted NNX module."""

import qwix
from qwix._src.utils import flax_util

import jax
import jax.numpy as jnp

from flax import nnx

from lilax._backend.layers.quantization.gpu_fp8 import Fp8DirectDotGeneralOp, Fp8DirectEinsum, NANOOFp8DotGeneralOp


def _apply_fp8_op_in_nnx(op_cls, op_id, *args, **kwargs):
  """Applies an fp8 op for a Qwix provider, keeping its state on the NNX module Qwix intercepts."""
  try:
    parent = flax_util.get_current_module()
  except ValueError:
    parent = None
  if not isinstance(parent, nnx.Module):
    raise ValueError(f"The fp8 op {op_id} needs an NNX module to hold its scales; got {type(parent).__name__}.")
  attr_name = f"_qwix_fp8_gpu_{op_id}"
  if not hasattr(parent, attr_name):
    # Fork the parent's rngs as the bridge did, so its later draws (e.g. dropout) stay the same.
    rngs = getattr(parent, "qwix_rngs", None)
    if rngs is None:
      rngs = getattr(parent, "rngs", None)
    if hasattr(rngs, "fork"):
      rngs.fork()
    setattr(parent, attr_name, op_cls())
  return getattr(parent, attr_name)(*args, **kwargs)


class NvidaFp8Provider(qwix.QtProvider):
  """Runs the fp8 ops of NVIDIA GPUs through Qwix's provider interface."""

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return _apply_fp8_op_in_nnx(Fp8DirectDotGeneralOp, op_id, *args, **kwargs)

  def einsum(self, *args, **kwargs):
    rule, op_id = self._get_current_rule_and_op_id("einsum")
    if rule is None:
      return jnp.einsum(*args, **kwargs)
    return _apply_fp8_op_in_nnx(Fp8DirectEinsum, op_id, *args, **kwargs)


class NANOOFp8Provider(qwix.QtProvider):

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return _apply_fp8_op_in_nnx(NANOOFp8DotGeneralOp, op_id, *args, **kwargs)
