"""Optionally quantized dot product for the DeepSeek FP8 batch-split schedule."""

import jax.numpy as jnp


def dot(x, y, quant=None, axes=1):
  """Computes the dot product of two arrays, optionally using quantization."""
  if quant is not None:
    # Convert axes to jax.lax.dot_general dimension_numbers
    if isinstance(axes, int):
      x_contract = tuple(range(x.ndim - axes, x.ndim))
      y_contract = tuple(range(axes))
    else:
      x_contract, y_contract = axes
    dimension_numbers = ((x_contract, y_contract), ((), ()))
    # Instantiate and call qwix dot_general
    custom_dot = quant.dot_general_cls()()
    return custom_dot(lhs=x, rhs=y, dimension_numbers=dimension_numbers)

  # Unquantized
  return jnp.tensordot(x, y, axes=axes)
