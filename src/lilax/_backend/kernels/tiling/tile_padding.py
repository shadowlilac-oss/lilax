"""Padding of kernel inputs to tile-size multiples."""

import jax
import jax.numpy as jnp


def maybe_pad(inputs, tile_size):
  """Pads the inputs leading dimension to be divisible by tile_size."""
  inputs_dim = inputs.shape[0]
  padding_amount = 0
  if inputs_dim % tile_size:
    padding_amount = tile_size - inputs_dim % tile_size
    inputs = jax.lax.pad(
        inputs,
        jnp.array(0.0, dtype=inputs.dtype),
        [(0, padding_amount, 0)] + [(0, 0, 0)] * (inputs.ndim - 1),
    )
  return inputs, padding_amount
