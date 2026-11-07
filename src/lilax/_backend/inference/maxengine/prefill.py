"""Prefill input shaping and final-position logit selection."""

import jax
import jax.numpy as jnp

from lilax._backend.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR


def prefill_inputs(existing_prefix, padded_tokens, positions, true_length):
  """Batch one prompt and place it after any existing prefix."""
  start_position = 0
  previous_chunk = None
  if existing_prefix is not None:
    start_position = existing_prefix.common_prefix_tokens.shape[0]
    previous_chunk = jnp.expand_dims(existing_prefix.common_prefix_tokens, 0)

  full_true_length = start_position + true_length

  input_tokens = jnp.expand_dims(padded_tokens, 0)

  if positions is not None:
    if positions.ndim in (1, 2):
      positions = jnp.expand_dims(positions, 0)
  else:
    positions = jnp.expand_dims(jnp.arange(start_position, start_position + input_tokens.shape[1]), 0)
  return start_position, previous_chunk, full_true_length, input_tokens, positions


def active_sequence_indicator(start_position, length, true_length):
  """Mark the prompt positions before `true_length` as active decoder segments."""
  start_to_n = jnp.arange(start_position, start_position + length)
  ones_to_keep = start_to_n < true_length
  one_d_output = ones_to_keep * DECODING_ACTIVE_SEQUENCE_INDICATOR
  return jnp.expand_dims(one_d_output, 0)


def select_logits(flat_logits, index):
  """Select the logits at one sequence position."""
  return jax.lax.dynamic_slice(
      flat_logits,
      (0, index, 0),
      (flat_logits.shape[0], 1, flat_logits.shape[2]),
  )
