"""JetStream result layout for sampled tokens."""

from jetstream.engine import engine_api
import jax.numpy as jnp


def result_tokens(tokens, all_valid, generated_tokens, log_prob, samples_per_slot):
  """Pack tokens, validity, and lengths into adjacent JetStream result columns."""
  return engine_api.ResultTokens(
      data=jnp.concatenate((tokens, all_valid, generated_tokens), axis=1),
      tokens_idx=(0, 1),
      valid_idx=(1, 2),
      length_idx=(2, 3),
      log_prob=log_prob,
      samples_per_slot=samples_per_slot,
  )
