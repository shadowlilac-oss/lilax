"""YaRN rotary embedding for the DeepSeek BF16 batch-split schedule."""

import math

import jax
import jax.numpy as jnp


def initialize_yarn_mask(embedding_dims):
  """Initializes YaRN mask."""
  indices = jnp.arange(embedding_dims)
  # [1, 0, 3, 2, 5, 4, ...]
  swap_indices = jnp.where(indices % 2 == 0, indices + 1, indices - 1)
  negation_mask = jnp.where(indices % 2 == 0, -1, 1)
  identity = jnp.eye(embedding_dims, dtype=jnp.int32)
  return identity[swap_indices] * negation_mask


def initialize_yarn_freqs(
    positions,
    embedding_dims,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mesh,
    activation_pspec,
):
  """Initializes YaRN frequencies."""
  half_dim = embedding_dims // 2
  # Compute base frequencies for each (even-indexed) dimension.
  # (Note: We use jnp.arange with float32 for precision.)
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / embedding_dims))

  low = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  high = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  low = max(math.floor(low), 0)
  high = min(math.ceil(high), embedding_dims - 1)
  diff = high - low if high > low else 0.001
  linear_func = (jnp.arange(half_dim, dtype=jnp.float32) - low) / diff
  smooth = 1 - jnp.clip(linear_func, 0, 1)
  # The corrected frequency is a weighted mix of the scaled and base values.
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

  # Precompute frequencies for all positions by taking the outer product.
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
  # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
  freqs = jnp.outer(t, freqs)
  # Lookup the precomputed frequencies using the position indices.
  # self.freqs has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
  # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
  freqs = freqs.at[positions].get(out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec))
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]
  return jnp.repeat(freqs, 2, axis=-1)  # shape: [B, S, 1, embedding_dims]


def yarn(
    inputs,
    freqs,
    *,
    pairwise_swap_and_negate_mask,
    fprop_dtype,
):
  """Performs YaRN rotary embedding."""
  # inputs @ mask: [B, S, N, embedding_dims] @ [embedding_dims, embedding_dims] -> [B, S, N, embedding_dims]
  output = inputs * jnp.cos(freqs) + jnp.matmul(inputs, pairwise_swap_and_negate_mask) * jnp.sin(freqs)
  return output.astype(fprop_dtype)
