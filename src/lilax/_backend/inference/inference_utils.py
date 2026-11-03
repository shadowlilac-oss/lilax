"""Inference sampling and prompt log-probability utilities."""

import jax
import jax.numpy as jnp

NEG_INF = -1.0e7


def str2bool(value: str) -> bool:
  value = value.lower()
  if value in ["y", "yes", "t", "true", "1"]:
    return True
  elif value in ["n", "no", "f", "false", "0"]:
    return False
  else:
    raise ValueError(f"Invalid value '{value}'!")


@jax.jit
def prompt_logprobs_from_packed_prefill(logits, input_tokens, decoder_positions, decoder_segment_ids, true_lengths):
  batch, _, _ = logits.shape
  logps = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
  targets = input_tokens[:, 1:]
  scored = jnp.take_along_axis(logps, targets[..., None], axis=-1)[..., 0]
  pad = jnp.full((batch, 1), jnp.nan, dtype=logits.dtype)
  shifted = jnp.concatenate([pad, scored], axis=1)
  lengths = jnp.take(true_lengths, decoder_segment_ids, mode="clip")
  valid = (decoder_positions > 0) & (decoder_positions < lengths)
  return jnp.where(valid, shifted, jnp.nan)


@jax.jit
def prompt_logprobs_from_prefill(logits, input_tokens, true_length):
  batch, sequence = input_tokens.shape
  logps = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
  targets = input_tokens[:, 1:]
  scored = jnp.take_along_axis(logps, targets[..., None], -1)[..., 0]
  pad = jnp.full((batch, 1), jnp.nan, dtype=logps.dtype)
  result = jnp.concatenate([pad, scored], axis=1)
  lengths = jnp.asarray(true_length)
  lengths = jnp.broadcast_to(lengths, (batch,)) if lengths.ndim == 0 else lengths
  positions = jnp.arange(sequence)[None, :]
  valid = (positions < lengths[:, None]) & (positions > 0)
  return jnp.where(valid, result, jnp.nan)


@jax.jit
def log_prob_of_chosen_token(logits, chosen_index):
  logps = jax.nn.log_softmax(logits, axis=-1)
  chosen_prob = jnp.take_along_axis(logps, chosen_index[..., None], axis=-1)
  return chosen_prob[..., 0]


def sampling(logits, rng, algorithm, topk=0, nucleus_topp=0, temperature=1.0):
  if algorithm == "greedy":
    return jnp.argmax(logits, axis=-1)
  elif algorithm == "weighted":
    return jax.random.categorical(rng, logits / temperature)
  elif algorithm == "nucleus":
    return sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng)
  elif algorithm == "topk":
    return sample_topk_logits(logits, topk, temperature, rng)
  elif algorithm == "composite":
    return sample_topk_topp_weighted(logits, topk, nucleus_topp, temperature, rng)
  else:
    raise ValueError(f"Sampling {algorithm=} not supported!")


def sample_nucleus_topp_logits(logits, nucleus_topp, temperature, rng):
  if nucleus_topp < 0:
    raise ValueError("Can't apply nucleus with parameter {nucleus_topp=} less zero")
  logits_sorted = jnp.sort(logits, axis=-1)[..., ::-1]
  sorted_cum_probs = jnp.cumsum(jax.nn.softmax(logits_sorted, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit, jnp.full_like(logits, NEG_INF), logits)
  return jax.random.categorical(rng, logits / temperature)


def sample_topk_logits(logits, topk, temperature, rng):
  if topk <= 0:
    raise ValueError("Can't apply algorithm topk with parameter {topk=} less than or equal to zero")
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)
  topk_token = jnp.expand_dims(jax.random.categorical(rng, topk_logits / temperature).astype(jnp.int32), axis=-1)
  return jnp.squeeze(jnp.take_along_axis(topk_idxs, topk_token, axis=-1), axis=-1).astype(jnp.int32)


def sample_topk_topp_weighted(logits, topk, nucleus_topp, temperature, rng):
  if topk <= 0:
    raise ValueError(f"topk must be positive, got {topk=}")
  if not 0.0 < nucleus_topp <= 1.0:
    raise ValueError(f"nucleus_topp must be in (0, 1], got {nucleus_topp=}")
  topk_logits, topk_idxs = jax.lax.top_k(logits, topk)
  sorted_cum_probs = jnp.cumsum(jax.nn.softmax(topk_logits, axis=-1), axis=-1)
  cutoff_index = jnp.sum(sorted_cum_probs < nucleus_topp, axis=-1, keepdims=True)
  indices = jnp.arange(topk_logits.shape[-1])
  mask = indices <= cutoff_index
  filtered_topk_logits = jnp.where(mask, topk_logits, jnp.full_like(topk_logits, NEG_INF))
  scaled_logits = filtered_topk_logits / jnp.maximum(temperature, 1e-6)
  sampled_topk_index = jax.random.categorical(rng, scaled_logits).astype(jnp.int32)
  sampled_token = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, jnp.expand_dims(sampled_topk_index, axis=-1), axis=-1), axis=-1
  )
  return sampled_token.astype(jnp.int32)
