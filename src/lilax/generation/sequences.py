"""Canonical batched token sequences over the existing single-prompt engine."""

import jax
import jax.numpy as jnp
import numpy as np

from ._engine import _prompt_config, generate
from lilax._processing.media import fold_media


def generate_sequences(model, input_ids, generation_config, *, attention_mask=None, rng=None, position_ids=None,
                       mrope_deltas=None, **media):
  input_ids = np.asarray(input_ids)
  if input_ids.ndim == 1:
    input_ids = input_ids[None, :]
  if input_ids.ndim != 2 or input_ids.shape[0] == 0:
    raise ValueError("input_ids must have shape (batch, sequence).")
  attention_mask = np.ones_like(input_ids) if attention_mask is None else np.asarray(attention_mask)
  if attention_mask.shape != input_ids.shape:
    raise ValueError("attention_mask must have the same shape as input_ids.")
  if generation_config.max_new_tokens == 0:
    return jnp.asarray(input_ids)
  prompt_lengths = np.count_nonzero(attention_mask, axis=1)
  if not prompt_lengths.all():
    raise ValueError("Each generation input must contain at least one unmasked token.")
  model.materialize(runtime=model.runtime)
  config = _prompt_config(model.construction_config, int(prompt_lengths.max()))
  prefill_length = config.max_prefill_predict_length
  eos_ids = generation_config.eos_token_id
  eos_ids = [] if eos_ids is None else [eos_ids] if isinstance(eos_ids, int) else eos_ids
  sequences = []
  for index, row in enumerate(input_ids):
    mask = attention_mask[index] != 0
    prompt = row[mask]
    if not 0 < len(prompt) <= prefill_length:
      raise ValueError(f"Prompt length must be between 1 and {prefill_length}.")
    tokens = np.pad(prompt, (0, prefill_length - len(prompt)))
    positions = None
    if position_ids is not None:
      positions = np.asarray(position_ids)[index][mask]
      positions = np.pad(positions, [(0, prefill_length - len(prompt))] + [(0, 0)] * (positions.ndim - 1))
    deltas = None if mrope_deltas is None else mrope_deltas[index:index + 1]
    if deltas is None and positions is not None and positions.ndim == 2 and config.use_mrope:
      deltas = np.asarray([[positions[:len(prompt)].max() + 1 - len(prompt)]])
    row_media = fold_media(config, {
        name: None if value is None else value[index:index + 1] for name, value in media.items()
    })
    generated = generate(
        model.native_model, model.runtime, tokens, len(prompt), construction_config=config,
        max_new_tokens=generation_config.max_new_tokens, rng=None if rng is None else jax.random.fold_in(rng, index),
        positions=positions, mrope_deltas=deltas,
        **generation_config.sampling_options(),
        **row_media,
    )
    generated = np.asarray(generated)
    ends = np.flatnonzero(np.isin(generated, eos_ids))
    if len(ends):
      generated = generated[:ends[0] + 1]
    sequences.append(np.concatenate((row, generated)))
  width = max(len(sequence) for sequence in sequences)
  pad_id = getattr(model.config, "pad_token_id", None)
  pad_id = eos_ids[0] if pad_id is None and eos_ids else 0 if pad_id is None else pad_id
  return jnp.asarray(np.stack([np.pad(sequence, (0, width - len(sequence)), constant_values=pad_id)
                               for sequence in sequences]))
