"""Python interface for the full-forward-pass decoding example."""

import jax
from jax.experimental import multihost_utils
import numpy as np

from lilax._backend.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR
from lilax.generation import sampling


def decode_next_token(model, tokenizer, prompt):
  """Run the decoding demo using a loaded MaxTextModel and tokenizer.

  The model's runtime configuration supplies the sequence length, batch size,
  random seed, and sampling strategy. The tokenizer must provide encode and
  decode methods. Returns the prepared inputs, gathered logits, token ID, and
  decoded token.
  """
  config = model.runtime.config
  input_ids = tokenizer.encode(prompt)
  padded_ids = np.zeros(config.max_target_length, dtype=np.int32)
  padded_ids[: len(input_ids)] = input_ids
  batch_size = config.global_batch_size_to_train_on
  ids = np.stack([padded_ids for _ in range(batch_size)])
  decoder_segment_ids = np.zeros((batch_size, config.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
  decoder_positions = np.stack([np.arange(config.max_target_length, dtype=np.int32) for _ in range(batch_size)])

  with model.runtime.context():
    logits = model(ids, decoder_positions, decoder_segment_ids, enable_dropout=False)
  logits = multihost_utils.process_allgather(logits)
  selected_logits = jax.lax.dynamic_slice(
      logits, (0, 0, logits.shape[2] - 2, 0), (1, 1, 1, logits.shape[3])
  )
  rng = jax.random.PRNGKey(config.init_weights_seed)
  _, sample_rng = jax.random.split(rng)
  token_id = sampling(selected_logits, sample_rng, config.decode_sampling_strategy).item()
  return {
      "input_ids": input_ids,
      "ids": ids,
      "decoder_segment_ids": decoder_segment_ids,
      "decoder_positions": decoder_positions,
      "logits": logits,
      "token_id": token_id,
      "text": tokenizer.decode([token_id]),
  }
