"""Token sampling with the engine's configured sampling defaults."""

from lilax._backend.inference import inference_utils


def sample_tokens(config, logits, rng, algorithm, topk, nucleus_topp, temperature):
  """Sample tokens, using the configured default for each sampling argument left as None."""
  return inference_utils.sampling(
      logits,
      rng,
      algorithm if algorithm is not None else config.decode_sampling_strategy,
      topk=topk if topk is not None else config.decode_sampling_top_k,
      nucleus_topp=nucleus_topp if nucleus_topp is not None else config.decode_sampling_nucleus_p,
      temperature=temperature if temperature is not None else config.decode_sampling_temperature,
  )
