"""Continuous batching for the current model through the existing offline engine."""

import numpy as np

from lilax._backend.configs.pyconfig import HyperParameters
from lilax._backend.inference.offline_engine import CompletionOutput, OfflineEngine
from ._engine import _prompt_config


def generate_batch(model, inputs, generation_config, continuous_batching_config, rng):
  if not inputs:
    return []
  inputs = [np.asarray(tokens) for tokens in inputs]
  if any(tokens.ndim != 1 or not len(tokens) for tokens in inputs):
    raise ValueError("Each generation input must be a nonempty, unpadded token sequence.")
  if generation_config.max_new_tokens == 0:
    return [CompletionOutput(str(index), tokens.copy(), np.empty(0), len(tokens))
            for index, tokens in enumerate(inputs)]
  model.materialize(runtime=model.runtime)
  sampling = generation_config.sampling_options()
  config = _prompt_config(model.construction_config, max(map(len, inputs)))
  config = HyperParameters(config._pydantic_config.model_copy(update={
      "decode_sampling_strategy": sampling["algorithm"], "decode_sampling_top_k": sampling["topk"],
      "decode_sampling_nucleus_p": sampling["nucleus_topp"], "decode_sampling_temperature": sampling["temperature"],
      "return_log_prob": True,
  }))
  if any(len(tokens) > config.max_prefill_predict_length for tokens in inputs):
    raise ValueError("A generation input exceeds the model's max prefill length.")
  eos_ids = generation_config.eos_token_id
  if isinstance(eos_ids, int):
    eos_ids = [eos_ids]
  batching = continuous_batching_config.model_dump()
  if batching["prefill_lengths"] == "auto":
    limit = config.max_prefill_predict_length
    batching["prefill_lengths"] = [2**i for i in range(6, max(6, (limit - 1).bit_length()))] + [limit]
  engine = OfflineEngine(config, model=model.native_model, mesh=model.runtime.mesh, rng=rng, eos_ids=eos_ids,
                         max_new_tokens=generation_config.max_new_tokens, **batching)
  outputs = engine.batch_inference(inputs, rng=rng)
  return sorted(outputs, key=lambda output: int(output.index))
