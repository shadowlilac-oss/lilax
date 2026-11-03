"""Public access to the native MaxEngine inference lifecycle."""

import jax
import jax.numpy as jnp

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine.engine_config import MaxEngineConfig
from lilax._backend.inference.maxengine.maxengine import ExistingPrefix
from lilax._backend.inference.maxengine.maxengine import MaxEngine as NativeMaxEngine
from lilax.configuration import MaxTextConfig, RLConfig
from lilax.runtime import Runtime


class MaxEngine(NativeMaxEngine):
  """Native inference engine using its own inference mesh."""

  def __init__(self, config, devices=None):
    if isinstance(config, Runtime):
      if devices is None:
        devices = config.devices
      config = config.config
    if isinstance(config, (MaxTextConfig, RLConfig)):
      config = resolve_runtime_config(config)
    super().__init__(config, devices=devices)


def generate(
    native_model, runtime, padded_tokens, true_length, *, construction_config, max_new_tokens=None, rng=None,
    algorithm=None, topk=None, nucleus_topp=None, temperature=None, positions=None, mrope_deltas=None,
    images=None, image_masks=None, videos=None, video_masks=None, video_grid_thw=None,
    audio_values=None, audio_masks=None,
):
  """Generate token IDs for one padded prompt using a loaded native model.

  The result is a one-dimensional array of generated token IDs, excluding the prompt.
  The caller supplies tokenization and any multimodal preprocessing.
  """
  if runtime.mesh is None:
    raise ValueError("Runtime must be initialized before generating tokens.")
  engine = MaxEngine(construction_config, devices=runtime.devices)
  padded_tokens = jnp.asarray(padded_tokens)
  if padded_tokens.ndim != 1 or not 0 < true_length <= padded_tokens.shape[0] <= engine.max_prefill_length:
    raise ValueError("Expected one padded prompt with 0 < true_length <= padded length <= max prefill length.")
  max_steps = engine.config.max_target_length - engine.config.max_prefill_predict_length + 1
  if max_new_tokens is None:
    max_new_tokens = max_steps
  if not 0 <= max_new_tokens <= max_steps:
    raise ValueError(f"max_new_tokens must be between 0 and {max_steps} for this engine configuration.")
  if max_new_tokens == 0:
    return jnp.empty((0,), dtype=jnp.int32)

  if rng is None:
    rng = jax.random.PRNGKey(0)
  rng, load_rng, prefill_rng = jax.random.split(rng, 3)
  params = engine.load_params(model=native_model, rng=load_rng)
  sampling = dict(algorithm=algorithm, topk=topk, nucleus_topp=nucleus_topp, temperature=temperature)
  prefix, first = engine.prefill(
      params=params, padded_tokens=padded_tokens, true_length=true_length, rng=prefill_rng, slot=0,
      positions=positions, mrope_deltas=mrope_deltas, images=images, image_masks=image_masks,
      videos=videos, video_masks=video_masks, video_grid_thw=video_grid_thw,
      audio_values=audio_values, audio_masks=audio_masks, **sampling,
  )
  tokens = [first.data[0, slice(*first.tokens_idx)]]
  if max_new_tokens > 1:
    decode_state = engine.insert(prefix, engine.init_decode_state(), slot=0)
    for _ in range(max_new_tokens - 1):
      rng, step_rng = jax.random.split(rng)
      decode_state, result = engine.generate(params, decode_state, rng=step_rng, **sampling)
      tokens.append(result.data[0, slice(*result.tokens_idx)])
  return jnp.concatenate(tokens)


def create_engine_from_config(
    config: MaxTextConfig, batch_size, max_prefill_predict_length, max_target_length, *, devices=None, **overrides
):
  """Create an MLPerf-configured engine from an inert public configuration."""
  defaults = {
      "scan_layers": False,
      "async_checkpointing": False,
      "ici_fsdp_parallelism": 1,
      "ici_autoregressive_parallelism": 1,
      "ici_tensor_parallelism": -1,
      "weight_dtype": "bfloat16",
      "attention": "dot_product",
  }
  values = config.get_keys()
  values.update({key: value for key, value in defaults.items() if key not in config._explicit_keys})
  values.update(
      per_device_batch_size=batch_size,
      max_prefill_predict_length=max_prefill_predict_length,
      max_target_length=max_target_length,
  )
  values.update(overrides)
  explicit_keys = config._explicit_keys | defaults.keys() | {
      "per_device_batch_size", "max_prefill_predict_length", "max_target_length", *overrides
  }
  assert "load_parameters_path" in config._explicit_keys or "load_parameters_path" in overrides, (
      "load_parameters_path must be defined"
  )
  return MaxEngine(MaxTextConfig._from_resolved(values, explicit_keys), devices=devices)
