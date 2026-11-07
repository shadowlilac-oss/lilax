"""Native single-stream decode."""

import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jetstream.engine import engine_api

from lilax._backend.common import profiler
from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine import maxengine
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils
from lilax._backend.observability import max_logging
from lilax._backend.observability.diagnostics import system_info
from lilax._backend.configs.inert import MaxTextConfig, RLConfig
from lilax.runtime import Runtime


_NUM_STREAMS = 1


def _batch_first_result_token(first_tokens: list[Any], batch_size: int):
  data = jnp.vstack([first_token.data for first_token in first_tokens])

  def _pad_to_batch_size(data: jax.Array, batch_size: int):
    pad_width = [(0, batch_size - data.shape[0]), (0, 0)]
    return jnp.pad(data, pad_width, mode="constant", constant_values=0)

  result_tokens = engine_api.ResultTokens(
      data=_pad_to_batch_size(data, batch_size), tokens_idx=(0, 1), valid_idx=(1, 2), length_idx=(2, 3),
      samples_per_slot=1,
  )

  def _all_equals(elements: Sequence[jax.Array], target: jax.Array):
    stacked = jnp.stack(elements)
    return jnp.all(stacked == target)

  assert _all_equals([jnp.array(token.tokens_idx) for token in first_tokens], jnp.array(result_tokens.tokens_idx))
  assert _all_equals([jnp.array(token.valid_idx) for token in first_tokens], jnp.array(result_tokens.valid_idx))
  assert _all_equals([jnp.array(token.length_idx) for token in first_tokens], jnp.array(result_tokens.length_idx))
  assert _all_equals(
      [jnp.array(token.samples_per_slot) for token in first_tokens], jnp.array(result_tokens.samples_per_slot)
  )
  return result_tokens


def decode(config):
  """Decode the configured prompt with the native MaxEngine lifecycle."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  devices = None
  if isinstance(config, Runtime):
    config.resolve_config()
    devices = config.devices
    config = config.config
  elif isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  _validate_config(config)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  system_info.print_system_information()

  engine = maxengine.MaxEngine(config, devices=devices)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)
  prof = profiler.Profiler(config)

  text = config.prompt
  prefill_length = config.max_prefill_predict_length
  processor_outputs = mm_utils.PreprocessorOutput()
  if config.use_multimodal:
    processor_outputs = mm_processor.preprocess_mm_data(config)
    image_offsets = mm_processor.get_image_offsets(config=config, processor_output=processor_outputs)
    prefill_length -= image_offsets
    text = mm_processor.reformat_prompt(
        prompt=config.prompt, image_placeholder=config.image_placeholder, video_placeholder=config.video_placeholder,
        model_name=config.model_name, num_images=processor_outputs.num_images,
        num_videos=getattr(processor_outputs, "num_videos", 0), audio_placeholder=config.audio_placeholder,
        num_audios=getattr(processor_outputs, "num_audios", 0),
    )

  tokenizer_model = engine.build_tokenizer(engine.get_tokenizer())
  try:
    has_chat_template = getattr(tokenizer_model.tokenizer, "chat_template", False)
  except AttributeError:
    has_chat_template = False
  is_bos = config.add_bos and not has_chat_template
  tokens, true_length = tokenizer_model.encode(text, is_bos=is_bos, prefill_lengths=[prefill_length])
  position_ids = None
  mrope_position_deltas = None
  if config.use_multimodal:
    tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=processor_outputs)
    true_length += image_offsets
    if config.use_mrope:
      # Loads the decord/GCS video dependencies here, as the former processor import did.
      from lilax._backend.multimodal.qwen3_omni.video import loading  # pylint: disable=unused-import
      from lilax._backend.multimodal.qwen3_omni import positions

      position_ids, mrope_position_deltas = positions.get_rope_index(
          input_ids=tokens[np.newaxis, :], image_grid_thw=processor_outputs.pixel_grid_thw,
          video_grid_thw=processor_outputs.video_grid_thw, attention_mask=np.ones_like(tokens)[np.newaxis, :],
          use_audio_in_video=config.use_audio and getattr(processor_outputs, "num_videos", 0) > 0,
          audio_lengths=processor_outputs.audio_lengths, second_per_grids=processor_outputs.video_second_per_grid,
          spatial_merge_size=config.spatial_merge_size_for_vit,
          position_id_per_seconds=config.position_id_per_seconds, config=config,
      )
  assert true_length <= config.max_prefill_predict_length, (
      f"Input token length {true_length} is longer than {config.max_prefill_predict_length=}"
  )
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
  assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"
  batch_size = int(config.per_device_batch_size * jax.device_count())
  assert 0 < _NUM_STREAMS <= batch_size, (
      f"The number of streams {_NUM_STREAMS} must be > 0 and <= batch size {batch_size}"
  )

  prefill_result_list = []
  first_token_list = []
  sampled_tokens_list = []
  prof.activate(optional_postfix="trace")
  rng, rng_prefill = jax.random.split(rng)
  for stream in range(_NUM_STREAMS):
    with jax.profiler.StepTraceAnnotation("prefill", stream=stream):
      prefill_result, first_token = engine.prefill(
          params=params, padded_tokens=tokens, positions=position_ids, mrope_deltas=mrope_position_deltas,
          images=processor_outputs.pixel_values if config.use_multimodal else None,
          image_masks=processor_outputs.pixel_mask if config.use_multimodal and "llama4" in config.model_name else None,
          videos=getattr(processor_outputs, "video_values", None) if config.use_multimodal else None,
          video_masks=getattr(processor_outputs, "video_mask", None) if config.use_multimodal else None,
          video_grid_thw=getattr(processor_outputs, "video_grid_thw", None) if config.use_multimodal else None,
          audio_values=processor_outputs.audio_values if config.use_audio else None,
          audio_masks=processor_outputs.audio_mask if config.use_audio else None,
          true_length=true_length, rng=rng_prefill, slot=stream,
      )
    prefill_result_list.append(prefill_result)
    first_token_list.append(first_token)

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  for stream in range(_NUM_STREAMS):
    decode_state = engine.insert(prefill_result_list[stream], decode_state, slot=stream)

  prof_deactivated = False
  sampled_tokens_list.append(_batch_first_result_token(first_token_list, batch_size))
  for step in range(config.max_prefill_predict_length, config.max_target_length):
    rng, rng_generate = jax.random.split(rng)
    with jax.profiler.StepTraceAnnotation("generate", step=step):
      decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
    if step > config.max_prefill_predict_length + config.profiler_steps:
      prof.deactivate(blocking_object=sampled_tokens)
      prof_deactivated = True
    sampled_tokens_list.append(sampled_tokens)

  for stream in range(_NUM_STREAMS):
    results = [token.get_result_at_slot(stream).tokens.item() for token in sampled_tokens_list]
    max_logging.debug(f"raw token ids for stream {stream}: {results}")
    eos_token_id = None
    if hasattr(tokenizer_model, "tokenizer") and hasattr(tokenizer_model.tokenizer, "eos_token_id"):
      eos_token_id = tokenizer_model.tokenizer.eos_token_id
    max_logging.debug(f"using eos_token_id: {eos_token_id}")
    if eos_token_id is not None and eos_token_id in results:
      max_logging.info(
          f"EOS token {eos_token_id} found at index {results.index(eos_token_id)};"
          f" output sequence truncated to length {results.index(eos_token_id) + 1}"
      )
      results = results[: results.index(eos_token_id) + 1]
    output = tokenizer_model.decode(results)
    print(f"Input `{text}` -> `{output}`")

  assert " ".join(output.split()).startswith(" ".join(config.autoregressive_decode_assert.split())), (
      f"generated text mismatch {output=}, {config.autoregressive_decode_assert=}"
  )
  if not prof_deactivated:
    prof.deactivate(blocking_object=output)
  prof.post_process()
  return output


def _validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first."
      "Using generate_param_only_checkpoint."
  )
