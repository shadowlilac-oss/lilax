"""Native batched generator for the benchmark API server."""

from dataclasses import dataclass, field
from io import StringIO
import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine import maxengine
from lilax._backend.multimodal import processor as mm_processor
from lilax._backend.multimodal import utils as mm_utils
from lilax._backend.observability.diagnostics import system_info
from lilax.configuration import MaxTextConfig, RLConfig
from lilax.runtime import Runtime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class LogProbs:
  tokens: list[int]
  token_logprobs: list[float]
  top_logprobs: list[None] | None = None
  text_offset: list[int] = field(default_factory=list)


@dataclass
class Completion:
  index: int
  text: str
  tokens: list[int]
  logprobs: LogProbs | None
  finish_reason: str = "stop"
  prompt_token_count: int = 0
  completion_token_count: int = 0


@dataclass
class GenerationStream:
  tokens: np.ndarray
  true_length: int
  image: np.ndarray | None
  generated_ids: list[int] = field(default_factory=list)
  generated_logprobs: list[float] = field(default_factory=list)
  prompt_ids: list[int] = field(default_factory=list)
  prompt_logprobs: list[float] = field(default_factory=list)
  finished: bool = False
  finish_reason: str = "length"


class MaxTextGenerator:
  """Generate batches through a native MaxEngine and its decode state."""

  def __init__(self, config):
    start_time = time.time()
    devices = None
    if isinstance(config, Runtime):
      config.resolve_config()
      devices = config.devices
      config = config.config
    elif isinstance(config, (MaxTextConfig, RLConfig)):
      config = resolve_runtime_config(config)
    self.config = config
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    self.rank = jax.process_index()
    self.logger = logging.getLogger(__name__)
    self._validate_config(config)
    self.logger.info("System information:")
    old_stdout = sys.stdout
    sys.stdout = captured_stdout = StringIO()
    system_info.print_system_information()
    sys.stdout = old_stdout
    self.logger.info(captured_stdout.getvalue())
    self.engine = maxengine.MaxEngine(config, devices=devices)
    self.batch_size = self.engine.max_concurrent_decodes
    assert self.batch_size >= 1, "Total batch size must be at least 1."
    self.rng = jax.random.PRNGKey(1234)
    self.logger.info("Loading model parameters...")
    self.rng, rng_load_params = jax.random.split(self.rng)
    self.params = self.engine.load_params(rng=rng_load_params)
    self.logger.info("Model parameters loaded.")
    self.metadata = self.engine.get_tokenizer()
    self.tokenizer = self.engine.build_tokenizer(self.metadata)
    eos_id = self.tokenizer.eos_id
    if not isinstance(eos_id, list):
      eos_id = [eos_id]
    self.eos_ids = eos_id
    try:
      self.has_chat_template = getattr(self.tokenizer.tokenizer, "chat_template", False)
    except AttributeError:
      self.has_chat_template = False
    self.logger.info("Chat Template available: %s", self.has_chat_template)
    self.rng, rng_init_decode = jax.random.split(self.rng)
    self.decode_state = self.engine.init_decode_state(rng=rng_init_decode)
    self._jitted_reset_state = jax.jit(lambda state: jax.tree_util.tree_map(jnp.zeros_like, state))
    self.logger.info("Initialization complete in %.2f seconds. Max batch size: %d", time.time() - start_time, self.batch_size)

  def generate_batch(
      self, prompts, image_paths=None, max_tokens=None, logprobs=None, echo=False, stop=None,
      temperature=None, seed=None, top_k=None, top_p=None,
  ):
    if image_paths is None:
      image_paths = [None] * len(prompts)
    if len(prompts) != len(image_paths):
      raise ValueError("The number of prompts must equal the number of image paths.")
    all_results = []
    for index in range(0, len(prompts), self.batch_size):
      all_results.extend(self._process_chunk(
          prompts[index:index + self.batch_size], image_paths[index:index + self.batch_size],
          max_tokens, logprobs, echo, stop, temperature, seed, top_k, top_p,
      ))
    return all_results

  def _process_chunk(self, prompts, image_paths, max_tokens, logprobs, echo, stop, temperature, seed, top_k, top_p):
    start_time = time.time()
    initialize_start_time = time.time()
    self.decode_state = self._jitted_reset_state(self.decode_state)
    streams, rng = self._initialize_streams_and_state(prompts, image_paths, seed)
    self.logger.info(
        "Initialization complete in %.2f seconds. Max batch size: %d",
        time.time() - initialize_start_time, self.batch_size,
    )
    if max_tokens is not None and max_tokens <= 0:
      self.logger.warning("max_tokens <= 0, returning empty completions.")
      return [Completion(index=index, text="", tokens=[], logprobs=None) for index in range(len(streams))]
    prefill_start_time = time.time()
    self.decode_state, rng = self._run_prefill_step(
        streams, self.decode_state, rng, logprobs, echo, temperature, top_k, top_p,
    )
    self.logger.info("Prefill step took %.2fs.", time.time() - prefill_start_time)
    generation_start_time = time.time()
    self.decode_state = self._run_generation_loop(
        streams, self.decode_state, rng, max_tokens, stop, temperature, top_k, top_p,
    )
    self.logger.info("Generation loop took %.2fs.", time.time() - generation_start_time)
    completions_start_time = time.time()
    completions = self._build_completions(streams, logprobs, echo)
    self.logger.info("Completions loop took %.2fs.", time.time() - completions_start_time)
    self.logger.info("Processed %d prompts in %.2fs.", len(prompts), time.time() - start_time)
    return completions

  def _initialize_streams_and_state(self, prompts, image_paths, seed):
    prefill_length = getattr(self.config, "max_prefill_predict_length", 1024)
    streams = []
    for prompt, image_path in zip(prompts, image_paths):
      tokens, true_length, images = self._preprocess_inputs(prompt, prefill_length, image_path)
      assert true_length <= prefill_length, f"Input token length {true_length} is > {prefill_length}"
      streams.append(GenerationStream(tokens=tokens, true_length=true_length, image=images))
    if seed is not None:
      rng = jax.random.PRNGKey(seed)
    else:
      self.rng, rng = jax.random.split(self.rng)
    return streams, rng

  def _determine_sampling_algorithm(self, temperature, top_k, top_p):
    if temperature == 0.0:
      return "greedy"
    if top_k is not None and top_p is not None:
      return "composite"
    if top_k is not None:
      return "topk"
    if top_p is not None:
      return "nucleus"
    if temperature is not None:
      return "weighted"
    return None

  def _run_prefill_step(self, streams, decode_state, rng, logprobs, echo, temperature, top_k, top_p):
    algorithm = self._determine_sampling_algorithm(temperature, top_k, top_p)
    prefill_results = {}
    for index, stream in enumerate(streams):
      rng, rng_prefill = jax.random.split(rng)
      prefill_result, _ = self.engine.prefill(
          params=self.params, padded_tokens=stream.tokens, true_length=stream.true_length, images=stream.image,
          rng=rng_prefill, slot=index, return_prompt_logp=logprobs is not None and echo,
          temperature=temperature, algorithm=algorithm, topk=top_k, nucleus_topp=top_p,
      )
      prefill_results[index] = prefill_result
      stream.prompt_ids.extend(map(int, np.array(stream.tokens[:stream.true_length], dtype=np.int32).tolist()))
      if prefill_result.get("prompt_logp") is not None:
        prompt_logp = np.array(prefill_result["prompt_logp"])[0, :stream.true_length]
        stream.prompt_logprobs.extend(float(value) for value in prompt_logp.tolist())
      stream.generated_ids.append(int(np.array(prefill_result["tokens"])[0, 0]))
      if prefill_result.get("token_logp") is not None:
        stream.generated_logprobs.append(float(np.array(prefill_result["token_logp"])[0, 0]))
    for slot_idx, result in prefill_results.items():
      decode_state = self.engine.insert(prefix=result, decode_state=decode_state, slot=slot_idx)
    return decode_state, rng

  def _run_generation_loop(self, streams, decode_state, rng, max_tokens, stop, temperature, top_k, top_p):
    target_length = getattr(self.config, "max_target_length", 2048)
    prefill_length = getattr(self.config, "max_prefill_predict_length", 1024)
    algorithm = self._determine_sampling_algorithm(temperature, top_k, top_p)
    stop_sequences = []
    max_stop_seq_len_tokens = 0
    if stop:
      stop_sequences = [sequence for sequence in ([stop] if isinstance(stop, str) else stop) if sequence]
      for sequence in stop_sequences:
        token_ids = self.tokenizer.tokenizer.encode(sequence, add_special_tokens=False)
        max_stop_seq_len_tokens = max(max_stop_seq_len_tokens, len(token_ids))
    total_steps = target_length - prefill_length
    if max_tokens is not None:
      total_steps = min(total_steps, max_tokens - 1)
    for step in range(total_steps):
      active_streams = [(index, stream) for index, stream in enumerate(streams) if not stream.finished]
      if not active_streams:
        self.logger.info("All streams finished. Breaking generation loop.")
        break
      rng, rng_generate = jax.random.split(rng)
      decode_state, _ = self.engine.generate(
          self.params, decode_state, rng=rng_generate, temperature=temperature,
          algorithm=algorithm, topk=top_k, nucleus_topp=top_p,
      )
      state_tokens = np.array(decode_state["tokens"])
      state_logp = None
      if (logp := decode_state.get("token_logp")) is not None:
        state_logp = np.array(logp)
      for slot_idx, stream in active_streams:
        token_id = int(state_tokens[slot_idx, 0])
        stream.generated_ids.append(token_id)
        if state_logp is not None:
          stream.generated_logprobs.append(float(state_logp[slot_idx, 0]))
        is_max_len = stream.true_length + 1 + step >= target_length
        is_eos = token_id in self.eos_ids
        stop_sequence_found = False
        if stop_sequences:
          start_index = max(0, len(stream.generated_ids) - (max_stop_seq_len_tokens + 2))
          trailing_ids = stream.generated_ids[start_index:]
          if trailing_ids:
            trailing_text = self.tokenizer.decode([int(token_id) for token_id in trailing_ids])
            stop_sequence_found = next((True for sequence in stop_sequences if sequence in trailing_text), False)
        if is_max_len or is_eos or stop_sequence_found:
          stream.finished = True
          if is_eos or stop_sequence_found:
            stream.finish_reason = "stop"
          if getattr(self.config, "attention", "") == "paged":
            self.engine.release_pages(slot=slot_idx)
    return decode_state

  def _build_completions(self, streams, logprobs, echo):
    completions = []
    for index, stream in enumerate(streams):
      generated_ids = stream.generated_ids[:]
      generated_logprobs = stream.generated_logprobs[:]
      if generated_ids and generated_ids[-1] in self.eos_ids:
        generated_ids = generated_ids[:-1]
        if len(generated_logprobs) >= len(stream.generated_ids):
          generated_logprobs = generated_logprobs[:-1]
      tokens = stream.prompt_ids + generated_ids if echo else generated_ids
      logps = stream.prompt_logprobs + generated_logprobs if echo else generated_logprobs
      text = self.tokenizer.decode(tokens)
      offsets = self._token_offsets(tokens, 0)
      payload = None
      if logprobs is not None:
        if len(tokens) != len(logps):
          self.logger.warning("[warn] Mismatched token/logprob lengths for stream %d. No logprobs returned.", index)
        else:
          payload = LogProbs(tokens=tokens, token_logprobs=logps, top_logprobs=None, text_offset=offsets)
      completions.append(Completion(
          index=index, text=text, tokens=tokens, logprobs=payload, finish_reason=stream.finish_reason,
          prompt_token_count=len(stream.prompt_ids), completion_token_count=len(generated_ids),
      ))
    return completions

  def _preprocess_inputs(self, text, prefill_length, image_path):
    processor_output = mm_utils.PreprocessorOutput()
    images = None
    if self.config.use_multimodal and image_path:
      text = mm_processor.reformat_prompt(
          prompt=self.config.prompt, image_placeholder=self.config.image_placeholder,
          model_name=self.config.model_name, num_images=1,
      )
      processor_output = mm_processor.preprocess_mm_data(self.config)
      prefill_length -= mm_processor.get_image_offsets(config=self.config, processor_output=processor_output)
      images = processor_output.pixel_values
    tokens, true_length = self.tokenizer.encode(text, is_bos=not self.has_chat_template, prefill_lengths=[prefill_length])
    if self.config.use_multimodal and image_path:
      tokens = mm_processor.prepare_text_for_image_fusion(tokens, config=self.config, processor_output=processor_output)
      true_length += mm_processor.get_image_offsets(config=self.config, processor_output=processor_output)
    return tokens, true_length, images

  def _validate_config(self, config):
    assert config.load_full_state_path == "", (
        "Decode doesn't operate on full states! Convert to parameter checkpoint first. Using generate_param_only_checkpoint."
    )
    assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    assert config.quantization != "nanoo_fp8", "NANOO fp8 on AMD MI300/MI325 GPUs is not supported in decode.py yet"

  def _token_offsets(self, token_ids, start=0):
    offsets = []
    pos = start
    decoded_so_far = ""
    prefix_ids = []
    for token_id in token_ids:
      offsets.append(pos)
      prefix_ids.append(int(token_id))
      new_text = self.tokenizer.decode(prefix_ids)
      piece_len = len(new_text) - len(decoded_so_far)
      if piece_len < 0:
        piece_len = len(self.tokenizer.decode([int(token_id)]))
      pos += piece_len
      decoded_so_far = new_text
    return offsets
