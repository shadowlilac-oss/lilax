import dataclasses
import functools
import os
import queue
import signal
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Hashable
from enum import Enum
from typing import Any, Callable

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from lilax._backend.inference.offline_output import completion_logprobs, completion_token_ids
from lilax._backend.inference.maxengine.maxengine import MaxEngine
from lilax._backend.input_pipeline.packing.prefill_packing import PrefillProcessor
from lilax._backend.input_pipeline.packing.prefill_packing import BatchedPrefillProcessor
from lilax._backend.observability import max_logging
from lilax._backend.distributed.mesh_topology import mesh_axes

DecodeState = Any
Params = Any
MaxTextConfig = Any


@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array | np.ndarray
  true_length: int


@dataclasses.dataclass
class CompletionOutput:
  index: str
  token_ids: np.ndarray
  logprobs: np.ndarray
  prompt_length: int


@dataclasses.dataclass
class TokenOutput:
  token: np.ndarray
  log_prob: np.ndarray


@dataclasses.dataclass
class DetokenizationTask:
  task_type: str
  result_tokens: Any = None
  log_prob: Any = None
  prompt_logp: Any = None
  prompt_ids: list = None
  slots: list = None
  tokens_buffer: list = None
  logprob_buffer: list = None


class SafeThread(threading.Thread):
  def run(self):
    try:
      super().run()
    except Exception as _:
      traceback.print_exc()
      os.kill(os.getpid(), signal.SIGKILL)


class PrefillType(Enum):
  DEFAULT = "default"
  BATCH = "batch"


@dataclasses.dataclass
class PrefillResult:
  result_tokens: "jetstream.engine_api.ResultTokens"
  slot: int
  prompt_logp: None | jax.Array


class PrefillHelper:
  def __init__(
      self,
      prefill_type: PrefillType,
      engine: MaxEngine,
      prefill_lengths: list[int],
      batch_prefill_max_batch_size: int = 16,
      rng=None,
  ):
    self._type = prefill_type
    self.engine = engine
    self.prefill_lengths = sorted(prefill_lengths)
    self.max_prefill_length = self.prefill_lengths[-1]
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    if prefill_type == PrefillType.DEFAULT:
      self._processor = PrefillProcessor(engine)
    elif prefill_type == PrefillType.BATCH:
      self._batch_processor = BatchedPrefillProcessor(
          engine=engine, max_batch_size=batch_prefill_max_batch_size, auto_layout_supported=False
      )
      self._processor = PrefillProcessor(engine)
    else:
      raise ValueError(f"Invalid prefill type: {prefill_type}")

  @functools.partial(jax.jit, static_argnums=(0), donate_argnames=("decode_state",))
  def _jitted_single_prefill(
      self, params, tokens, slot, true_length, decode_state, rng
  ) -> tuple[jax.Array, jax.Array, DecodeState, jax.Array] | tuple[jax.Array, jax.Array, DecodeState]:
    first_token, decode_state = self._processor._process(
        params, tokens, slot, true_length, decode_state, rng, return_prompt_logp=True
    )
    return first_token, decode_state, decode_state["prompt_logp"]

  def process(
      self,
      model_params: Params,
      decode_state: DecodeState,
      decode_slot: int,
      input_id: int,
      input_tokens_padded: jax.Array,
      input_true_length: int,
      prefill_done: Callable,
  ) -> None:
    padded_length = len(input_tokens_padded)
    if self._type == PrefillType.DEFAULT or padded_length == self.max_prefill_length:
      first_token, decode_state, prompt_logp = self._jitted_single_prefill(
          model_params, input_tokens_padded, decode_slot, input_true_length, decode_state, self.rng
      )
      prefill_done([PrefillResult(first_token, decode_slot, prompt_logp)], [input_id], decode_state)
    elif self._type == PrefillType.BATCH:
      self._batch_processor.process(
          model_params, decode_state, decode_slot, input_id, input_tokens_padded[:input_true_length],
          padded_length, self.max_prefill_length, prefill_done, return_prompt_logp=True,
      )

  def finalize(self, model_params: Params, decode_state: DecodeState, prefill_done: Callable) -> None:
    if self._type == PrefillType.DEFAULT:
      pass
    elif self._type == PrefillType.BATCH:
      self._batch_processor.flush(model_params, decode_state, prefill_done, return_prompt_logp=True)


class InferenceWorker:
  def __init__(
      self,
      config: MaxTextConfig,
      params: Params | None,
      min_decode_steps: int,
      enable_batch_prefill: bool,
      devices: list[Any],
      tokenizer: Any,
      eos_ids: list[int],
      prefill_lengths: list[int],
      max_decode_length: int,
      batch_prefill_max_batch_size: int,
      is_pw_reshard: bool = True,
      rng: jax.random.PRNGKey = None,
      mesh: Mesh = None,
      debug: bool = False,
      model=None,
  ):
    self.config = config
    self.params = params
    self.model = model
    self.devices = devices
    self.is_pw_reshard = is_pw_reshard
    self.enable_batch_prefill = enable_batch_prefill
    self.prefill_type = PrefillType.BATCH if enable_batch_prefill else PrefillType.DEFAULT
    self.prefill_lengths = prefill_lengths
    self.max_prefill_length = self.prefill_lengths[-1]
    self.max_decode_length = max_decode_length
    self.eos_ids = eos_ids
    self.tokenizer = tokenizer
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.min_decode_steps = min_decode_steps
    self.mesh = mesh
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    self.debug = debug
    self.running = False
    self.detokenization_queue = queue.Queue()
    self.empty_decode_slots = set()
    self.slot_to_id: dict[int, None | int] = {}
    self.completed_sequences: set = set()
    self.decode_state: DecodeState = None
    self.completion_tokens_by_id: dict[Hashable, list[TokenOutput]] = {}
    self.prompt_logprobs_by_id: dict[Hashable, list[np.ndarray]] = {}
    self.true_lengths: dict[Hashable, int] = {}
    self.engine = None
    self.decode_batch_size = None
    self.prefill_helper = None
    self.generate_fn = None

    start_time = time.time()
    self.params, self.engine = self._init_engine(self.params)
    self.tokenizer = self._init_tokenizer()
    self.decode_batch_size = self.engine.max_concurrent_decodes
    self.prefill_helper = PrefillHelper(
        self.prefill_type, self.engine, self.prefill_lengths, self.batch_prefill_max_batch_size, rng=self.rng
    )
    start_time_decode_state = time.time()
    self.generate_fn = self.engine.generate
    self.decode_state = self.engine.init_decode_state(self.rng)
    if self.debug:
      max_logging.log(f"time taken to initialize decode_state: {time.time() - start_time_decode_state} seconds")
    max_logging.log(f"Initialized Inference worker in {time.time() - start_time} seconds")

  def _init_engine(self, params):
    start_time = time.time()
    engine = MaxEngine(self.config, self.devices)
    params = engine.load_params(params=params, model=self.model, rng=self.rng)
    max_logging.log(f"Time taken to initialize engine: {time.time() - start_time} seconds")
    return params, engine

  def _init_tokenizer(self):
    if self.eos_ids is None and self.tokenizer is None:
      tokenizer_params = self.engine.get_tokenizer()
      self.tokenizer = self.engine.build_tokenizer(tokenizer_params)
    if self.eos_ids is None:
      self.eos_ids = [self.tokenizer.eos_id]
    return self.tokenizer

  def update_params(self, params: Params):
    self.params = params

  def reset_state(self):
    max_logging.log("Resetting InferenceWorker state")
    self.running = False
    self.completion_tokens_by_id = defaultdict(list)
    self.prompt_logprobs_by_id = defaultdict(list)
    self.empty_decode_slots = set()
    for i in range(self.decode_batch_size):
      self.empty_decode_slots.add(i)
    self.slot_to_id = {}
    self.true_lengths = {}
    self.detokenization_queue = queue.Queue()
    self.completed_sequences = set()
    max_logging.log("InferenceWorker state reset complete")

  def run_inference(self, data: list[InputData], rng=None):
    self.reset_state()
    if rng is not None:
      self.rng = rng
    self.true_lengths = {input.id: input.true_length for input in data}
    self.running = True
    max_logging.log("Continuous batching started")
    self._run_continuous_batching(data)
    return self._build_final_outputs(data)

  def _run_continuous_batching(self, data: list[InputData]):
    detokenization_thread = SafeThread(target=self.background_detokenization, name="detokenization")
    detokenization_thread.start()
    for row in data:
      while not self.empty_decode_slots:
        self.decode()
      slot = self.empty_decode_slots.pop()
      self.prefill_helper.process(
          model_params=self.params,
          decode_state=self.decode_state,
          decode_slot=slot,
          input_id=row.id,
          input_tokens_padded=row.tokens,
          input_true_length=row.true_length,
          prefill_done=self.prefill_done,
      )
    self.prefill_helper.finalize(self.params, self.decode_state, self.prefill_done)
    while not all(value is None for value in self.slot_to_id.values()):
      self.decode()
    self.running = False
    max_logging.log("Inference worker: joining detokenization thread")
    start_time = time.time()
    with jax.profiler.TraceAnnotation("Flushing detokenization thread"):
      detokenization_thread.join()
    max_logging.log(f"Inference worker: detokenization thread joined in {time.time() - start_time} seconds")

  def _build_final_outputs(self, input_data: list[InputData]) -> list[CompletionOutput]:
    with jax.profiler.TraceAnnotation("offline_engine.batch_inference.return_final_output"):
      completion_outputs = []
      for row in input_data:
        input_id = row.id
        prompt_length = row.true_length
        prompt_tokens = row.tokens[: row.true_length].squeeze()
        completion_tokens = completion_token_ids(self.completion_tokens_by_id[input_id])
        logprobs = completion_logprobs(self.completion_tokens_by_id[input_id])
        prompt_logprobs = np.array(self.prompt_logprobs_by_id[input_id]).flatten()
        completion_outputs.append(
            CompletionOutput(
                index=str(input_id),
                prompt_length=prompt_length,
                token_ids=np.concatenate((prompt_tokens, completion_tokens)),
                logprobs=np.concatenate((prompt_logprobs, logprobs)),
            )
        )
    return completion_outputs

  def prefill_done(self, prefill_result: list[PrefillResult], prompt_ids: list[int], decode_state: DecodeState):
    self.decode_state = decode_state
    slots = []
    result_tokens_list = []
    prompt_logp_list = []
    for i, result in enumerate(prefill_result):
      input_id = prompt_ids[i]
      slot = result.slot
      self.slot_to_id[slot] = input_id
      slots.append(slot)
      result_tokens_list.append(result.result_tokens)
      prompt_logp_list.append(result.prompt_logp)
    task = DetokenizationTask(
        task_type="prefill", result_tokens=result_tokens_list, prompt_logp=prompt_logp_list,
        prompt_ids=prompt_ids, slots=slots,
    )
    self.detokenization_queue.put_nowait(task)

  def decode(self):
    for i in range(self.min_decode_steps):
      self.decode_state, result_tokens, log_prob = self._jitted_generate_fn(self.params, self.decode_state, self.rng)
      if i == self.min_decode_steps - 1:
        jax.block_until_ready(result_tokens)
      task = DetokenizationTask(task_type="decode", tokens_buffer=result_tokens, logprob_buffer=log_prob)
      self.detokenization_queue.put_nowait(task)

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def _jitted_generate_fn(self, params, decode_state, rng):
    decode_state, result_tokens = self.engine.generate(params, decode_state, rng=rng)
    return decode_state, result_tokens.data[:, 0], result_tokens.log_prob

  def background_detokenization(self):
    max_logging.log("Inference worker: starting detokenization thread")
    while True:
      try:
        task = self.detokenization_queue.get(timeout=0.1)
      except queue.Empty:
        if not self.running and self.detokenization_queue.empty():
          break
        continue
      start_time = time.time()
      newly_empty = []
      if task.task_type == "prefill":
        with jax.profiler.TraceAnnotation("convert_to_numpy_and_emit_prefill"):
          for i, result_tokens in enumerate(task.result_tokens):
            prompt_id = task.prompt_ids[i]
            slot = task.slots[i]
            prompt_logp = task.prompt_logp[i]
            true_length = self.true_lengths[prompt_id]
            first_token = np.array(result_tokens.data[:, 0])
            log_prob = np.array(result_tokens.log_prob)
            prompt_logp_np = np.array(prompt_logp)[:, :true_length]
            should_terminate = self.emit_token(prompt_id, int(first_token), log_prob, prompt_logp=prompt_logp_np)
            if should_terminate:
              newly_empty.append(slot)
      elif task.task_type == "decode":
        active_slots = []
        for slot, id_ in self.slot_to_id.items():
          if id_ is not None and id_ not in self.completed_sequences:
            active_slots.append((slot, id_))
        if not active_slots:
          continue
        with jax.profiler.TraceAnnotation("convert_to_numpy_and_emit_decode_step"):
          result_tokens_step = np.array(task.tokens_buffer)
          log_prob_step = np.array(task.logprob_buffer)
          for slot, id_ in active_slots:
            log_prob_at_slot = log_prob_step[slot]
            result_tokens_at_slot = result_tokens_step[slot]
            should_terminate = self.emit_token(id_, int(result_tokens_at_slot), log_prob_at_slot)
            if should_terminate:
              newly_empty.append(slot)
      for slot in newly_empty:
        self.slot_to_id[slot] = None
        self.empty_decode_slots.add(slot)
      if self.debug:
        max_logging.log(f"Inference worker: detokenization in {time.time() - start_time} seconds")

  def emit_token(self, prompt_id, result_token: int, log_prob: float, prompt_logp: None | np.ndarray = None):
    if prompt_id in self.completed_sequences:
      return True
    index = len(self.completion_tokens_by_id[prompt_id])
    if prompt_logp is not None:
      self.prompt_logprobs_by_id[prompt_id] = [prompt_logp]
    self.completion_tokens_by_id[prompt_id].append(TokenOutput(np.array(result_token), np.array(log_prob)))
    should_terminate = (result_token in self.eos_ids) or (index + 1 == self.max_decode_length)
    if should_terminate:
      self.completed_sequences.add(prompt_id)
    return should_terminate


class OfflineEngine:
  def __init__(
      self,
      config: Any,
      params: None | Params = None,
      enable_batch_prefill: bool = False,
      min_decode_steps: int = 10,
      tokenizer: Any = None,
      eos_ids: list[int] | None = None,
      prefill_lengths: list[int] | str = "auto",
      batch_prefill_max_batch_size: int = 16,
      mesh: Mesh = None,
      rng: jax.random.PRNGKey = None,
      debug: bool = False,
      *, model=None, max_new_tokens=None,
  ):
    max_logging.log("Initializing OfflineEngine")
    self.config = config
    self.params = params
    self.min_decode_steps = min_decode_steps
    self.enable_batch_prefill = enable_batch_prefill
    self.mesh = mesh
    self.tokenizer = tokenizer
    self.eos_ids = eos_ids
    self.prefill_lengths = prefill_lengths
    self.batch_prefill_max_batch_size = batch_prefill_max_batch_size
    self.max_prefill_length = self.config.max_prefill_predict_length
    self.max_decode_length = self.config.max_target_length - self.max_prefill_length
    if max_new_tokens is not None:
      if not 0 < max_new_tokens <= self.max_decode_length:
        raise ValueError(f"max_new_tokens must be between 1 and {self.max_decode_length} for this engine.")
      self.max_decode_length = max_new_tokens
    self.rng = jax.random.PRNGKey(0) if rng is None else rng
    self.debug = debug
    self._validate_config()
    if prefill_lengths == "auto":
      self.prefill_lengths = [2**i for i in range(6, max(6, (self.max_prefill_length - 1).bit_length()) + 1)]
    else:
      self.prefill_lengths = sorted(prefill_lengths)
    if not self.mesh:
      self.mesh = OfflineEngine.create_mesh(jax.devices(), self.config)
    self.worker = InferenceWorker(
        config=self.config,
        params=self.params,
        min_decode_steps=self.min_decode_steps,
        enable_batch_prefill=self.enable_batch_prefill,
        mesh=self.mesh,
        devices=self.mesh.devices.flatten(),
        tokenizer=self.tokenizer,
        eos_ids=self.eos_ids,
        prefill_lengths=self.prefill_lengths,
        max_decode_length=self.max_decode_length,
        batch_prefill_max_batch_size=self.batch_prefill_max_batch_size,
        rng=self.rng,
        debug=self.debug,
        model=model,
    )
    self.tokenizer = self.worker.tokenizer

  def update_params(self, params: Params):
    self.worker.update_params(params)

  def batch_inference(
      self, data: list[InputData] | list[jax.Array] | list[np.ndarray], desc: str = "", rng=None
  ) -> list[CompletionOutput]:
    data = self.prepare_data(data)
    return self.worker.run_inference(data, rng)

  def prepare_data(self, data: list[InputData | jax.Array | np.ndarray]) -> list[InputData]:
    if isinstance(data[0], jax.Array):
      data = [np.array(array) for array in data]
    if isinstance(data[0], np.ndarray):
      max_logging.log(
          "When you provide JAX/numpy arrays to Offline Engine, "
          "make sure that the arrays are not padded with padding tokens."
      )
      data = [InputData(id=i, tokens=array, true_length=len(array)) for i, array in enumerate(data)]
    if len(data) != len({item.id for item in data}):
      raise ValueError("All data ids must be unique")
    data = self.pad_data(data)
    if self.enable_batch_prefill:
      return sorted(data, key=lambda x: x.tokens.shape[0])
    return data

  def pad_data(self, data: list[InputData]) -> list[InputData]:
    padded_data = []
    for item in data:
      target_length = None
      for length in self.prefill_lengths:
        if length >= item.true_length:
          target_length = length
          break
      if target_length is None:
        target_length = self.max_prefill_length
      if len(item.tokens) < target_length:
        padded_tokens = np.zeros(target_length, dtype=item.tokens.dtype)
        padded_tokens[: item.true_length] = item.tokens[: item.true_length]
      else:
        padded_tokens = item.tokens[:target_length]
      padded_data.append(InputData(id=item.id, tokens=padded_tokens, true_length=item.true_length))
    return padded_data

  @staticmethod
  def create_mesh(devices, config):
    ici_parallelism = mesh_axes.fill_unspecified_mesh_axes(config.ici_parallelism.copy(), len(devices), "ICI")
    devices_array = mesh_utils.create_device_mesh(
        ici_parallelism, devices, contiguous_submeshes=False,
        allow_split_physical_axes=config.allow_split_physical_axes or False,
    )
    mesh = Mesh(devices_array.reshape(ici_parallelism), config.mesh_axes)
    return mesh

  def _validate_config(self):
    if not self.config.return_log_prob:
      raise ValueError("return_log_prob must be True when using OfflineEngine")
    if self.enable_batch_prefill and self.config.scan_layers:
      raise ValueError("scan_layers must be False if enable_batch_prefill is True")
    if self.max_decode_length <= 0:
      raise ValueError("Make sure max_target_length - max_prefill_predict_length is greater than 0")
    if self.config.scan_layers:
      max_logging.log(
          "WARNING: scan_layers=True will result in slow step time. " "It is recommended for debugging purposes only."
      )
