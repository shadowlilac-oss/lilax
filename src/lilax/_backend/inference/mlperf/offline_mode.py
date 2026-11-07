"""MLPerf LoadGen offline inference orchestration."""

import array
import contextlib
import copy
from dataclasses import dataclass, field
import gc
import json
import logging
import math
import os
import time
import warnings

import jax
import jax.numpy as jnp
import mlperf_loadgen as lg
import numpy as np
import pandas as pd

from lilax._backend.inference.mlperf import offline_inference
from lilax.generation._engine import create_engine_from_config


warnings.simplefilter("ignore", category=FutureWarning)
_MLPERF_ID = "llama2-70b"
log = logging.getLogger(__name__)
log.setLevel(os.getenv("LOGLEVEL", "INFO"))


@dataclass
class OfflineModeArguments:
  dataset_path: str
  mlperf_test_mode: str = "performance"
  api_url: str | None = None
  is_stream: bool = False
  input_mode: str = "tokenized"
  output_mode: str = "tokenized"
  audit_conf: str = "audit.conf"
  mlperf_conf: str = "mlperf.conf"
  user_conf: str = "user.conf"
  total_sample_count: int = 24576
  perf_count_override: int | None = None
  output_log_dir: str = "output-logs"
  enable_log_trace: bool = False
  prefill_lengths_and_per_device_batch_sizes: str = "256,80|512,40|1024,20"
  jax_profiler_port: int = 9999
  enable_profile: bool = False
  enable_batch_prefill: bool = False
  skip_warmup: bool = False
  tok_outlen_multiplier: float = 3.0
  allow_skipping_queries: bool = False
  rename_dataset_cols: str = ""
  engine_overrides: dict = field(default_factory=dict)


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))), 128)
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


def _init_query_batches(args):
  query_batches = {}
  for length_batch in args.prefill_lengths_and_per_device_batch_sizes.split("|"):
    length, batch = length_batch.split(",")
    query_batches[(int(length), int(batch))] = []
  return query_batches


@contextlib.contextmanager
def timed(msg):
  log.info("%s start", msg)
  start = time.perf_counter()
  yield
  log.info("%s done: %d", msg, time.perf_counter() - start)


def _classify_query(dataset_rows, index, query_batches, args):
  sample = dataset_rows[index][1]
  input_len = sample.tok_input_length
  total_len = int(input_len + args.tok_outlen_multiplier * sample.tok_output_length)
  query_batch_keys = sorted(query_batches.keys())
  target_inputs = [length_batch[0] for length_batch in query_batch_keys]
  target_totals = [2 * length for length in target_inputs]
  for index, target_input in enumerate(target_inputs):
    if total_len <= target_totals[index] and input_len <= target_input:
      log.debug("Added sample of input length %d total_len %d for %s", input_len, total_len, query_batch_keys[index])
      return query_batch_keys[index]
  if input_len <= target_inputs[-1]:
    log.debug("Added sample of input length %d total_len %d for %s", input_len, total_len, query_batch_keys[-1])
    return query_batch_keys[-1]
  if not args.allow_skipping_queries:
    assert False, f"Invalid query input_len {input_len} > max prefill_len configured {query_batch_keys[-1]}."
  return -1


def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  return math.ceil(num_samples / mult * (sample_size / dataset_size))


def get_warmup_samples(dataset, args):
  query_batches = _init_query_batches(args)
  pandas_rows = tuple(dataset.iterrows())
  input_data = {}
  for sample_id, panda_row in enumerate(pandas_rows):
    padded, length = pad_tokens(panda_row[1].tok_input)
    input_data[sample_id] = offline_inference.InputData("", jnp.array(padded), length)
  for data in input_data.values():
    jax.block_until_ready(data.tokens)
  for sample_id in range(len(input_data)):
    group_idx = _classify_query(pandas_rows, sample_id, query_batches, args)
    if group_idx == -1:
      continue
    input_ = copy.copy(input_data[sample_id])
    input_.id = sample_id
    query_batches[group_idx].append(input_)

  interesting_buckets = [0, 16, 32, 64, 128, 256, 512, 1024]
  warmup_samples = _init_query_batches(args)
  for group_idx, group_val in query_batches.items():
    idx = int(math.log2(group_idx[0])) - 3
    for start, end in zip(interesting_buckets[:idx], interesting_buckets[1 : (idx + 1)]):
      for sample in group_val:
        if start < sample.true_length <= end:
          warmup_samples[group_idx].append(sample)
          break
    warmup_samples[group_idx].extend(group_val[:50])
  return warmup_samples


class SUT:
  """System under test for LoadGen."""

  def __init__(self, data, offline_inf_instances, args):
    self.offline_inf_instances = offline_inf_instances
    self._dataset = data
    self._queries = None
    self._processed_data = None
    self._sample_id_to_input = None
    self._query_batches = _init_query_batches(args)
    self.args = args

  def issue_queries(self, queries):
    log.info("Issue queries start")
    assert self._sample_id_to_input is not None
    self._processed_data = []
    self._queries = queries
    num_queries = len(queries)
    num_skipped_queries = 0
    log.info("Before Issue %d queries - classified queries %s", num_queries, str(list(map(len, self._query_batches.values()))))
    self._query_batches = _init_query_batches(self.args)
    for query in queries:
      group_idx = _classify_query(self.pandas_rows, query.index, self._query_batches, self.args)
      if group_idx == -1:
        num_skipped_queries += 1
      else:
        input_data = copy.copy(self._sample_id_to_input[query.index])
        input_data.id = query.id
        self._query_batches[group_idx].append(input_data)
    num_grouped_queries = list(map(len, self._query_batches.values()))
    log.info("Issue %d queries - classified queries %s num_skipped %d", num_queries, str(num_grouped_queries), num_skipped_queries)
    assert len(self._queries) - num_skipped_queries == sum(num_grouped_queries), (
        f"num_queries {num_queries} does not match num_grouped_queries {num_grouped_queries}"
    )
    log.info("Issue queries end")

  @timed("flush_queries")
  def flush_queries(self):
    log.info("Flush queries start")
    start = time.perf_counter()
    for group_idx, group in self._query_batches.items():
      log.info("Flush queries processing %s with %d samples", str(group_idx), len(group))
      self.offline_inf_instances[group_idx].init_decode_state()
      result = self.offline_inf_instances[group_idx].batch_inference(group, desc=f"batch-{group_idx}")
      self.offline_inf_instances[group_idx].decode_state = None
      for key, val in result.items():
        if not val:
          log.info("Value empty for key %s", key)
          continue
        key = int(key)
        lg.FirstTokenComplete([make_response(key, [val[0]])])
        lg.QuerySamplesComplete([make_response(key, val)])
    log.info("Flush queries end-start: %d", time.perf_counter() - start)
    gc.collect()

  def LoadSamplesToRam(self, sample_list):
    log.info("LoadSamplesToRam start")
    start = time.perf_counter()
    input_data = {}
    self.pandas_rows = list(self._dataset.iterrows())
    for sample_id in sample_list:
      padded, length = pad_tokens(self.pandas_rows[sample_id][1].tok_input)
      input_data[sample_id] = offline_inference.InputData("", jnp.array(padded), length)
    for data in input_data.values():
      jax.block_until_ready(data.tokens)
    self._sample_id_to_input = input_data
    log.info("LoadSamplesToRam finished: %ds", time.perf_counter() - start)

  def UnloadSamplesFromRam(self, sample_list):
    log.info("UnloadSamplesFromRam called")


def make_response(id_, response_token_ids):
  n_tokens = len(response_token_ids)
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  return lg.QuerySampleResponse(id_, response_data, response_size, n_tokens)


def _estimated_counts_by_bucket(dataset, args):
  total_len = dataset.tok_input_length + dataset.tok_output_length
  query_batches = _init_query_batches(args)
  prefix_lens = sorted(length for length, batch in query_batches.keys())
  mult = args.total_sample_count / len(dataset) * 1.05
  prev_len = 0
  total_count = 0
  estimates = {}
  for prefix_len in prefix_lens[:-1]:
    condition = (total_len <= 2 * prefix_len) & (dataset.tok_input_length <= prefix_len)
    count = len(dataset[condition])
    estimates[f"{prev_len}-{prefix_len}"] = math.ceil((count - total_count) * mult)
    total_count = count
  estimates[f">{prefix_lens[-1]}"] = math.ceil((len(dataset) - total_count) * mult)
  return estimates


def run_offline(config, args: OfflineModeArguments, *, devices=None):
  """Run the MLPerf offline benchmark with a public engine configuration."""
  gc.disable()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  if args.enable_profile:
    jax.profiler.start_server(args.jax_profiler_port)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline
  settings.FromConfig(args.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(args.user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", args.mlperf_conf)
  log.info("User config: %s", args.user_conf)

  ext = os.path.splitext(args.dataset_path)[1].lower()
  if ext == ".parquet":
    dataset = pd.read_parquet(args.dataset_path)
  elif ext == ".csv":
    dataset = pd.read_csv(args.dataset_path)
  elif ext in (".json", ".jsonl"):
    dataset = pd.read_json(args.dataset_path)
  else:
    raise ValueError(
        f"Unsupported dataset file format: {args.dataset_path}. "
        "Please use safe formats like Parquet (.parquet), CSV (.csv), or JSON/JSONL (.json/.jsonl)."
    )
  if args.rename_dataset_cols:
    rename_dict = json.loads(args.rename_dataset_cols)
    dataset.rename(columns=rename_dict, inplace=True)
    log.info("Renaming columns of dataset with mapping: %s", rename_dict)
  if args.total_sample_count < len(dataset):
    dataset = dataset.sample(n=args.total_sample_count)
  log.info("Dataset len %d, estimated counts by bucket %s", len(dataset), _estimated_counts_by_bucket(dataset, args))
  log.info("Prefill lengths and Batch sizes: %s", args.prefill_lengths_and_per_device_batch_sizes)
  log.info("Maxengine overrides: %s", args.engine_overrides)

  log.info("Get warmup samples")
  warmup_samples = get_warmup_samples(dataset, args)
  offline_inf_instances = {}
  params = None
  base_engine = None
  for group_idx in _init_query_batches(args):
    length, batch = group_idx
    log.info("Using batch size: %d and length: %d", batch, length)
    engine = create_engine_from_config(
        config, batch, length, 2 * length, devices=devices, **args.engine_overrides
    )
    offline_inf = offline_inference.OfflineInference(engine, params, base_engine, args.enable_batch_prefill)
    if params is None and offline_inf.params is not None:
      base_engine = engine
    params = offline_inf.params
    offline_inf_instances[group_idx] = offline_inf

  if not args.skip_warmup:
    with timed("warmup"):
      for group_idx in offline_inf_instances:
        length, batch = group_idx
        log.info("warm up for %d", length)
        offline_inf_instances[group_idx].warmup(length, warmup_samples[group_idx])
        offline_inf_instances[group_idx].decode_state = None
        gc.collect()

  sut = SUT(dataset, offline_inf_instances, args)
  if args.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
  elif args.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.Submission
    settings.print_timestamps = True
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True
  settings.use_token_latencies = True

  os.makedirs(args.output_log_dir, exist_ok=True)
  log.info("Logging to %s", args.output_log_dir)
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = args.output_log_dir
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = args.enable_log_trace

  loadgen_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  qsl = lg.ConstructQSL(len(dataset), args.total_sample_count, sut.LoadSamplesToRam, sut.UnloadSamplesFromRam)
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(loadgen_sut, qsl, settings, log_settings, args.audit_conf)
  log.info("query counts %s", str(list(map(len, sut._query_batches.values()))))
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(loadgen_sut)
  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)
  if args.enable_profile:
    jax.profiler.stop_server()
