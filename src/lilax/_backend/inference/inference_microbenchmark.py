"""Inference microbenchmark for prefill and autoregressive steps."""

import datetime
import json
import os
from collections.abc import MutableMapping

import jax

from lilax._backend.flops import flops
from lilax._backend.common import profiler
from lilax._backend.inference.maxengine import maxengine
from lilax._backend.input_pipeline.packing import prefill_packing
from lilax._backend.storage import gcs as gcs_utils
from lilax._backend.observability.diagnostics import pytree

_WARMUP_ITERS = 2
_FLATTEN_MICROBENCHMARK_RESULTS = False


def prefill_benchmark_loop(engine_prefill, params, tokens, true_length, iters, num_samples: int | None = None):
  """Run the timed prefill loop."""
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(iters):
    rng, rng_prefill = jax.random.split(rng)
    if num_samples is None:
      prefill_result, _ = engine_prefill(params, tokens, true_length, rng_prefill)
    else:
      prefill_result, _ = engine_prefill[num_samples](params, tokens, true_length, rng_prefill, None)
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  del prefill_result
  return (end - start).total_seconds()


def prefill_benchmark(config, engine_prefill, params, tokens, true_length, num_model_params, iters):
  """Warm up and measure prefill."""
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(_WARMUP_ITERS):
    rng, rng_prefill = jax.random.split(rng)
    prefill_result, _ = engine_prefill(params, tokens, true_length, rng_prefill)
  jax.block_until_ready(prefill_result)
  del prefill_result
  print(f"Prefill benchmark results for length {tokens.size}:\n")
  time_in_s = prefill_benchmark_loop(engine_prefill, params, tokens, true_length, iters)
  prefill_average_ms = 1000 * time_in_s / iters
  prefill_tflops_per_device, _, _ = flops.calculate_prefill_tflops_per_device(
      num_model_params, tokens.size, config
  )
  tflops_per_sec_per_device = prefill_tflops_per_device / prefill_average_ms * 1000.0
  print(
      f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
      f"\tPrefill total TFLOPs/device: {prefill_tflops_per_device:.3f}\n"
      f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n\n\n\n"
  )
  return {
      "time_in_ms": prefill_average_ms,
      "total_tflops_per_device": prefill_tflops_per_device,
      "tflops_per_sec_per_device": tflops_per_sec_per_device,
  }


def prefill_multisampling_benchmark(config, engine_prefill_multisampling, params, tokens, true_length, iters):
  """Warm up and measure multisampling prefill."""
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(_WARMUP_ITERS):
    rng, rng_prefill = jax.random.split(rng)
    for num_samples in config.inference_microbenchmark_num_samples:
      prefill_result, _ = engine_prefill_multisampling[num_samples](params, tokens, true_length, rng_prefill, None)
  jax.block_until_ready(prefill_result)
  del prefill_result
  print(f"Multi-sampling prefill benchmark results for length {tokens.size}:\n")
  result_dict = {}
  for num_samples in config.inference_microbenchmark_num_samples:
    time_in_s = prefill_benchmark_loop(engine_prefill_multisampling, params, tokens, true_length, iters, num_samples)
    average_ms = 1000 * time_in_s / iters
    print(f"\nNum samples: {num_samples}\n\tPrefill step average time: {average_ms:.3f} ms\n\n\n\n")
    result_dict[num_samples] = {"time_in_ms": average_ms}
  return result_dict


def prefill_insert_benchmark_loop(config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters,
                                  profile_name):
  """Run the profiled prefill and insert loop."""
  prof = profiler.Profiler(config)
  prof.activate(optional_postfix=profile_name)
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  rng, _ = jax.random.split(rng)
  for index in range(iters):
    _, decode_state = engine_insert(params, tokens, int(index % total_slots), true_length, decode_state, rng)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  prof.deactivate()
  return (end - start).total_seconds(), decode_state


def prefill_insert_benchmark(config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters):
  """Warm up and measure prefill and insert."""
  rng = jax.random.PRNGKey(1234)
  rng, _ = jax.random.split(rng)
  for index in range(_WARMUP_ITERS):
    _, decode_state = engine_insert(params, tokens, int(index % total_slots), true_length, decode_state, rng)
  jax.block_until_ready(decode_state)
  print(f"Prefill and insert benchmark results for length {tokens.size}:\n")
  time_in_s, decode_state = prefill_insert_benchmark_loop(
      config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters, f"prefill_insert_{tokens.size}"
  )
  average_ms = time_in_s / iters * 1000.0
  print(f"\tPrefill + Insert step average time: {average_ms:.3f} ms\n\n\n\n")
  return {"time_in_ms": average_ms}, decode_state


def ar_benchmark_loop(config, engine_generate, params, decode_state, iters, profile_name):
  """Run the profiled autoregressive loop."""
  prof = profiler.Profiler(config)
  prof.activate(optional_postfix=profile_name)
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  for _ in range(iters):
    rng, rng_generate = jax.random.split(rng)
    decode_state, _ = engine_generate(params, decode_state, rng_generate)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  prof.deactivate()
  return (end - start).total_seconds(), decode_state


def ar_benchmark(config, engine_generate, params, decode_state, global_batch_size, cache_size, model_size, iters):
  """Warm up and measure autoregressive generation."""
  rng = jax.random.PRNGKey(1234)
  for _ in range(_WARMUP_ITERS):
    rng, rng_generate = jax.random.split(rng)
    decode_state, _ = engine_generate(params, decode_state, rng_generate)
  jax.block_until_ready(decode_state)
  time_in_s, decode_state = ar_benchmark_loop(config, engine_generate, params, decode_state, iters, "autoregress")
  seconds_per_step = time_in_s / iters
  average_ms = seconds_per_step * 1000
  total_throughput = global_batch_size / seconds_per_step
  gb_per_step_per_device = (model_size + cache_size) / 1e9 / jax.device_count()
  bw_per_device = gb_per_step_per_device / seconds_per_step
  print(
      f"AutoRegressive results:\n"
      f"\tAR step average time: {average_ms:.3f} ms\n"
      f"\tAR step average time per seq: {average_ms/global_batch_size:.3f} ms\n"
      f"\tAR global batch size: {global_batch_size}\n"
      f"\tAR throughput: {total_throughput:.3f} tokens/second\n"
      f"\tAR memory bandwidth per device: {bw_per_device:.3f} GB/s\n\n\n"
  )
  return {
      "step_in_ms": average_ms,
      "step_in_ms_per_seq": average_ms / global_batch_size,
      "global_batch_size": global_batch_size,
      "total_throughput_tokens_per_second": total_throughput,
      "bw_per_device_GB_per_second": bw_per_device,
  }, decode_state


def collate_results(config, results, model_size, cache_size, num_model_params, incl_config=False):
  """Add model and cache sizes and optionally configuration."""
  results["sizes"] = {
      "model_size_in_gb": model_size / 1e9,
      "cache_size_in_gb": cache_size / 1e9,
      "model_params_in_billions": num_model_params / 1e9,
  }
  if incl_config:
    results["config"] = {}
    for key, value in dict(config.get_keys()).items():
      results["config"][key] = str(value) if key == "dtype" else value
  return results


def flatten_dict(dictionary, prefix="", sep="_"):
  results = []
  for key, value in dictionary.items():
    new_key = str(prefix) + sep + str(key) if prefix else key
    if isinstance(value, MutableMapping):
      results.extend(flatten_dict(value, new_key, sep=sep).items())
    else:
      results.append((new_key, value))
  return dict(results)


def write_results(results, filename, flatten_microbenchmark_results):
  """Write benchmark results as JSON."""
  if flatten_microbenchmark_results:
    results["flattened_results"] = flatten_dict(results)
  if filename:
    with open(filename, "wt", encoding="utf-8") as output:
      json.dump(results, output, indent=2)
  return results


def upload_results_to_gcs(results_file_name, destination_gcs_name):
  """Upload a results file to GCS."""
  gcs_utils.upload_blob(destination_gcs_name, results_file_name)


def print_results_for_analyze(results):
  """Print results for analyze_sharegpt.py."""
  print("\nFor usage in analyze_sharegpt.py :")
  if "prefill" in results:
    prefill = {int(key): round(value["time_in_ms"], 3) for key, value in results["prefill"].items()}
    print(f"PREFILL_BUCKET_SIZE_TO_MS = {prefill}")
  if "prefill-multisampling" in results:
    multisampling = {}
    for prefill_length, result_dict in results["prefill-multisampling"].items():
      multisampling[int(prefill_length)] = {
          num_samples: round(value["time_in_ms"], 3) for num_samples, value in result_dict.items()
      }
    print(f"MULTISAMPLING_PREFILL_BUCKET_SIZE_TO_MS = {multisampling}")
  if "insert" in results:
    insert = {int(key): round(value["time_in_ms"], 3) for key, value in results["insert"].items()}
    print(f"INSERT_BUCKET_SIZE_TO_MS = {insert}")
  if "autoregressive" in results:
    print(f"SYSTEM_TIME_PER_DECODE_TOKEN_MS = {results['autoregressive']['step_in_ms_per_seq']}")


def summarize_prefill_result(engine_prefill, params, tokens, true_length):
  """Summarize the prefill logits and cache."""
  print(f"Prefill result of length {tokens.size}:\n")
  rng = jax.random.PRNGKey(1234)
  prefill_result, _ = engine_prefill(params, tokens, true_length, rng)
  jax.block_until_ready(prefill_result)
  num_logits, total_logits, avg_logits = pytree.summarize_pytree_data(
      prefill_result["logits"], name="Prefill Logits", raw=True
  )
  num_cache, total_cache, avg_cache = pytree.summarize_pytree_data(prefill_result["cache"], name="Prefill Cache")
  del prefill_result
  return {
      "num_logits_params": num_logits,
      "total_logits_size": total_logits,
      "avg_logits_param_size": avg_logits,
      "num_cache_params": num_cache,
      "total_cache_size": total_cache,
      "avg_cache_param_size": avg_cache,
  }


def run_benchmarks(config):
  """Run microbenchmarks using an initialized native configuration."""
  engine = maxengine.MaxEngine(config)
  prefill_processor = prefill_packing.PrefillProcessor(engine)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)
  prefill_lengths = [int(length) for length in config.inference_microbenchmark_prefill_lengths.split(",")]
  stages_to_benchmark = config.inference_microbenchmark_stages.split(",")
  benchmark_loop_iters = config.inference_microbenchmark_loop_iters

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  rng, rng_init_decode = jax.random.split(rng)
  generate_executable, params, decode_state_executable = engine.aot_compile(params, pass_rng_shape=True)
  decode_state = decode_state_executable(rng_init_decode)

  _, cache_size, _ = pytree.summarize_pytree_data(decode_state["cache"], name="Cache")
  num_model_params, model_size, _ = pytree.summarize_pytree_data(params, name="Model")

  benchmark_results = {}
  if "prefill" in stages_to_benchmark:
    benchmark_results["prefill-result-sizes"] = {}
    benchmark_results["prefill"] = {}
    benchmark_results["insert"] = {}
    prefill_tokens = {}
    prefill_true_lengths = {}
    prefill_executable = {}
    prefill_insert_executable = {}
    i32_scalar = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))

    for prefill_length in prefill_lengths:
      is_bos = tokenizer_model.bos_id is not None
      prefill_tokens[prefill_length], prefill_true_lengths[prefill_length] = tokenizer_model.encode(
          text, is_bos=is_bos, prefill_lengths=[prefill_length]
      )
      key_shape = jax.ShapeDtypeStruct([prefill_length], jax.numpy.dtype("int32"))
      prefill_executable[prefill_length] = (
          jax.jit(engine.prefill_aot, in_shardings=(engine.param_layouts, None, None, None))
          .lower(params, key_shape, i32_scalar, rng_shape)
      ).compile(compiler_options=None)
      prefill_insert_executable[prefill_length] = prefill_processor.aot_compile(params, prefill_length)
      benchmark_results["prefill-result-sizes"][prefill_length] = summarize_prefill_result(
          prefill_executable[prefill_length], params, prefill_tokens[prefill_length], prefill_true_lengths[prefill_length]
      )

    for prefill_length in prefill_lengths:
      benchmark_results["prefill"][prefill_length] = prefill_benchmark(
          config, prefill_executable[prefill_length], params, prefill_tokens[prefill_length],
          prefill_true_lengths[prefill_length], num_model_params, benchmark_loop_iters,
      )
      prefill_insert_time, decode_state = prefill_insert_benchmark(
          config, prefill_insert_executable[prefill_length], decode_state, params, engine.max_concurrent_decodes,
          prefill_tokens[prefill_length], prefill_true_lengths[prefill_length], benchmark_loop_iters,
      )
      benchmark_results["insert"][prefill_length] = {}
      benchmark_results["insert"][prefill_length]["time_in_ms"] = (
          prefill_insert_time["time_in_ms"] - benchmark_results["prefill"][prefill_length]["time_in_ms"]
      )

  if "prefill-multisampling" in stages_to_benchmark:
    benchmark_results["prefill-multisampling"] = {}
    multisampling_prefill_executable = {}
    i32_scalar = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))
    for prefill_length in prefill_lengths:
      key_shape = jax.ShapeDtypeStruct([prefill_length], jax.numpy.dtype("int32"))
      multisampling_prefill_executable[prefill_length] = {}
      for num_samples in config.inference_microbenchmark_num_samples:
        multisampling_prefill_executable[prefill_length][num_samples] = (
            jax.jit(
                engine.prefill_multisampling_aot,
                in_shardings=(engine.param_layouts, None, None, None, None),
                static_argnames=("num_samples",),
            ).lower(params, key_shape, i32_scalar, rng_shape, num_samples, None)
        ).compile(compiler_options=None)

    for prefill_length in prefill_lengths:
      benchmark_results["prefill-multisampling"][prefill_length] = prefill_multisampling_benchmark(
          config, multisampling_prefill_executable[prefill_length], params, prefill_tokens[prefill_length],
          prefill_true_lengths[prefill_length], benchmark_loop_iters,
      )

  if "generate" in stages_to_benchmark:
    benchmark_results["autoregressive"], decode_state = ar_benchmark(
        config, generate_executable, params, decode_state, engine.max_concurrent_decodes,
        cache_size, model_size, benchmark_loop_iters,
    )

  results = collate_results(config, benchmark_results, model_size, cache_size, num_model_params)
  print_results_for_analyze(results)
  if config.inference_microbenchmark_log_file_path:
    write_results(results, config.inference_microbenchmark_log_file_path, _FLATTEN_MICROBENCHMARK_RESULTS)
  if config.gcs_metrics:
    metrics_filename = f"{config.run_name}_results.txt"
    write_results(results, metrics_filename, _FLATTEN_MICROBENCHMARK_RESULTS)
    gcs_filename = os.path.join(config.base_output_directory, metrics_filename)
    upload_results_to_gcs(metrics_filename, gcs_filename)
  return results


def run_benchmarks_with_unsafe_rbg(config):
  """Select unsafe RBG and run with an initialized native configuration."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  return run_benchmarks(config)
