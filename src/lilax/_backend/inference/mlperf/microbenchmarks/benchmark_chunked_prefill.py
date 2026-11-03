"""Chunked-prefill and prefix-cache microbenchmarks."""

import datetime
import functools
import os

import jax
from jetstream.core import prefix_cache
from jetstream.engine import chunked_prefill

from lilax._backend.configs.runtime import resolve_runtime_config
from lilax._backend.inference.maxengine import maxengine
from lilax._backend.utils.diagnostics import system_info
from lilax.configuration import MaxTextConfig, RLConfig
from lilax.runtime import Runtime

_WARMUP_ITERS = 2
_BENCHMARK_ITERS = 5


@jax.jit
def _copy(cache):
  def _array_copy(array):
    return array.copy()
  return jax.tree.map(_array_copy, cache)


def _run_benchmark_loop(func_to_benchmark, iters, label):
  print(f"\nStarting warmup for {label}...")
  for index in range(_WARMUP_ITERS):
    start = datetime.datetime.now()
    result = func_to_benchmark()
    jax.block_until_ready(result)
    end = datetime.datetime.now()
    print(f"  Warmup iteration {index+1} time: {end - start}")
  print(f"\nStarting benchmark for {label}...")
  total_time = datetime.timedelta()
  for index in range(iters):
    start = datetime.datetime.now()
    result = func_to_benchmark()
    jax.block_until_ready(result)
    end = datetime.datetime.now()
    iter_time = end - start
    total_time += iter_time
    print(f"  Benchmark iteration {index+1} time: {iter_time}")
  average_time = total_time / iters
  print(f"\nAverage time taken for {label} over {iters} iterations: {average_time}")
  return average_time


def benchmark_chunked_prefill(engine, params, chunked_tokens_list):
  def run_chunked_prefill_utility():
    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine, prefill_params=params, chunked_tokens_list=chunked_tokens_list,
    )
    return prefill_result
  return _run_benchmark_loop(run_chunked_prefill_utility, _BENCHMARK_ITERS, "standard chunked prefill")


def fill_prefix_cache(prefix_cache_inst, prefix, cache_num, hit_tokens, padded_length):
  true_length = len(hit_tokens)
  key_to_hit = tuple(hit_tokens.tolist())

  def copy_prefix():
    return _copy(prefix)

  print("Filling cache with", cache_num, "dummy entries...")
  for index in range(cache_num):
    dummy_key = tuple(int(token) + 1000 + index * true_length for token in key_to_hit)
    dummy_value_with_key = prefix_cache.Value(
        prefix=copy_prefix(), true_length=true_length, padded_length=padded_length, tokens=dummy_key,
    )
    prefix_cache_inst.save(dummy_key, dummy_value_with_key)
  print("Finished filling cache with", cache_num, "dummy entries.")
  print("Adding the target entry with key length ", len(key_to_hit), "...", sep="")
  value_to_hit = prefix_cache.Value(
      prefix=copy_prefix(), true_length=true_length, padded_length=padded_length, tokens=key_to_hit,
  )
  prefix_cache_inst.save(key_to_hit, value_to_hit)
  jax.effects_barrier()
  print("Finished adding the target entry.")


def create_prefix_cache(engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
                        prefix_caching_dram_byte, max_prefill_length):
  print("\n--- Creating and Populating Prefix Cache ---")
  prefill_result, _ = chunked_prefill.do_chunked_prefill(
      prefill_engine=engine, prefill_params=params, chunked_tokens_list=chunked_tokens_list,
  )
  prefix_cache_inst = prefix_cache.PrefixCache(prefix_caching_hbm_byte, prefix_caching_dram_byte)
  prefill_result_cache_byte_size = jax.tree.reduce(lambda acc, array: acc + array.nbytes, prefill_result["cache"], 0)
  cache_entries_num = prefix_caching_dram_byte // prefill_result_cache_byte_size
  fill_prefix_cache(prefix_cache_inst, prefill_result["cache"], cache_entries_num, tokens, max_prefill_length)
  return prefix_cache_inst


def benchmark_prefix_cache_loop(engine, params, tokens, chunked_tokens_list, prefix_cache_inst, chunk_size, run_time):
  print("\n--- Starting Prefix Cache Benchmark ---")

  def run_chunked_prefill_with_prefix_caching(cache_hit_chunk, need_save):
    tokens_list = tokens.tolist()
    existing_prefix, _ = prefix_cache.load_existing_prefix_and_get_remain_tokens(prefix_cache_inst, tokens, chunk_size)
    assert existing_prefix is not None, "Should hit in benchmark"
    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine,
        prefill_params=params,
        chunked_tokens_list=chunked_tokens_list[cache_hit_chunk:],
        existing_prefix=existing_prefix if cache_hit_chunk > 0 else None,
    )
    if need_save:
      run_time[0] += 1
      prefix_cache_inst.save(
          tuple(tokens_list + [run_time[0]]),
          prefix_cache.Value(
              prefix=_copy(prefill_result["cache"]), true_length=len(tokens_list),
              padded_length=len(tokens_list), tokens=tuple(tokens_list),
          ),
      )
    return prefill_result

  for cache_hit_chunk in range(len(chunked_tokens_list)):
    for need_save in [True, False]:
      label = f"prefix caching (hit_chunks={cache_hit_chunk}, save={need_save})"
      benchmark_func = functools.partial(
          run_chunked_prefill_with_prefix_caching, cache_hit_chunk=cache_hit_chunk, need_save=need_save,
      )
      _run_benchmark_loop(benchmark_func, _BENCHMARK_ITERS, label)


def benchmark_prefix_cache(engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
                           prefix_caching_dram_byte, chunk_size, max_prefill_length):
  print("\n--- Starting Prefix Cache Benchmark ---")
  prefix_cache_inst = create_prefix_cache(
      engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
      prefix_caching_dram_byte, max_prefill_length,
  )
  run_time = [0]
  benchmark_prefix_cache_loop(engine, params, tokens, chunked_tokens_list, prefix_cache_inst, chunk_size, run_time)


def prepare_setting(config):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  devices = None
  if isinstance(config, Runtime):
    config.resolve_config()
    devices = config.devices
    config = config.config
  elif isinstance(config, (MaxTextConfig, RLConfig)):
    config = resolve_runtime_config(config)
  system_info.print_system_information()
  prefix_caching_hbm_byte = config.prefix_caching_hbm_byte
  prefix_caching_dram_byte = config.prefix_caching_dram_byte
  max_prefill_length = config.max_prefill_predict_length
  text = config.prompt
  chunk_size = config.prefill_chunk_size
  engine = maxengine.MaxEngine(config, devices=devices)
  if not engine.use_chunked_prefill:
    raise ValueError("Engine must be configured with use_chunked_prefill=True")
  params = engine.load_params()
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, _ = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[max_prefill_length])
  chunked_tokens_list = chunked_prefill.gen_chunked_padded_tokens(
      tokens=tokens, chunk_size=chunk_size, tokenizer=tokenizer_model, jax_padding=True,
  )
  return (engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
          prefix_caching_dram_byte, chunk_size, max_prefill_length)


def run_chunked_prefill_benchmark(config):
  (engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
   prefix_caching_dram_byte, chunk_size, max_prefill_length) = prepare_setting(config)
  benchmark_chunked_prefill(engine, params, chunked_tokens_list)
  benchmark_prefix_cache(
      engine, params, tokens, chunked_tokens_list, prefix_caching_hbm_byte,
      prefix_caching_dram_byte, chunk_size, max_prefill_length,
  )
