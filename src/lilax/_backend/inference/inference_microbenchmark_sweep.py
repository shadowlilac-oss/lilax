"""Sweep across inference microbenchmarks with an initialized native configuration."""

import json
import os

import jax
import jsonlines

from lilax._backend.configs.runtime import with_inference_sweep_overrides
from lilax._backend.inference import inference_microbenchmark


def run_sweep(config):
  """Run the configurations specified by the inference metadata file."""
  base_run_name = config.run_name
  with open(config.inference_metadata_file, "rt", encoding="utf-8") as json_file:
    inference_metadata = json.load(json_file)
    print(f"inference_metadata: {inference_metadata}")

  two_axis_order_product_id_list = inference_metadata["two_axis_order_product_id_list"].split(":")
  prefill_cache_axis_order_list = inference_metadata["prefill_cache_axis_order_list"].split(":")
  ar_cache_axis_order_list = inference_metadata["ar_cache_axis_order_list"].split(":")
  start_two_axis_order_product_id = two_axis_order_product_id_list[0]
  end_two_axis_order_product_id = two_axis_order_product_id_list[-1]

  results = []
  for two_axis_order_product_id, prefill_cache_axis_order, ar_cache_axis_order in zip(
      two_axis_order_product_id_list, prefill_cache_axis_order_list, ar_cache_axis_order_list
  ):
    print(f"two_axis_order_product_id {two_axis_order_product_id}")
    print(f"prefill_cache_axis_order {prefill_cache_axis_order}")
    print(f"ar_cache_axis_order {ar_cache_axis_order}")
    run_tag = (
        f"{two_axis_order_product_id}-{prefill_cache_axis_order.replace(',','')}-{ar_cache_axis_order.replace(',','')}"
    )
    run_name = f"{base_run_name}/{run_tag}"
    tensorboard_dir = os.path.join(config.base_output_directory, run_name, "tensorboard", "")
    run_config = with_inference_sweep_overrides(
        config, prefill_cache_axis_order=prefill_cache_axis_order, ar_cache_axis_order=ar_cache_axis_order,
        tensorboard_dir=tensorboard_dir, run_name=run_name,
    )
    dimensions_json = {
        "base_output_directory": run_config.base_output_directory,
        "model_name": run_config.model_name,
        "tokenizer": run_config.tokenizer_path,
        "weight_dtype": run_config.weight_dtype,
        "inference_microbenchmark_prefill_lengths": f"{run_config.inference_microbenchmark_prefill_lengths}",
        "inference_microbenchmark_stages": run_config.inference_microbenchmark_stages,
        "inference_microbenchmark_loop_iters": f"{run_config.inference_microbenchmark_loop_iters}",
        "max_prefill_predict_length": f"{run_config.max_prefill_predict_length}",
        "max_target_length": f"{run_config.max_target_length}",
        "per_device_batch_size": f"{run_config.per_device_batch_size}",
        "ici_fsdp_parallelism": f"{run_config.ici_fsdp_parallelism}",
        "ici_autoregressive_parallelism": f"{run_config.ici_autoregressive_parallelism}",
        "ici_tensor_parallelism": f"{run_config.ici_tensor_parallelism}",
        "profiler": f"{run_config.profiler}",
        "scan_layers": f"{run_config.scan_layers}",
        "quantization": run_config.quantization,
        "quantize_kvcache": f"{run_config.quantize_kvcache}",
        "attention": run_config.attention,
        "two_axis_order_product_id": f"{two_axis_order_product_id}",
        "prefill_cache_axis_order": f"{prefill_cache_axis_order}",
        "ar_cache_axis_order": f"{ar_cache_axis_order}",
        "compute_axis_order": f"{run_config.compute_axis_order}",
        "reshape_q": f"{run_config.reshape_q}",
        "kv_quant_axis": f"{run_config.kv_quant_axis}",
        "run_name": f"{run_name}",
        "run_tag": f"{run_tag}",
        "config_json_string": json.dumps(
            run_config.get_keys(), default=lambda x: f"<<non-serializable: {type(x).__qualname__}>>"
        ),
    }
    dimensions_json = {**dimensions_json, **inference_metadata}
    try:
      microbenchmark_results = inference_microbenchmark.run_benchmarks_with_unsafe_rbg(run_config)
      if microbenchmark_results:
        metrics = microbenchmark_results["flattened_results"]
        metrics = {k.lower(): v for k, v in metrics.items()}
      else:
        metrics = {}
      dimensions_json["oom"] = "False"
      print(
          f"Completed run {two_axis_order_product_id} out of: "
          f"{start_two_axis_order_product_id} to {end_two_axis_order_product_id}"
      )
    except jax.errors.JaxRuntimeError:
      metrics = {}
      dimensions_json["oom"] = "True"
      print(
          f"Failed at run {two_axis_order_product_id} out of: "
          f"{start_two_axis_order_product_id} to {end_two_axis_order_product_id}"
      )
    final = {"metrics": metrics, "dimensions": dimensions_json}
    print(f"Result: {final}")
    results.append(final)

  print(f"All results {results}")
  path = "inference_microbenchmark_sweep_results.jsonl"
  with jsonlines.open(path, mode="w") as writer:
    writer.write_all(results)
  return results
