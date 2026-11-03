"""Native MLPerf offline inference capabilities."""

import json
from pathlib import Path

from lilax._backend.inference.mlperf.offline_inference import InputData, OfflineInference
from lilax._backend.inference.mlperf.offline_mode import OfflineModeArguments, run_offline
from lilax.generation.engine import create_engine_from_config
from lilax.benchmarks.mlperf.accuracy import evaluate_accuracy, evaluate_accuracy_fast


def offline_arguments(dataset_path, *, model="llama2-70b", test_run=False, **overrides):
  """Build offline arguments using the source model-specific benchmark defaults.

  Supply engine checkpoint and tokenizer paths through the configuration or
  ``engine_overrides``. Other keyword arguments replace OfflineModeArguments defaults.
  """
  defaults = {
      "llama2-70b": {
          "total_sample_count": 1000 if test_run else 24576,
          "prefill_lengths_and_per_device_batch_sizes": "256,216|512,108|1024,54",
          "engine_overrides": {
              "model_name": "llama2-70b", "quantization": "int8", "quantize_kvcache": True,
              "checkpoint_is_quantized": True, "skip_jax_distributed_system": True,
          },
      },
      "mixtral-8x7b": {
          "total_sample_count": 100 if test_run else 15000,
          "prefill_lengths_and_per_device_batch_sizes": "256,144|512,72|2048,18",
          "rename_dataset_cols": json.dumps({"tok_input_len": "tok_input_length", "tok_ref_output_len": "tok_output_length"}),
          "engine_overrides": {
              "model_name": "mixtral-8x7b", "quantization": "int8", "quantize_kvcache": True,
              "checkpoint_is_quantized": True, "compute_axis_order": "0,2,1,3", "ar_cache_axis_order": "0,2,1,3",
              "megablox": False, "sparse_matmul": False, "capacity_factor": 1, "model_call_mode": "inference",
          },
      },
  }[model]
  defaults.update(audit_conf="no_audit", tok_outlen_multiplier=2.5)
  defaults.update(overrides)
  return OfflineModeArguments(dataset_path=dataset_path, **defaults)


def run_offline_workflow(
    config, args, *, model="llama2-70b", checkpoint_path=None, tokenizer_path=None, fast_eval=False,
    accuracy_options=None, devices=None
):
  """Run LoadGen and score an existing accuracy log for an accuracy run.

  For Mixtral accuracy, pass its tokenizer path. The scorer's result is returned;
  performance runs and accuracy runs without a log return None.
  """
  run_offline(config, args, devices=devices)
  accuracy_log = Path(args.output_log_dir) / "mlperf_log_accuracy.json"
  if args.mlperf_test_mode == "accuracy" and accuracy_log.exists():
    if model == "llama2-70b" and checkpoint_path is None:
      checkpoint_path = "meta-llama/Llama-2-70b-chat-hf"
    options = accuracy_options or {}
    if fast_eval:
      return evaluate_accuracy_fast(
          str(accuracy_log), args.dataset_path, checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path,
          **options,
      )
    return evaluate_accuracy(checkpoint_path, str(accuracy_log), args.dataset_path, **options)

__all__ = [
    "InputData", "OfflineInference", "OfflineModeArguments", "create_engine_from_config", "offline_arguments",
    "run_offline", "run_offline_workflow",
]
