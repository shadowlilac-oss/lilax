# Measure or estimate an inference workload

Use microbenchmarks to run inference stages. Use ShareGPT analysis to estimate a conversation workload from timing data.

## Run the configured inference stages

This operation compiles and runs accelerator work and loads the configured checkpoint into a separate inference engine. Replace the path with a parameter checkpoint matching the selected model.

```python
from lilax.benchmarks.inference.inference_benchmarks import run_benchmarks
from lilax import AutoConfig, AutoModelForCausalLM
from lilax.runtime import Runtime

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_config(config)
source = model.runtime.public_config.model_copy(update={
	"load_parameters_path": "/checkpoints/qwen3-8b/0/items", "prompt": "Explain checkpoints.",
})
runtime = Runtime(source)
runtime.resolve_config()
results = run_benchmarks(runtime.config)
```

This does not reuse the model wrapper's live parameters. The public [Omni runtime loader](../components/omni-checkpoints.md) is an alternative for its stitched checkpoint workflow.

The resolved configuration selects stages, prefill lengths, iteration count, and output paths. `results` is a dictionary containing `sizes` and the selected stage results, such as `prefill` and `autoregressive`. See [benchmark arguments](../../reference/benchmarks/inference-benchmarks.md).

See [benchmark source fields](../../reference/execution/runtime-source-fields-inference.md#inference-microbenchmarks) for exact keys, types, defaults, and stage dependencies.

## Estimate a ShareGPT file

```python
from lilax.benchmarks.inference.sharegpt import analyze_sharegpt

input_tokens, output_tokens, total_seconds, prefill_seconds, generation_seconds = analyze_sharegpt(
	"/data/sharegpt.json", mb_timing_file="/results/sharegpt-timings.json",
)
```

The timing file contains the analyzer's expected `Prefill` and `AutoRegressive` keys. Its format is distinct from the current microbenchmark's lowercase result structure. The analyzer estimates token counts from word counts. Its output is an estimate, not a measured model run.

The conversation file is a JSON list with records such as `{"conversations": [{"value": "What is a checkpoint?"}, {"value": "A saved model state."}]}`. Supply actual timing values in `Prefill["128"]["prefill_time_in_ms"]` and the other required length buckets, plus `AutoRegressive["ar_step_in_ms_per_seq"]`. The five returned values are token counts followed by estimated durations in seconds.
