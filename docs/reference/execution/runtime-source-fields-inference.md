# Scoring and benchmark source fields

These fields extend the [runtime-source workflow reference](runtime-source-fields.md). Defaults below are declarations. Model configuration and execution profiles can supply different effective values.

## vLLM scoring engine

`score(source, **options)` resolves the runtime source and constructs a separate vLLM engine. The source supplies checkpoint and tokenizer selection, prompt and system text, chat-template selection, and the engine fields below. Without `load_parameters_path`, vLLM uses dummy loading. A wrapper's deferred Hub weights are not the engine's weight source.

The engine also reads `max_target_length`, `weight_dtype`, `scan_layers`, `use_multimodal`, `use_mrope`, and `debug_sharding` from the model-produced configuration. `debug_sharding` is a boolean with declared default `False` and prints weight placement. `ici_tensor_parallelism` and `ici_data_parallelism` select tensor and data parallel sizes. `ici_expert_parallelism > 1` enables expert parallelism. Their [profile fields](execution-profile-placement.md) describe placement defaults. Keep model identity and layout from the public producer rather than changing them to load unrelated weights.

Scoring request controls are keyword arguments, not runtime-source fields. Their accepted options are listed under [score](#score). `top_k` and `repeat` must be positive. `method` accepts `auto`, `top_k`, or `exact`.

`prompts_file` supplies JSONL records with `prompt` and optional `candidates`. When supplied, those prompts replace configured prompt construction. Text candidates must encode as exactly one next token at the prompt boundary. Integer candidates must be valid vocabulary IDs. `output_file` writes JSONL results. See [configured scoring](../components/configured-decoding.md) for result records and candidate-method behavior.

Prefused MoE weights must match model execution and weight layout.

<details>
<summary>Parameters</summary>

- `vllm_hf_config_path` (`str`, optional, defaults to `""`). HF configuration path for the vLLM implementation.
- `vllm_hf_overrides` (`dict[str, Any]`, optional, defaults to `{}`). Overrides for vLLM's HF configuration.
- `hbm_utilization_vllm` (`float`, optional, defaults to `0.72`). Target device-memory fraction in `(0, 1]`.
- `async_scheduling` (`bool`, optional, defaults to `False`). Asynchronous scheduling.
- `max_num_batched_tokens` (`int | None`, optional, defaults to `None`). Token capacity when set.
- `max_num_seqs` (`int | None`, optional, defaults to `None`). Sequence capacity when set.
- `enable_dp_attention` (`bool`, optional, defaults to `False`). Attention data parallelism.
- `prefuse_moe_weights` (`bool`, optional, defaults to `False`). Prefused MoE weight layout.

</details>

### score

Import from `lilax.serving.vllm_score`.

```python
score(config, **options)
```

Resolves a runtime-source configuration, constructs a vLLM scoring engine, and optionally writes JSONL. It sets process-wide JAX and serving environment options. The [configured scoring reference](../components/configured-decoding.md) describes those effects and the result records.

Returns a list of scoring records from the last repetition. Invalid methods, nonpositive top-k, or nonpositive repeat count raise `ValueError`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Runtime-source configuration from `model.runtime.public_config` with the checkpoint, tokenizer, prompt, and engine settings above. See [configuration objects](runtime.md#configuration-objects).
- `**options` (`Any`). Variadic keyword arguments accepting the following forwarded request controls.
- `candidates` (`Any`, optional, defaults to `""`). Comma-separated text or a sequence of text candidates or integer token IDs. `None` supplies no candidates.
- `prompts_file` (`str`, optional, defaults to `""`). JSONL prompt records replacing configured prompt construction.
- `output_file` (`str`, optional, defaults to `""`). JSONL result destination. Empty omits this write.
- `top_k` (`int`, optional, defaults to `20`). Requested log-probability count.
- `method` (`str`, optional, defaults to `"auto"`). `auto`, `top_k`, or `exact` candidate scoring.
- `add_special_tokens` (`bool`, optional, defaults to `False`). Tokenizer special-token inclusion.
- `enable_prefix_caching` (`bool`, optional, defaults to `False`). Engine prefix caching.
- `repeat` (`int`, optional, defaults to `1`). Number of scoring repetitions.
- `enable_thinking` (`bool`, optional, defaults to `False`). Thinking selection for prompt construction.

</details>

## Inference microbenchmarks

For example, this continues the source setup on the [runtime-source page](runtime-source-fields.md). It selects only the prefill measurement and writes its result to a caller-selected path.

```python
from lilax.benchmarks.inference.inference_benchmarks import run_benchmarks
from lilax.runtime import Runtime

benchmark_source = source.model_copy(update={
	"inference_microbenchmark_stages": "prefill",
	"inference_microbenchmark_log_file_path": "./prefill-results.json",
})
results = run_benchmarks(Runtime(benchmark_source).resolve_config().config)
```

This loads weights, compiles functions, and executes device work. It is not a predicted result or hardware recommendation.

Multisampling requires the `prefill` stage to prepare its inputs.

<details>
<summary>Parameters</summary>

- `inference_microbenchmark_prefill_lengths` (`str`, optional, defaults to `"64,128,256,512,1024"`). Comma-separated lengths to compile and measure.
- `inference_microbenchmark_stages` (`str`, optional, defaults to `"prefill,generate"`). Comma-separated `prefill`, `generate`, or `prefill-multisampling` stages.
- `inference_microbenchmark_loop_iters` (`int`, optional, defaults to `10`). Iterations per loop.
- `inference_microbenchmark_log_file_path` (`str`, optional, defaults to `""`). Local result path. Empty omits this write.
- `inference_microbenchmark_num_samples` (`list[int]`, optional, defaults to `[1, 2, 3, 4, 5]`). Multisampling counts.
- `inference_metadata_file` (`str`, optional, defaults to `""`). JSON metadata path required by `run_sweep`.
- `inference_benchmark_test` (`bool`, optional, defaults to `False`). Benchmark-test mesh selection. Device work still runs.
- `gcs_metrics` (`bool`, optional, defaults to `False`). Result writing under the run's base output directory.

</details>

## Inference layout and sweeps

Sweep metadata requires string keys `two_axis_order_product_id_list`, `prefill_cache_axis_order_list`, and `ar_cache_axis_order_list`. Each value separates entries with `:`. The three lists are zipped. The first supplies labels, while the other two supply actual cache-order overrides. It is not a Cartesian product and the label does not change `compute_axis_order`. Keep the lists equally long to avoid truncation by `zip`.

`run_sweep` derives a sub-run name and TensorBoard destination for each pair. It writes `inference_microbenchmark_sweep_results.jsonl` in the working directory. [Benchmark operations](../benchmarks/inference-benchmarks.md) describes result and failure records. MLPerf's `OfflineModeArguments` and `engine_overrides` are separate call arguments, described in [MLPerf offline](../benchmarks/mlperf-offline.md), rather than new root model fields.

<details>
<summary>Parameters</summary>

- `stack_prefill_result_cache` (`bool`, optional, defaults to `False`). Prefill-cache stacking across layers.
- `prefill_cache_axis_order` (`str`, optional, defaults to `"1,2,0,3"`). Comma-separated prefill KV permutation.
- `ar_cache_axis_order` (`str`, optional, defaults to `"1,2,0,3"`). Autoregressive KV permutation.
- `compute_axis_order` (`str`, optional, defaults to `"0,1,2,3"`). Compute permutation.
- `reshape_q` (`bool`, optional, defaults to `False`). Query reshaping in supported paths.

</details>
