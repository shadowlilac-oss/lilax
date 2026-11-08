# MLPerf offline inference

Import from `lilax.benchmarks.mlperf.offline`. See [offline guide](../../guides/benchmarks/mlperf-offline.md).

`engine_api.Engine` is the JetStream engine interface. Runtime-source configuration (`model.runtime.public_config`) and resolved configuration (`runtime.config`) are distinct from the root `lilax.MaxTextConfig`. See [configuration objects](../execution/runtime.md#configuration-objects).

## OfflineModeArguments

Import from `lilax.benchmarks.mlperf.offline`. Dataclass.

`class lilax.benchmarks.mlperf.offline.OfflineModeArguments` is a dataclass declaring the fields below.

<details>
<summary>Parameters</summary>

- `dataset_path` (`str`). Dataset file.
- `mlperf_test_mode` (`str`, optional, defaults to `"performance"`). LoadGen mode.
- `api_url` (`str | None`, optional, defaults to `None`). An external API instead of local inference.
- `is_stream` (`bool`, optional, defaults to `False`). External API streaming.
- `input_mode` (`str`, optional, defaults to `"tokenized"`). API input representation.
- `output_mode` (`str`, optional, defaults to `"tokenized"`). API output representation.
- `audit_conf` (`str`, optional, defaults to `"audit.conf"`). Audit configuration.
- `mlperf_conf` (`str`, optional, defaults to `"mlperf.conf"`). MLPerf configuration.
- `user_conf` (`str`, optional, defaults to `"user.conf"`). User configuration.
- `total_sample_count` (`int`, optional, defaults to `24576`). Limit on dataset samples.
- `perf_count_override` (`int | None`, optional, defaults to `None`). Performance sample count override.
- `output_log_dir` (`str`, optional, defaults to `"output-logs"`). Logs output.
- `enable_log_trace` (`bool`, optional, defaults to `False`). Flag for LoadGen trace logging.
- `prefill_lengths_and_per_device_batch_sizes` (`str`, optional, defaults to `"256,80|512,40|1024,20"`). Prefill length and batch size pairs.
- `jax_profiler_port` (`int`, optional, defaults to `9999`). The profiler server port.
- `enable_profile` (`bool`, optional, defaults to `False`). Flag for profiling.
- `enable_batch_prefill` (`bool`, optional, defaults to `False`). Flag for batched prefill.
- `skip_warmup` (`bool`, optional, defaults to `False`). Flag to skip warmup.
- `tok_outlen_multiplier` (`float`, optional, defaults to `3.0`). Reference output length multiplier for query classification.
- `allow_skipping_queries` (`bool`, optional, defaults to `False`). Flag to skip queries beyond configured prefill lengths.
- `rename_dataset_cols` (`str`, optional, defaults to `""`). JSON column renames.
- `engine_overrides` (`dict`, optional, defaults to `{}`). Engine setting overrides. Each instance receives a fresh value.

</details>

### offline_arguments

```python
offline_arguments(dataset_path, *, model="llama2-70b", test_run=False, **overrides)
```

Returns arguments with source model-specific defaults, which may differ from dataclass defaults.

<details>
<summary>Parameters</summary>

- `dataset_path` (`str`). The dataset file.
- `model` (`str`, optional, defaults to `"llama2-70b"`). Recipe, `llama2-70b` or `mixtral-8x7b`.
- `test_run` (`bool`, optional, defaults to `False`). Flag for the reduced sample count.
- `**overrides` (`dict[str, Any]`). Overrides for generated argument values.

</details>

## Workflow and engine

### run_offline

```python
run_offline(config, args: OfflineModeArguments, *, devices=None)
```

Runs LoadGen, preparing configuration-driven inference when no external API URL is selected.

Dataset files must be Parquet, CSV, JSON, or JSONL. Other formats raise `ValueError`. The run disables garbage collection and changes JAX PRNG configuration; optional profiling starts a server.

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible runtime-source configuration (`model.runtime.public_config`).
- `args` (`OfflineModeArguments`). Dataset and LoadGen settings.
- `devices` (`Sequence[jax.Device] | None`, optional, defaults to `None`). Execution devices.

</details>

### run_offline_workflow

```python
run_offline_workflow(config, args, *, model="llama2-70b", checkpoint_path=None,
	tokenizer_path=None, fast_eval=False, accuracy_options=None, devices=None)
```

Runs the workload and optionally scores an existing accuracy log.

Returns scorer results for supported accuracy runs, otherwise `None`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible runtime-source configuration (`model.runtime.public_config`).
- `args` (`OfflineModeArguments`). The workload settings.
- `model` (`str`, optional, defaults to `"llama2-70b"`). Scorer defaults.
- `checkpoint_path` (`str | None`, optional, defaults to `None`). The scorer tokenizer checkpoint.
- `tokenizer_path` (`str | None`, optional, defaults to `None`). A tokenizer file for fast scoring.
- `fast_eval` (`bool`, optional, defaults to `False`). Parallel scoring.
- `accuracy_options` (`dict[str, Any] | None`, optional, defaults to `None`). Scorer keywords.
- `devices` (`Sequence[jax.Device] | None`, optional, defaults to `None`). Execution devices.

</details>

### create_engine_from_config

```python
create_engine_from_config(config, batch_size, max_prefill_predict_length,
	max_target_length, *, devices=None, **overrides)
```

Returns the benchmark engine. Configuration must match the runtime-source execution contract, not the root HF model configuration.
An explicit parameter checkpoint selection is required.

<details>
<summary>Parameters</summary>

- `config` (`object`). Runtime-source configuration (`model.runtime.public_config`).
- `batch_size` (`float`). The per-device batch size.
- `max_prefill_predict_length` (`int`). Prefill capacity.
- `max_target_length` (`int`). Total target capacity.
- `devices` (`Sequence[jax.Device] | None`, optional, defaults to `None`). JAX devices.
- `**overrides` (`dict[str, Any]`). Engine setting overrides applied after benchmark defaults.

</details>

### InputData

Import from `lilax.benchmarks.mlperf.offline`. Dataclass.

```python
InputData(id: str, tokens: jax.Array, true_length: int)
```

Holds one input request.

<details>
<summary>Parameters</summary>

- `id` (`str`). Request identifier.
- `tokens` (`jax.Array`). Padded input tokens.
- `true_length` (`int`). The unpadded token count.

</details>

### OfflineInference

Import from `lilax.benchmarks.mlperf.offline`. Class.

```python
OfflineInference(engine: engine_api.Engine, params, base_engine: engine_api.Engine, enable_batch_prefill: bool)
```

Owns decode state for offline inference.

<details>
<summary>Parameters</summary>

- `engine` (`engine_api.Engine`). The inference engine.
- `params` (`dict | nnx.State | None`). Loaded parameters.
- `base_engine` (`engine_api.Engine | None`). The base engine for shared state.
- `enable_batch_prefill` (`bool`). Batched prefill.

</details>

### init_decode_state

```python
offline_inference.init_decode_state()
```

Initializes owned decode state.

### warmup

```python
offline_inference.warmup(max_length, warmup_samples)
```

Compiles and warms up inference for supplied samples.

<details>
<summary>Parameters</summary>

- `max_length` (`int`). Maximum prefill length.
- `warmup_samples` (`list[InputData]`). Warmup requests.

</details>

### batch_inference_with_callback

```python
offline_inference.batch_inference_with_callback(data: list[InputData],
	emit_first_token: Callable[[str, int], bool], emit_token: Callable[[str, int], bool], desc: str)
```

Runs batch inference and emits results through callbacks.

<details>
<summary>Parameters</summary>

- `data` (`list[InputData]`). `InputData` requests.
- `emit_first_token` (`Callable[[str, int], bool]`). Callback for first-token results.
- `emit_token` (`Callable[[str, int], bool]`). Callback for subsequent token results.
- `desc` (`str`). Workload label.

</details>

### batch_inference

```python
offline_inference.batch_inference(data: list[InputData], desc="") -> dict[str, list[int]]
```

Returns token lists keyed by request ID.

<details>
<summary>Parameters</summary>

- `data` (`list[InputData]`). `InputData` requests.
- `desc` (`str`, optional, defaults to `""`). Workload label.

</details>

## Chunked-prefill benchmark

Import from `lilax.benchmarks.inference.chunked_prefill_benchmark`.

```python
run_chunked_prefill_benchmark(config)
```

Initializes model and inference state and runs the specialized prefill workload.
An arbitrary HF model configuration does not supply the required benchmark configuration.

<details>
<summary>Parameters</summary>

- `config` (`object`). Compatible `Runtime`, runtime-source configuration (`model.runtime.public_config`), or resolved configuration (`runtime.config`) with chunked-prefill benchmark settings.

</details>

## XLA flag formatting

Import from `lilax.benchmarks.mlperf.xla_flags`, which also exports `xla_flags_autotuned` and `xla_flags_manually_selected`.

```python
build_xla_flag_str(flags_set=xla_flags_autotuned)
```

Returns the flag string used by the benchmark launch environment.

<details>
<summary>Parameters</summary>

- `flags_set` (`dict[str, str]`, optional, defaults to `xla_flags_autotuned`). Compilation flags to format.

</details>
