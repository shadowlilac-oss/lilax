# Inference benchmark operations

These operations accept resolved configuration (`runtime.config`). See [configuration objects](../execution/runtime.md#configuration-objects).

## Microbenchmarks

Import from `lilax.benchmarks.inference.inference_benchmarks`.

### run_benchmarks

```python
run_benchmarks(config)
```

Loads parameters, compiles inference functions, and benchmarks selected prefill and autoregressive stages.
Optional log and GCS settings write results.

Returns a results dictionary.

The root model configuration does not replace the resolved configuration. See [runtime example](../../guides/benchmarks/inference-benchmarks.md).

<details>
<summary>Parameters</summary>

- `config` (`object`). A resolved execution configuration. Comma-separated `inference_microbenchmark_stages` selects stages, `inference_microbenchmark_prefill_lengths` selects lengths, and `inference_microbenchmark_loop_iters` sets iterations.

</details>

### run_benchmarks_with_unsafe_rbg

```python
run_benchmarks_with_unsafe_rbg(config)
```

Changes JAX's default PRNG implementation before running the same benchmark.

Returns its results dictionary.

<details>
<summary>Parameters</summary>

- `config` (`object`). The resolved execution configuration accepted by `run_benchmarks`.

</details>

### run_sweep

```python
run_sweep(config)
```

Reads `inference_metadata_file`. Colon-separated `two_axis_order_product_id_list`, `prefill_cache_axis_order_list`, and `ar_cache_axis_order_list` are zipped, not multiplied into a Cartesian grid.
Writes `inference_microbenchmark_sweep_results.jsonl`. JAX runtime failures produce empty metrics and an OOM marker.

Returns metric and dimension records.

<details>
<summary>Parameters</summary>

- `config` (`object`). A resolved execution configuration containing the metadata file path and base benchmark settings.

</details>

## ShareGPT estimates

Import from `lilax.benchmarks.inference.sharegpt`.

### analyze_sharegpt

```python
analyze_sharegpt(convo_file, mb_timing_file="", verbose=False)
```

Prints a workload estimate.
Bundled measurements are not a claim about your hardware.

Returns input token total, output token total, total seconds, prefill seconds, and generation seconds.

<details>
<summary>Parameters</summary>

- `convo_file` (`str`). The ShareGPT JSON conversation file.
- `mb_timing_file` (`str`, optional, defaults to `""`). Timing JSON. Empty uses bundled measurements.
- `verbose` (`bool`, optional, defaults to `False`). Flag for per-conversation estimates.

</details>

### get_conversations_from_file

```python
get_conversations_from_file(filename, max_input_tokens, max_output_tokens)
```

Reads the first two conversation messages, estimates tokens, and filters by limits. Prints retention statistics.
No retained conversations can cause division by zero.

Returns input and output token-count pairs.

<details>
<summary>Parameters</summary>

- `filename` (`str`). The ShareGPT JSON file.
- `max_input_tokens` (`int`). Limit on estimated input tokens.
- `max_output_tokens` (`int`). Limit on estimated output tokens.

</details>

### get_prefill_and_generate_times

```python
get_prefill_and_generate_times(filename="")
```

Returns a prefill bucket mapping and decode milliseconds per token.

<details>
<summary>Parameters</summary>

- `filename` (`str`, optional, defaults to `""`). Timing JSON with `Prefill.<length>.prefill_time_in_ms` and `AutoRegressive.ar_step_in_ms_per_seq`. Empty selects bundled measurements.

</details>

### compute_times

```python
compute_times(conversations, prefill_bucket_size_to_ms, system_time_per_decode_token_ms, verbose=False)
```

Selects power-of-two prefill buckets with a minimum of 128. Missing buckets raise `KeyError`.
Prints estimates.

Returns total, prefill, and generation seconds.

<details>
<summary>Parameters</summary>

- `conversations` (`list[tuple[int, int]]`). Input and output token-count pairs.
- `prefill_bucket_size_to_ms` (`dict[int, float]`). Mapping of bucket lengths to milliseconds.
- `system_time_per_decode_token_ms` (`float`). Decode milliseconds per token.
- `verbose` (`bool`, optional, defaults to `False`). Flag for per-conversation estimates.

</details>

### get_num_tokens_in_convos

```python
get_num_tokens_in_convos(conversations)
```

Returns input and output token totals.

<details>
<summary>Parameters</summary>

- `conversations` (`list[tuple[int, int]]`). Input and output token-count pairs.

</details>

### tokens_in_input_str

```python
tokens_in_input_str(s)
```

Returns the estimate `int(1.3 * word_count)`.

<details>
<summary>Parameters</summary>

- `s` (`str`). Whitespace-separated text.

</details>

### next_power_of_2

```python
next_power_of_2(x)
```

Returns the next power-of-two bucket size, with zero mapped to one.

<details>
<summary>Parameters</summary>

- `x` (`int`). The token count.

</details>
