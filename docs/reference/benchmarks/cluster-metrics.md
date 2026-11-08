# Cluster metric summaries

Import from `lilax.benchmarks.cluster.benchmark_metrics`. See [cluster guide](../../guides/benchmarks/cluster-workflows.md).

## Metrics

Import from `lilax.benchmarks.cluster.benchmark_metrics`. Dataclass.

```python
Metrics(avg_tflops_per_sec: float, avg_tokens_per_sec: float, median_step_time: float, e2e_step_time: float)
```

Holds a computed summary in a dataclass record.

<details>
<summary>Parameters</summary>

- `avg_tflops_per_sec` (`float`). Average per-device TFLOPs per second.
- `avg_tokens_per_sec` (`float`). Average per-device tokens per second.
- `median_step_time` (`float`). Median step time in seconds.
- `e2e_step_time` (`float`). Summed step time in seconds.

</details>

## parse_metrics

```python
parse_metrics(metrics_text: str, total_steps: int, last_n_steps: int = 10) -> Metrics
```

Reads JSONL and averages the final matching `perf/per_device_tflops_per_sec` and `perf/per_device_tokens_per_sec` values.
Takes the median of final `perf/step_time_seconds` values and sums up to `total_steps` matching step times.
Malformed JSON propagates. Missing metric values can cause division or statistics errors.

Returns `Metrics`.

<details>
<summary>Parameters</summary>

- `metrics_text` (`str`). JSONL metric records.
- `total_steps` (`int`). Limit on step-time values included in the sum.
- `last_n_steps` (`int`, optional, defaults to `10`). Limit on matching values used for averages and median.

</details>

## Upload records

### BenchmarkRun

Import from `lilax.benchmarks.cluster.benchmark_metrics`. Dataclass.

```python
BenchmarkRun(
	metrics_gcs_file: str, model_id: str, hardware_id: str, software_id: str, number_of_chips: int,
	container_image_name: str, global_batch_size: int, precision: str, optimizer: str, seq_length: int,
	number_of_steps: int, xla_flags: str, dataset: str, run_type: str, config_file: str, topology: str,
	tuning_params: dict, db_project: str, db_dataset: str, hardware_num_slices: Optional[int] = None,
)
```

Describes the run being uploaded in a dataclass record.

<details>
<summary>Parameters</summary>

- `metrics_gcs_file` (`str`). The Cloud Storage metrics prefix.
- `model_id` (`str`). The model.
- `hardware_id` (`str`). Hardware used for utilization calculation.
- `software_id` (`str`). Software.
- `number_of_chips` (`int`). The chip count.
- `container_image_name` (`str`). The container image.
- `global_batch_size` (`int`). The run batch size.
- `precision` (`str`). Numerical precision.
- `optimizer` (`str`). The optimizer.
- `seq_length` (`int`). Sequence length.
- `number_of_steps` (`int`). Limit on metric aggregation.
- `xla_flags` (`str`). Compilation flags.
- `dataset` (`str`). The dataset.
- `run_type` (`str`). Run category.
- `config_file` (`str`). The configuration file loaded during upload.
- `topology` (`str`). Hardware topology.
- `tuning_params` (`dict`). Overrides for the loaded configuration in the uploaded record.
- `db_project` (`str`). The BigQuery project.
- `db_dataset` (`str`). The BigQuery dataset.
- `hardware_num_slices` (`Optional[int]`, optional, defaults to `None`). Accelerator slices.

</details>

### upload_benchmark_metrics

```python
upload_benchmark_metrics(run: BenchmarkRun) -> Optional[Metrics]
```

Reads the first matching Cloud Storage metric object, loads configuration, and writes BigQuery's `run_summary` table.
It can create the table. Incompatible required schema fields raise `ValueError`. Upload errors are logged.
The uploaded environment record includes process environment variables.

Returns `Metrics` after upload, or `None` outside Pathways and worker zero.

<details>
<summary>Parameters</summary>

- `run` (`BenchmarkRun`). Metric inputs, run metadata, and the upload destination.

</details>

### upload_workload_metrics

Import from `lilax.benchmarks.cluster.benchmark_recipes`.

```python
upload_workload_metrics(workload: WorkloadConfig, *, db_project: str, db_dataset: str)
```

Constructs an upload record using an explicit BigQuery destination. The runner variant uses the destination stored in the workload.
The upload side effects and process selection match `upload_benchmark_metrics`.

<details>
<summary>Parameters</summary>

- `workload` (`WorkloadConfig`). The submitted workload's configuration and run metadata.
- `db_project` (`str`). The BigQuery project.
- `db_dataset` (`str`). The BigQuery dataset.

</details>
