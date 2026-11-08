# Diagnostic writers and reports

See [diagnostics guide](../../guides/execution/diagnostics.md).

Configuration inputs below require the resolved configuration, `model.runtime.resolve_config().config`. See [configuration objects](runtime.md#configuration-objects).

## Goodput

Import from `lilax.diagnostics.goodput`.

`GoodputEvent` members are `JOB`, `TPU_INIT`, `TRAINING_PREPARATION`, `DATA_LOADING`, and `STEP`. `RECORD_JOB_START_TIME` and `RECORD_JOB_END_TIME` are recorder method-name strings.

### get_goodput_job_name

```python
get_goodput_job_name(config) -> str
```

Returns the configured goodput name, or the run name when empty.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with `goodput_job_name` and `run_name`.

</details>

### create_goodput_recorder

```python
create_goodput_recorder(config)
```

Returns a recorder, or `None` when disabled. Elastic recording is selected when supported by the run.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with recording and elastic settings.

</details>

### record_goodput

```python
record_goodput(recorder, event_name, *args)
```

Calls the named recorder method when present.

Returns `None`.

<details>
<summary>Parameters</summary>

- `recorder` (`GoodputRecorder | ElasticGoodputRecorder | None`). Event destination from `ml_goodput_measurement`. None is a no-op.
- `event_name` (`str`). Recorder method name.
- `*args` (`Any`). Variadic positional arguments forwarded to the recorder method.

</details>

### maybe_record_goodput

```python
maybe_record_goodput(recorder, event_name, *args)
```

Returns a context manager recording event start and successful completion only. Failure does not record an end event.

<details>
<summary>Parameters</summary>

- `recorder` (`GoodputRecorder | ElasticGoodputRecorder | None`). Event destination from `ml_goodput_measurement`.
- `event_name` (`GoodputEvent`). Event selecting start and end recorder methods.
- `*args` (`Any`). Variadic positional arguments forwarded to both recorder calls.

</details>

### maybe_monitor_goodput

```python
maybe_monitor_goodput(config)
```

Returns a context manager owning background cumulative and optional rolling-window uploads on process zero when enabled. It stops the uploader on exit. Uploads can write to TensorBoard and Cloud Monitoring. Performance reporting disables the configuration's duplicate GCP step-deviation stream.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with monitoring settings and destinations.

</details>

### maybe_monitor_rolling_window_goodput

```python
maybe_monitor_rolling_window_goodput(goodput_monitor, config)
```

Returns a context manager owning optional rolling-window uploads to TensorBoard and Cloud Monitoring. Startup and cleanup failures are logged. Disabled monitoring is a no-op.

<details>
<summary>Parameters</summary>

- `goodput_monitor` (`GoodputMonitor | ElasticGoodputMonitor | None`). Monitor from `ml_goodput_measurement` owning the uploader.
- `config` (`object`). Resolved configuration with the enable flag and window durations.

</details>

## TensorBoard

Import from `lilax.diagnostics.tensorboard`.

### initialize_summary_writer

```python
initialize_summary_writer(tensorboard_dir, run_name, enable_tensorboard=True)
```

Returns a tensorboardX writer on process zero, or `None` when disabled, decoupled, or on another process. Missing directory or name uses the writer's default directory.

<details>
<summary>Parameters</summary>

- `tensorboard_dir` (`str | None`). Base log directory.
- `run_name` (`str | None`). Run subdirectory.
- `enable_tensorboard` (`bool`, optional, defaults to `True`). TensorBoard writer creation.

</details>

### add_text_to_summary_writer

```python
add_text_to_summary_writer(key, value, summary_writer)
```

Writes text on process zero. A `None` writer or another process is a no-op.

Returns `None`.

<details>
<summary>Parameters</summary>

- `key` (`str`). Summary tag.
- `value` (`str`). Summary text.
- `summary_writer` (`tensorboardX.SummaryWriter | None`). Destination.

</details>

### close_summary_writer

```python
close_summary_writer(summary_writer)
```

Closes the writer on process zero. A `None` writer or another process is a no-op.

Returns `None`.

<details>
<summary>Parameters</summary>

- `summary_writer` (`tensorboardX.SummaryWriter | None`). Writer to close.

</details>

### VertexTensorboardManager

Import from `lilax.diagnostics.tensorboard`. Class with the explicit constructor below.

```python
VertexTensorboardManager()
```

Owns Vertex TensorBoard setup and upload. Destruction stops an active uploader.

See [upload TensorBoard logs to Vertex AI](../../guides/execution/diagnostics.md#upload-tensorboard-logs-to-vertex-ai) for configuration and cloud side effects.

#### setup

```python
manager.setup()
```

Creates a Vertex instance and experiment using `TENSORBOARD_PROJECT`, `TENSORBOARD_REGION`, `TENSORBOARD_NAME`, and `EXPERIMENT_NAME`.

Returns the experiment URL or `None` when setup cannot proceed.

#### upload_data

```python
manager.upload_data(tensorboard_dir)
```

Starts cloud upload using the Vertex environment settings above.

Returns `None`.

<details>
<summary>Parameters</summary>

- `tensorboard_dir` (`str`). Log directory to upload.

</details>

#### configure_vertex_tensorboard

```python
manager.configure_vertex_tensorboard(config)
```

Applies resolved settings and missing environment values on process zero. Decoupled execution is a no-op. May create cloud resources and start upload. Requested setup failure raises `ValueError`.

Returns `None`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with Vertex project, region, run identity, and TensorBoard settings.

</details>

## Structured evaluation reports

Import from `lilax.evaluation.diagnostics`.

### SimpleEvalsDebugCollector

Import from `lilax.evaluation.diagnostics`. Class with the explicit constructor below.

```python
SimpleEvalsDebugCollector()
```

Tracks request records and peak concurrency.

#### set_task

```python
collector.set_task(task: str) -> None
```

Selects the task for new records.

Returns `None`.

<details>
<summary>Parameters</summary>

- `task` (`str`). Task name.

</details>

#### begin

```python
collector.begin(messages: list[dict[str, Any]]) -> dict[str, Any]
```

Returns a request dictionary with a start timestamp. Updates active and peak concurrency counters.

<details>
<summary>Parameters</summary>

- `messages` (`list[dict[str, Any]]`). Request messages.

</details>

#### finish

```python
collector.finish(record: dict[str, Any]) -> None
```

Replaces the record's start timestamp with elapsed latency, stores the record, and decrements active concurrency.

Returns `None`.

<details>
<summary>Parameters</summary>

- `record` (`dict[str, Any]`). Unfinished record returned by `begin`.

</details>

### build_debug_report

```python
build_debug_report(collector: SimpleEvalsDebugCollector, eval_results: dict[str, Any],
	task_durations: dict[str, float], model_name: str, max_tokens: int, max_model_len: int) -> str
```

Returns report text describing collected requests and evaluation results.

See [build a request report](../../guides/execution/diagnostics.md#build-a-request-report) for an application example.

<details>
<summary>Parameters</summary>

- `collector` (`SimpleEvalsDebugCollector`). Collected request diagnostics.
- `eval_results` (`dict[str, Any]`). Task results.
- `task_durations` (`dict[str, float]`). Per-task elapsed seconds.
- `model_name` (`str`). Report model identity.
- `max_tokens` (`int`). Completion limit for saturation reporting.
- `max_model_len` (`int`). Total sequence limit.

</details>

### write_debug_report

```python
write_debug_report(report: str, results_path: str, json_local_path: str) -> str
```

Writes a sibling `.debug.txt` file and uploads it when the results path is GCS.

Returns the absolute local path.

<details>
<summary>Parameters</summary>

- `report` (`str`). Report text.
- `results_path` (`str`). Result destination, including optional `gs://` upload directory.
- `json_local_path` (`str`). JSON artifact whose basename and directory determine the report path.

</details>

## Training metric logging

Import from `lilax.diagnostics.metrics`. `STEP_DIAGNOSTICS_KEYS` names recognized step diagnostics.

`MetadataKey` contains `PER_DEVICE_TFLOPS="per_device_tflops"` and `PER_DEVICE_TOKENS="per_device_tokens"`. Metric inputs use the logger's `scalar` mapping, with optional collected intermediates. Writer methods return `None` and can write local files or enabled external sinks.

### step_diagnostics_text

```python
step_diagnostics_text(scalars)
```

Returns formatted recognized diagnostics, using `n/a` for absent values.

<details>
<summary>Parameters</summary>

- `scalars` (`dict[str, Any]`). Step scalar values.

</details>

### record_activation_metrics

```python
record_activation_metrics(output_metrics, intermediate_outputs, config)
```

Adds activation statistics to `output_metrics["scalar"]`.

See [application metrics](../../guides/execution/diagnostics.md#write-application-metrics) for collected-leaf names and an example.

Returns `None`.

<details>
<summary>Parameters</summary>

- `output_metrics` (`dict[str, Any]`). Mutable metric destination.
- `intermediate_outputs` (`dict[str, Any]`). Collected activation tree.
- `config` (`object`). Resolved configuration with the decoder-layer count.

</details>

### MetricLogger

Import from `lilax.diagnostics.metrics`. Class with the explicit constructor below.

```python
MetricLogger(config, learning_rate_schedule, start_step=0, reporter=None)
```

Owns metric storage and reporting resources using resolved runtime settings. Construction can create writers, managed diagnostics, workload-monitoring threads, and a W&B run.

For trainer-managed logging, use [callbacks](../training/callbacks.md) rather than constructing a second logger for the same run.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with logging destinations and run settings.
- `learning_rate_schedule` (`Callable`). Step-to-learning-rate mapping.
- `start_step` (`int`, optional, defaults to `0`). Initial training step.
- `reporter` (`MetricReporter | None`, optional, defaults to `None`). Reporter from `lilax.trainers.telemetry.reporting`, or a compatible object exposing `log(metrics, step, metric_type)` and `close()` for cleanup.

</details>

#### mark_eval_loop_start

```python
logger.mark_eval_loop_start()
```

Records the wall-clock evaluation start time.

Returns `None`.

#### reset_eval_metrics

```python
logger.reset_eval_metrics()
```

Clears accumulated evaluation metrics and the pending-step count.

Returns `None`.

#### write_metrics

```python
logger.write_metrics(metrics, step, metric_type="train")
```

Logs and writes to configured sinks and the optional reporter. Training metrics can trigger configured abort conditions.

For `metric_type="eval"`, `metrics["scalar"]` requires `eval/avg_loss`, `eval/avg_perplexity`, and `eval/total_weights`. MoE models also require `eval/avg_moe_lb_loss`. Models with MTP require `eval/avg_mtp_loss` and `eval/avg_mtp_acceptance_rate_percent`. See [application metrics](../../guides/execution/diagnostics.md#write-application-metrics) for logger ownership and cleanup.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Scalar and grouped scalar values.
- `step` (`int`). Step index.
- `metric_type` (`str`, optional, defaults to `"train"`). `train`, `eval`, or `running_eval`.

</details>

#### log_metrics

```python
logger.log_metrics(metrics, step, metric_type)
```

Formats metrics for logging.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to log.
- `step` (`int`). Step index.
- `metric_type` (`str`). `train`, `eval`, or `running_eval`.

</details>

#### buffer_and_write_metrics

```python
logger.buffer_and_write_metrics(metrics, step, step_time_delta=None,
	is_training=True, *, record_train=True)
```

Flushes the preceding deferred entry and buffers the current metrics. Returning to training finalizes pending evaluation metrics and can raise the configured target-loss stop.

Returns `None`.

Recording training metrics calls `step_time_delta.total_seconds()`, so that path requires a duration despite the `None` default.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Mutable current step metrics.
- `step` (`int`). Step index.
- `step_time_delta` (`datetime.timedelta | None`, optional, defaults to `None`). Must expose `total_seconds()` when recording training metrics.
- `is_training` (`bool`, optional, defaults to `True`). Training-buffer selection instead of evaluation buffering.
- `record_train` (`bool`, optional, defaults to `True`). Add training timing and learning-rate metrics.

</details>

#### write_metrics_locally

```python
logger.write_metrics_locally(metrics, step)
```

Writes metrics to local storage.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to write.
- `step` (`int`). Step index.

</details>

#### write_metrics_for_gcs

```python
logger.write_metrics_for_gcs(metrics, step, metric_type)
```

Writes metrics through GCS storage.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to write.
- `step` (`int`). Step index.
- `metric_type` (`str`). Metric category.

</details>

#### write_metrics_to_tensorboard

```python
logger.write_metrics_to_tensorboard(metrics, step, metric_type)
```

Writes scalar summaries on process zero and flushes full training logs at the configured interval. Requires an initialized writer on the writing process.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to write.
- `step` (`int`). Step index.
- `metric_type` (`str`). Metric category.

</details>

#### write_metrics_to_managed_mldiagnostics

```python
logger.write_metrics_to_managed_mldiagnostics(metrics, step)
```

Records managed-diagnostics metrics at logging intervals and the final step.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to write.
- `step` (`int`). Step index.

</details>

#### write_metrics_to_wandb

```python
logger.write_metrics_to_wandb(metrics, step)
```

Flattens scalar groups and writes to W&B.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Values to write.
- `step` (`int`). Step index.

</details>

#### write_setup_info_to_tensorboard

```python
logger.write_setup_info_to_tensorboard(params)
```

Initializes FLOP and token metadata, logs parameter count, and writes setup summaries when a writer exists.

Returns `None`.

<details>
<summary>Parameters</summary>

- `params` (`flax.nnx.State | dict`). Model parameter tree used for setup reporting.

</details>

#### get_performance_metric_queue

```python
logger.get_performance_metric_queue(config)
```

Starts enabled heartbeat or performance reporting threads. Decoupled execution skips workload monitoring.

Returns a performance queue when enabled, otherwise `None`.

<details>
<summary>Parameters</summary>

- `config` (`object`). Resolved configuration with workload-monitor settings.

</details>

#### record_train_metrics

```python
logger.record_train_metrics(metrics, step, step_time)
```

Adds timing, learning rate, and available throughput metadata to the metrics. Requires setup metadata after ramp-up.

Returns `None`.

<details>
<summary>Parameters</summary>

- `metrics` (`dict[str, Any]`). Mutable destination with a `scalar` mapping.
- `step` (`int`). Step index.
- `step_time` (`int | float`). Step duration in seconds.

</details>

#### flush_metrics_and_cleanup

```python
logger.flush_metrics_and_cleanup()
```

Flushes pending metrics and closes the summary writer and optional reporter. This terminal operation ends writer ownership. Further writes are unsupported.

Returns `None`.

## Save evaluation JSON

Import from `lilax.evaluation.reporting`.

### write_results

```python
write_results(benchmark: str, model_name: str, scores: dict, generation_stats: dict,
	config: dict, results_path: str = "./eval_results") -> dict
```

Writes timestamped JSON. A GCS destination writes a temporary local copy and uploads it.

See [save and upload evaluation results](../../guides/execution/diagnostics.md#save-and-upload-evaluation-results) for local writing and a separate cloud upload.

Returns `{"results": payload, "local_path": absolute_path}`.

<details>
<summary>Parameters</summary>

- `benchmark` (`str`). Benchmark identity and filename prefix.
- `model_name` (`str`). Model identity. Filename replaces `/` and `:` with `_`.
- `scores` (`dict`). JSON-serializable scores.
- `generation_stats` (`dict`). JSON-serializable generation statistics.
- `config` (`dict`). JSON-serializable configuration.
- `results_path` (`str`, optional, defaults to `"./eval_results"`). Local or GCS output directory.

</details>

### upload_results

```python
upload_results(local_path: str, gcs_path: str) -> None
```

Uploads an existing file. A destination ending in `/` appends the local basename.

Returns `None`.

<details>
<summary>Parameters</summary>

- `local_path` (`str`). Existing local file.
- `gcs_path` (`str`). GCS destination object or directory.

</details>
