# Callbacks and profiling

Import callbacks from `lilax.callbacks`. See [callback guide](../../guides/training/callbacks.md).

## State and control

`TrainerCallback`, `TrainerState`, and `TrainerControl` retain their HF identities. The official [callback event reference](https://huggingface.co/docs/transformers/main_classes/callback) describes the upstream interfaces. Lilax dispatches these events, each with `args`, `state`, and `control`:

| Event | When dispatched | Additional event keywords |
| --- | --- | --- |
| `on_init_end` | Callback setup during construction | None |
| `on_train_begin`, `on_epoch_begin` | Training begins, including a resumed call | None |
| `on_step_begin` | Before an optimizer step starts | None |
| `on_substep_end` | SFT without multimodal input, DiLoCo, or optimizer host offload, DPO, distillation, or vLLM GRPO consumes an accumulation iteration without a completed optimizer update | None |
| `on_step_end` | An optimizer update completes | `loss`, a float |
| `on_log` | `trainer.log(...)` runs | `logs` |
| `on_evaluate` | Evaluation returns metrics | `metrics`, `metric_key_prefix` |
| `on_save` | A checkpoint save completes | `action_id`, `completed_step`, `checkpoint_step`, `tier` |
| `on_epoch_end`, `on_train_end` | The training loop finishes normally or through supported early stopping | None |

Epoch events bracket the training call rather than each pass through the dataset. End events are not guaranteed after an exception. Built-in reporting callbacks also close their resources when the training context exits.

The HF handler supplies `model`, `processing_class`, `optimizer`, `lr_scheduler`, `train_dataloader`, and `eval_dataloader` as callback keywords. Dataloader slots are not populated with Lilax iterators. Lilax does not dispatch `on_pre_optimizer_step`, `on_optimizer_step`, `on_prediction_step`, `on_predict`, or `on_push_begin`. `Trainer`, SFT with multimodal input, DiLoCo, or optimizer host offload, and GRPO with `engine="maxtext"` do not dispatch `on_substep_end`.

Return a `TrainerControl` from custom events to request `should_log`, `should_evaluate`, `should_save`, `should_training_stop`, or `should_epoch_stop`. Both stop flags end the current loop. Evaluation requests still require a supported evaluation path and dataset. `state.global_step` identifies the completed optimizer step. `trainer.add_callback`, `pop_callback`, and `remove_callback` manage callbacks.

## Reporting callbacks

`report_to` automatically adds selected reporting callbacks when absent. Tracker resources close at training end. Removing a TensorBoard, W&B, or Trackio callback with `pop_callback` or `remove_callback` immediately closes its active writer or run. `pop_callback` returns the object after closure.

### TensorBoardCallback

Import from `lilax.callbacks`. Callback class.

```python
TensorBoardCallback(*, use_vertex_tensorboard=False, project="", region="")
```

`TensorBoardCallback` writes numeric scalars on process zero. It uses `args.logging_dir` when nonempty, otherwise `<output_dir>/<run_name>/tensorboard/`, then appends the resolved run name. Default event files therefore live under `<output_dir>/<run_name>/tensorboard/<run_name>/`. Decoupled execution disables the writer.

`use_vertex_tensorboard=True` configures Vertex TensorBoard and starts upload. A nonempty `UPLOAD_DATA_TO_TENSORBOARD` environment variable also starts upload, even with the constructor default `False`, using the Vertex project, region, instance, and experiment environment settings. Decoupled execution skips Vertex configuration.

<details>
<summary>Parameters</summary>

- `use_vertex_tensorboard` (`bool`, optional, defaults to `False`). Vertex TensorBoard configuration and upload.
- `project` (`str`, optional, defaults to `""`). Vertex project.
- `region` (`str`, optional, defaults to `""`). Vertex region.

</details>

### WandbCallback

Import from `lilax.callbacks`. Callback class.

```python
WandbCallback(*, project="maxtext", name="")
```

Reports logs and evaluation metrics to W&B on world process zero. Closes the run at training end.

<details>
<summary>Parameters</summary>

- `project` (`str`, optional, defaults to `"maxtext"`). W&B project.
- `name` (`str`, optional, defaults to `""`). Run name. Empty uses `args.run_name`.

</details>

### TrackioCallback

Import from `lilax.callbacks`. Callback class.

```python
TrackioCallback(*, project="maxtext", name="")
```

Reports logs and evaluation metrics to Trackio on world process zero. Closes the run at training end.

<details>
<summary>Parameters</summary>

- `project` (`str`, optional, defaults to `"maxtext"`). Trackio project.
- `name` (`str`, optional, defaults to `""`). Run name. Empty uses `args.run_name`.

</details>

### MetricsCallback

Import from `lilax.callbacks`. Callback class.

```python
MetricsCallback(*, metrics_file=None, gcs_metrics=False, save_config_to_gcs=False)
```

`MetricsCallback` persists only `int` and `float` values from process zero. `metrics_file` selects a local JSON Lines file containing scalar values converted to floats, a float `step`, and `run_name`. It appends records, but a write at step zero truncates the file first. No local file is written when `metrics_file=None`.

With `gcs_metrics=True`, metrics are batched into JSON Lines files named `metrics_step_<start:06>_to_step_<end:06>.txt` in the working directory and uploaded to `<output_dir>/<run_name>/metrics/`. This GCS path requires a GCS output destination. Decoupled execution skips these uploads. The default callback enables neither local metric files nor GCS metric uploads.

<details>
<summary>Parameters</summary>

- `metrics_file` (`str | os.PathLike | None`, optional, defaults to `None`). Local scalar JSON Lines destination.
- `gcs_metrics` (`bool`, optional, defaults to `False`). Metric batch upload to GCS.
- `save_config_to_gcs` (`bool`, optional, defaults to `False`). Resolved configuration upload to GCS at training start.

</details>

## Stopping callbacks

### TargetEvalLossCallback

Import from `lilax.callbacks`. Callback class.

```python
TargetEvalLossCallback(target_eval_loss=0.0)
```

Requests a stop when the prefixed evaluation loss reaches a nonzero target.

<details>
<summary>Parameters</summary>

- `target_eval_loss` (`float`, optional, defaults to `0.0`). Stop threshold. Zero disables it.

</details>

### NonFiniteLossCallback

Import from `lilax.callbacks`. Callback class.

```python
NonFiniteLossCallback(*, abort_on_nan_loss=True, abort_on_inf_loss=True)
```

Aborts on selected nonfinite losses with `SystemExit(1)`.

<details>
<summary>Parameters</summary>

- `abort_on_nan_loss` (`bool`, optional, defaults to `True`). Abort selection for NaN loss.
- `abort_on_inf_loss` (`bool`, optional, defaults to `True`). Abort selection for infinite loss.

</details>

## Profiling and MLPerf

### ProfilerCallback

Import from `lilax.callbacks`. Callback class.

```python
ProfilerCallback(config: ProfilingConfig)
```

Selects runtime profiling through `lilax.profiling.ProfilingConfig`.

<details>
<summary>Parameters</summary>

- `config` (`ProfilingConfig`). Profiling settings. Defaults are listed below.

</details>

#### ProfilingConfig

Import from `lilax.profiling`. Pydantic model whose keyword fields configure runtime profiling.

`profiler` preserves the [ProfilerType enum](../models/configuration-enums.md#profilertype). Its serialized values are `""` for disabled profiling, `"xplane"`, and `"nsys"`.

<details>
<summary>Parameters</summary>

- `profiler` (`ProfilerType`, optional, defaults to `""`). Profiler selection.
- `skip_first_n_steps_for_profiler` (`int`, optional, defaults to `1`). Initial steps skipped.
- `profiler_steps` (`int`, optional, defaults to `5`). Captured step count.
- `profile_cleanly` (`bool`, optional, defaults to `True`). Clean profiling selection.
- `profile_periodically_period` (`int`, optional, defaults to `-1`). Periodic capture interval.
- `hide_profiler_step_metric` (`bool`, optional, defaults to `False`). Profiler step metric suppression.
- `enable_continuous_profiling` (`bool`, optional, defaults to `False`). Continuous profiling.

</details>

### RLProfilerCallback

Import from `lilax.callbacks`. Callback class.

```python
RLProfilerCallback(*, start_invocation=2, num_invocations=1, timeout_secs=60.0)
```

Selects invocation-based profiling.

<details>
<summary>Parameters</summary>

- `start_invocation` (`int`, optional, defaults to `2`). First profiled invocation.
- `num_invocations` (`int`, optional, defaults to `1`). Captured invocation count.
- `timeout_secs` (`float`, optional, defaults to `60.0`). Capture timeout in seconds.

</details>

### MLPerfCallback

Import from `lilax.callbacks`. Callback class.

```python
MLPerfCallback(*, mllog_file="")
```

Selects MLPerf logging.

<details>
<summary>Parameters</summary>

- `mllog_file` (`str`, optional, defaults to `""`). MLPerf log destination.

</details>

### profile_m3

Import from `lilax.profiling`.

```python
profile_m3(trainer) -> float
```

Runs the configured M3 synthetic profiling loop. Requires a trainer configured for an M3-supported family. Creates a separate initialized model, mesh over `jax.devices()`, and optimizer, using synthetic inputs instead of the trainer's weights or dataset. Gradient accumulation greater than one raises `ValueError`. The global batch must be positive and divisible by the synthetic mesh's `data` and `fsdp` shard count. See the [bounded profiling example](../../guides/training/callbacks.md#profile-synthetic-training-with-m3).

Returns the last synthetic training loss as a `float`, or `NaN` if no steps run. Logs per-step loss, elapsed time, and tokens per second. XPlane traces use the resolved `tensorboard_dir` on process zero unless `upload_all_profiler_results` enables capture on every process.

<details>
<summary>Parameters</summary>

- `trainer` (`Trainer`). Configured trainer exposing resolved `.config` settings for the M3 loop. Import from `lilax`.

</details>
