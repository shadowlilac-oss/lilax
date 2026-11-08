# Record diagnostics for an application

Use trainer callbacks for training metrics. The diagnostic modules also expose writer lifecycles and structured request reports for applications that own those resources.

## Prepare diagnostic configuration

Use the runtime bound to your model. The following setup creates a model wrapper without materializing weights. Runtime resolution initializes JAX and resolves the active execution profile.

```python
from lilax import AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("Qwen/Qwen3-8B"))
runtime = model.runtime
runtime.resolve_config()
config = runtime.config
```

`config` is the [resolved configuration](../../reference/execution/runtime.md#configuration-objects). Writer destinations and enabled services come from that configuration.

## Write TensorBoard text

```python
from lilax.diagnostics.tensorboard import (
	add_text_to_summary_writer, close_summary_writer, initialize_summary_writer,
)

writer = initialize_summary_writer("./logs", "application")
try:
	add_text_to_summary_writer("run", "Application diagnostics", writer)
finally:
	close_summary_writer(writer)
```

Only process zero creates a writer. Disabled or decoupled writing returns `None`; the add and close helpers accept it.

## Record completed data-loading work

Enable recording before launching the Python application. Save this complete profile as `goodput.yaml` and use `lilax launch --config-file goodput.yaml application.py`.

```yaml
version: 1
process:
  hardware: gpu
goodput:
  enable_goodput_recording: true
  goodput_job_name: application
```

Use the hardware that matches your deployment. The recorder needs access to Cloud Logging. This profile records events without starting a monitoring uploader. See [goodput and monitoring fields](../../reference/execution/execution-profile-monitoring.md) for the separate upload controls.

Supply `runtime` from your model's `model.runtime` and `load_batch` as your application's batch-loading callable. Resolving the runtime initializes JAX. When enabled, recording writes events to Cloud Logging using the launch profile above.

```python
from lilax.diagnostics.goodput import GoodputEvent, create_goodput_recorder, maybe_record_goodput

runtime.resolve_config()
recorder = create_goodput_recorder(runtime.config)
with maybe_record_goodput(recorder, GoodputEvent.DATA_LOADING):
	batch = load_batch()
```

Recording follows that runtime's goodput settings. Disabled recording returns no recorder. The context writes the end event only when the body completes successfully.

## Collect request diagnostics

```python
from lilax.evaluation.diagnostics import SimpleEvalsDebugCollector

collector = SimpleEvalsDebugCollector()
collector.set_task("application")
record = collector.begin([{"role": "user", "content": "Explain checkpoints."}])
record["status"] = "success"
collector.finish(record)
```

Call `begin` and `finish` around your actual request. The example shows the record lifecycle only; it does not send a request. See [diagnostic reference](../../reference/execution/diagnostics.md).

## Write application metrics

Constructing `MetricLogger` can create TensorBoard writers, a W&B run, managed cloud diagnostics, and monitoring threads when those services are enabled. Use a separate application-owned logger only when the trainer does not already own logging for the run.

This example continues the configuration setup above. It writes an application's existing evaluation summary and closes the logger's writers. Supply `eval_metrics` from your evaluation, including `scalar` keys `eval/avg_loss`, `eval/avg_perplexity`, and `eval/total_weights`. MoE and MTP models require their corresponding summary keys described in the [metric logging reference](../../reference/execution/diagnostics.md#training-metric-logging). Supply your learning-rate callable as `learning_rate_schedule` and completed step as `step`.

```python
from lilax.diagnostics.metrics import MetricLogger

logger = MetricLogger(config, learning_rate_schedule)
try:
	logger.write_metrics(eval_metrics, step, metric_type="eval")
finally:
	logger.flush_metrics_and_cleanup()
```

For activation statistics, supply the activation tree collected by your model as `intermediate_outputs`. Leaf keys are `activation_fraction_zero`, `activation_mean`, and `activation_stdev`. Each present statistic needs one value per decoder layer, either stacked or in separate layer leaves. The function adds `activ_fraction_zero/layer_NNN`, `activ_mean/layer_NNN`, and `activ_stdev/layer_NNN` for the corresponding collected leaves.

```python
from lilax.diagnostics.metrics import record_activation_metrics

activation_metrics = {"scalar": {}}
record_activation_metrics(activation_metrics, intermediate_outputs, config)
print(activation_metrics["scalar"])
```

## Upload TensorBoard logs to Vertex AI

Supply your cloud project as `project_id`, region as `region`, run name as `run_name`, and output root as `base_output_directory`. The following copies the model's runtime-source configuration and resolves it through a separate `Runtime`. Resolution initializes JAX without constructing a mesh and returns that runtime. Its `config.tensorboard_dir` is derived as `<base_output_directory>/<run_name>/tensorboard/` for a nonempty output root. Write logs to that directory.

`configure_vertex_tensorboard` fills missing `TENSORBOARD_*` and `EXPERIMENT_NAME` environment values on process zero. With `use_vertex_tensorboard=True`, it creates cloud resources and starts background upload. An existing `UPLOAD_DATA_TO_TENSORBOARD` environment value can also start upload. Cloud credentials and permissions are required. Decoupled execution skips configuration.

```python
from lilax.diagnostics.tensorboard import VertexTensorboardManager
from lilax.runtime import Runtime

vertex_source = model.runtime.public_config.model_copy(update={
	"vertex_tensorboard_project": project_id, "vertex_tensorboard_region": region,
	"run_name": run_name, "base_output_directory": base_output_directory, "use_vertex_tensorboard": True,
})
vertex_runtime = Runtime(vertex_source).resolve_config()
vertex_config = vertex_runtime.config
manager = VertexTensorboardManager()
manager.configure_vertex_tensorboard(vertex_config)
```

Keep `manager` alive while logs are written. Destruction stops an active uploader. See [VertexTensorboardManager](../../reference/execution/diagnostics.md#vertextensorboardmanager) for direct setup and upload operations.

## Save and upload evaluation results

`write_results` creates a timestamped local JSON file. A `gs://` result destination also uploads a temporary local copy. Supply your actual `scores`, `generation_stats`, and JSON-serializable `evaluation_config` mappings. The configuration argument here is a dictionary, not the runtime configuration object.

```python
from lilax.evaluation.reporting import write_results

artifact = write_results(
	benchmark="application", model_name=model.config.model_name, scores=scores,
	generation_stats=generation_stats, config=evaluation_config, results_path="./eval_results",
)
print(artifact["local_path"])
```

To upload that file separately, set `gcs_destination` to your writable `gs://` object path or directory ending in `/`. This call writes to Cloud Storage and requires cloud credentials.

```python
from lilax.evaluation.reporting import upload_results

upload_results(artifact["local_path"], gcs_destination)
```

## Build a request report

Continue with the collector above. Supply your actual simple-evals result mapping as `eval_results`, task durations in seconds as `task_durations`, and the request's completion and sequence limits as `max_tokens` and `max_model_len`. This builds text in memory without writing a file or sending requests.

```python
from lilax.evaluation.diagnostics import build_debug_report

report = build_debug_report(
	collector, eval_results, task_durations, model.config.model_name, max_tokens, max_model_len,
)
print(report)
```

See [structured evaluation reports](../../reference/execution/diagnostics.md#structured-evaluation-reports) for record fields and file writing, and [evaluation JSON](../../reference/execution/diagnostics.md#save-evaluation-json) for result destinations.
