# Cluster benchmark operations

See [cluster workflow](../../guides/benchmarks/cluster-workflows.md) and [configuration constructors](benchmarks.md).

`ClusterConfig`, `WorkloadConfig`, and `PathwaysConfig` are available from `lilax.benchmarks.cluster`. `datetime.datetime` is the standard-library timestamp type.

## Submission and device execution

Import from `lilax.benchmarks.cluster.benchmark_runner`.

### submit

```python
submit(cluster_config, workload_config, *, workload_name=None, user=None) -> tuple[str, int]
```

Sets a workload name and invokes Cluster Toolkit.

Returns `(name, submission_status)`.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). Cluster settings.
- `workload_config` (`WorkloadConfig`). The submitted workload.
- `workload_name` (`str`, optional, defaults to `None`). Workload name override.
- `user` (`str | None`, optional, defaults to `None`). Generated-name ownership.

</details>

### wait

```python
wait(cluster_config, workload_names)
```

Yields `(name, completion_status)` pairs in completion order.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). Cluster access settings.
- `workload_names` (`list[str]`). Submitted names to wait for.

</details>

### cancel

```python
cancel(cluster_config, workload_name) -> int
```

Cancels through `gcluster job cancel`.

Returns its exit code.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). Cluster access settings.
- `workload_name` (`str`). The workload to cancel.

</details>

### run_on_device

```python
run_on_device(workloads: list[WorkloadConfig]) -> None
```

Runs training commands sequentially and assigns missing names. Subprocess failure propagates.

Returns `None`.

<details>
<summary>Parameters</summary>

- `workloads` (`list[WorkloadConfig]`). Local training workloads.

</details>

### run_healthscan

```python
run_healthscan(device_type: str, base_output_directory: str, num_steps: int = 20) -> None
```

Selects the supported TPU healthscan model and runs it on the current device.

Returns `None`.

<details>
<summary>Parameters</summary>

- `device_type` (`str`). Supported TPU hardware.
- `base_output_directory` (`str`). Workload output.
- `num_steps` (`int`, optional, defaults to `20`). Training steps.

</details>

### upload_workload_metrics

```python
upload_workload_metrics(workload: WorkloadConfig)
```

Uploads only from Pathways or worker zero, using the workload's BigQuery destination.
Returns the metric uploader's result. See [upload side effects](cluster-metrics.md#upload_benchmark_metrics).

<details>
<summary>Parameters</summary>

- `workload` (`WorkloadConfig`). Run metadata and the destination.

</details>

### run_cluster_benchmarks

```python
run_cluster_benchmarks(cluster: ClusterConfig, workloads: list[WorkloadConfig], *, user: str | None = None,
	disruption_manager=None, wait: bool = False, timeout: str = "-1s") -> list[tuple[str, int]]
```

Registers disruptions with a supplied manager and submits jobs.

Returns submission or waited completion status pairs.

<details>
<summary>Parameters</summary>

- `cluster` (`ClusterConfig`). The destination cluster.
- `workloads` (`list[WorkloadConfig]`). Submitted jobs.
- `user` (`str | None`, optional, defaults to `None`). Generated-name ownership.
- `disruption_manager` (`DisruptionManager | None`, optional, defaults to `None`). Manager for disruption schedules.
- `wait` (`bool`, optional, defaults to `False`). Flag to wait for successful submissions.
- `timeout` (`str`, optional, defaults to `"-1s"`). Completion wait timeout.

</details>

## Workload recipes

Import from `lilax.benchmarks.cluster.benchmark_recipes`.

### run_benchmarks

```python
run_benchmarks(cluster_config, workload_configs, *, user=None) -> int
```

Submits without waiting.

Returns zero or one.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The destination.
- `workload_configs` (`list[WorkloadConfig]`). Workloads.
- `user` (`str | None`, optional, defaults to `None`). Generated-name ownership.

</details>

### run_headless

```python
run_headless(cluster_config, workload_config, *, workload_name, user=None) -> int
```

Requires Pathways configuration. Forces two slices, zero restarts, and headless mode.

Returns submission status.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The destination.
- `workload_config` (`WorkloadConfig`). A Pathways workload.
- `workload_name` (`str`). The submitted name.
- `user` (`str | None`, optional, defaults to `None`). Ownership.

</details>

### run_long_running

```python
run_long_running(cluster_config, workload_config, *, project, region, user=None) -> int
```

Enables Vertex TensorBoard and submits two slices, 10,000,000 steps, 10,000 restarts, and medium priority.
Returns submission status.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The destination.
- `workload_config` (`WorkloadConfig`). The model workload.
- `project` (`str`). The Vertex TensorBoard project.
- `region` (`str`). The TensorBoard region.
- `user` (`str | None`, optional, defaults to `None`). Ownership.

</details>

### run_checkpoint_benchmarks

```python
run_checkpoint_benchmarks(cluster_config, workload_configs, *, test_restore=False, restore_steps=20, user=None) -> int
```

Waits for successful submissions. Optional restore cancels and resubmits the same name for additional steps.
Returns zero or one. Restore completion is not awaited.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The destination.
- `workload_configs` (`list[WorkloadConfig]`). Initial jobs.
- `test_restore` (`bool`, optional, defaults to `False`). Flag for resubmission for restoration.
- `restore_steps` (`int`, optional, defaults to `20`). Additional steps for restored runs.
- `user` (`str | None`, optional, defaults to `None`). Ownership.

</details>

### checkpoint_pathways_config

```python
checkpoint_pathways_config(*, server_image, proxy_server_image, runner_image)
```

Returns a Pathways configuration with metrics collection enabled.

<details>
<summary>Parameters</summary>

- `server_image` (`str`). The server container.
- `proxy_server_image` (`str`). The proxy container.
- `runner_image` (`str`). The runner container.

</details>

### checkpoint_workloads

```python
checkpoint_workloads(workload_type, cluster_config, pathways_config, models=None, *,
	base_output_directory, runner_image, resume_checkpoints=None, now=None)
```

Returns workload records. Default selection uses two slices and 41 steps.

<details>
<summary>Parameters</summary>

- `workload_type` (`type[WorkloadConfig]`). Workload class, `WorkloadConfig` for this workflow.
- `cluster_config` (`ClusterConfig`). Hardware settings.
- `pathways_config` (`PathwaysConfig`). Pathways settings.
- `models` (`dict[str, dict] | None`, optional, defaults to `None`). Mapping of framework names to `{"models": [...], "num_slices_list": [...]}`. Omission selects built-in jobs.
- `base_output_directory` (`str`). Output storage.
- `runner_image` (`str`). The training container.
- `resume_checkpoints` (`dict[str, dict[int, dict]] | None`, optional, defaults to `None`). Mapping of framework and slice count to `base_output_directory`, `run_name`, and `num_steps`.
- `now` (`datetime.datetime | None`, optional, defaults to `None`). The output-directory timestamp. Omission uses current time.

</details>

### run_remote_python_sidecar

```python
run_remote_python_sidecar(cluster_config, *, user: str) -> int
```

Submits the recipe's one-slice Pathways sidecar with user-derived image paths and output bucket.

Returns submission status.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The destination and project.
- `user` (`str`). User-derived images and bucket names.

</details>

## Recipe configuration boundary

The following operations require a prepared `BenchmarkRecipeConfig`. `ClusterConfig` and `WorkloadConfig` are not interchangeable with it. For a workflow using public configuration constructors, use [cluster setup and submission](../../guides/benchmarks/benchmarks.md#submit-cluster-workloads).

### standard_workloads

```python
standard_workloads(config: BenchmarkRecipeConfig, *, disruption_method=None, disruptions=None) -> list[WorkloadConfig]
```

Returns workload records for selected frameworks, models, and slice counts.

<details>
<summary>Parameters</summary>

- `config` (`BenchmarkRecipeConfig`). Recipe selections.
- `disruption_method` (`DisruptionMethod | None`, optional, defaults to `None`). The signal.
- `disruptions` (`dict[str, list[int]] | None`, optional, defaults to `None`). Trigger lists.

</details>

### run_standard_benchmark

```python
run_standard_benchmark(config: BenchmarkRecipeConfig) -> int
```

Accesses or creates an output bucket and submits jobs.

Returns zero or one.
BigQuery enabled without a project and dataset returns one.

<details>
<summary>Parameters</summary>

- `config` (`BenchmarkRecipeConfig`). Cloud access, output, and workload selections.

</details>

### run_elastic_training

```python
run_elastic_training(config: BenchmarkRecipeConfig, disruption_manager=None, *, compare_with_mcjax=True) -> int
```

Schedules SIGILL at 120 and 600 seconds with ten restarts and starts disruption monitoring.
Returns zero or one. BigQuery enabled without a project and dataset returns one.

<details>
<summary>Parameters</summary>

- `config` (`BenchmarkRecipeConfig`). Recipe selections.
- `disruption_manager` (`DisruptionManager | None`, optional, defaults to `None`). An existing manager. Omission acquires one.
- `compare_with_mcjax` (`bool`, optional, defaults to `True`). Flag for inclusion of McJAX alongside Pathways.

</details>

### run_suspend_resume

```python
run_suspend_resume(config: BenchmarkRecipeConfig, disruption_manager=None) -> int
```

Schedules SIGTERM at step three with three restarts and starts disruption monitoring.
Returns zero or one. BigQuery enabled without a project and dataset returns one.

<details>
<summary>Parameters</summary>

- `config` (`BenchmarkRecipeConfig`). Recipe selections.
- `disruption_manager` (`DisruptionManager | None`, optional, defaults to `None`). An existing manager. Omission acquires one.

</details>

### check_and_create_bucket

```python
check_and_create_bucket(storage_client, bucket_name, region)
```

Can create cloud resources.

Returns a bucket, or `None` for handled access, creation, or conflict errors.

<details>
<summary>Parameters</summary>

- `storage_client` (`google.cloud.storage.Client`). A Cloud Storage client.
- `bucket_name` (`str`). The bucket.
- `region` (`str`). Creation location.

</details>

## Disruption records

Import from `lilax.benchmarks.cluster.benchmark_disruption`.

`TriggerType` contains `TIME_SECONDS` and `STEP`. `DisruptionMethod` contains `SIGILL` and `SIGTERM`. `RecoverMethod` contains `SIGTERM_METADATA_REMOVE`.

### DisruptionConfig

Import from `lilax.benchmarks.cluster.benchmark_disruption`. Dataclass.

```python
DisruptionConfig(name: str, trigger_type: TriggerType, trigger_value: int,
	disruption_method: DisruptionMethod, worker_container_name: str,
	recover_trigger_type: TriggerType | None = None, recover_trigger_value: int | None = None,
	recover_method: RecoverMethod | None = None, target_pod_regex: str = ".*(main-job|slice-job)-0-0.*",
	step_pod_regex: str = ".*(main-job|slice-job)-0-0.*")
```

Stores one schedule. Recovery fields do not cause recovery actions.

<details>
<summary>Parameters</summary>

- `name` (`str`). Schedule name.
- `trigger_type` (`TriggerType`). Time or step triggering.
- `trigger_value` (`int`). Seconds or step count.
- `disruption_method` (`DisruptionMethod`). The signal.
- `worker_container_name` (`str`). The container.
- `recover_trigger_type` (`TriggerType | None`, optional, defaults to `None`). Recovery trigger metadata.
- `recover_trigger_value` (`int | None`, optional, defaults to `None`). Recovery timing metadata.
- `recover_method` (`RecoverMethod | None`, optional, defaults to `None`). Recovery method metadata.
- `target_pod_regex` (`str`, optional, defaults to `".*(main-job|slice-job)-0-0.*"`). Target pods.
- `step_pod_regex` (`str`, optional, defaults to `".*(main-job|slice-job)-0-0.*"`). Pods for step monitoring.

</details>

### construct_disruption_configs

```python
construct_disruption_configs(framework: str, disruption_method: DisruptionMethod, disruptions: dict)
```

Returns disruption records. Framework selection determines container and pod patterns.

<details>
<summary>Parameters</summary>

- `framework` (`str`). Framework, `mcjax` or `pathways`.
- `disruption_method` (`DisruptionMethod`). A signal.
- `disruptions` (`dict`). Trigger lists under `time_seconds` or `step`.

</details>

### DisruptionManager

Import from `lilax.benchmarks.cluster.benchmark_disruption`. Class.

```python
DisruptionManager(api, namespace: str, *, api_client_factory=None)
```

Owns monitoring registrations. Matching triggers send signals to process 1 in selected containers.

<details>
<summary>Parameters</summary>

- `api` (`kubernetes.client.CoreV1Api`). A Kubernetes CoreV1 API.
- `namespace` (`str`). The Kubernetes namespace.
- `api_client_factory` (`Callable[[], kubernetes.client.ApiClient] | None`, optional, defaults to `None`). Factory for isolated worker API clients.

</details>

### add_workload

```python
disruption_manager.add_workload(workload_name: str, disruption_configs: list[DisruptionConfig]) -> None
```

Registers monitoring threads for a workload.

<details>
<summary>Parameters</summary>

- `workload_name` (`str`). The submitted workload.
- `disruption_configs` (`list[DisruptionConfig]`). Schedules.

</details>

### remove_workload

```python
disruption_manager.remove_workload(workload_name: str) -> None
```

Removes workload registrations.

<details>
<summary>Parameters</summary>

- `workload_name` (`str`). The registrations to remove.

</details>

### start_disruptions_and_wait_for_completion

```python
disruption_manager.start_disruptions_and_wait_for_completion() -> None
```

Starts and joins registered monitoring threads.

### cluster_disruption_manager

```python
cluster_disruption_manager(cluster_config)
```

Returns a context manager that acquires isolated gcloud credentials and yields a disruption manager.
Closes its client on exit.

<details>
<summary>Parameters</summary>

- `cluster_config` (`ClusterConfig`). The cluster name, location, and project for credential acquisition.

</details>
