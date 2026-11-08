# Run cluster benchmark variants

These operations submit training jobs or signal running workers. Construct `cluster` and `workload` with the [cluster setup example](benchmarks.md#submit-cluster-workloads). Use credentials and images for the target cluster.

## Submit and observe completion

`submit` launches an external training job through Cluster Toolkit. `wait` obtains cluster credentials with gcloud and observes job completion through kubectl.

```python
from lilax.benchmarks.cluster.benchmark_runner import submit, wait

name, status = submit(cluster, workload)
results = list(wait(cluster, [name])) if status == 0 else [(name, status)]
print(results)
```

The returned statuses distinguish submission from job completion. `wait` yields jobs in completion order.

## Build a checkpoint workload

Supply `server_image`, `proxy_image`, and `runner_image` as your deployed container image URIs and `output_uri` as your writable output location. These builders return workloads without submitting them.

```python
from lilax.benchmarks.cluster import WorkloadConfig
from lilax.benchmarks.cluster.benchmark_recipes import (
	checkpoint_pathways_config, checkpoint_workloads, run_checkpoint_benchmarks,
)

pathways = checkpoint_pathways_config(
	server_image=server_image, proxy_server_image=proxy_image, runner_image=runner_image,
)
workloads = checkpoint_workloads(
	WorkloadConfig, cluster, pathways, base_output_directory=output_uri, runner_image=runner_image,
)
```

The default selection is the recipe's two-slice Llama3.1-70B synthetic workload on v6e. Supply `models` to change that selection.

With `test_restore=True`, the following call submits the workloads, waits for initial jobs, cancels each successful initial job, and submits a longer restoration job. It does not wait for the restoration jobs. Supply your cluster and the workloads returned by the builder above:

```python
run_checkpoint_benchmarks(cluster, workloads, test_restore=True)
```

## Select a headless or long run

`run_headless` starts a two-slice Pathways server and proxy without a training command. `run_long_running` selects two slices, ten million steps, and ten thousand restarts, and enables Vertex TensorBoard. Those are recipe behaviors, not tuning recommendations. See [recipe arguments](../../reference/benchmarks/cluster-workflows.md).

## Schedule worker disruptions

Supply `trigger_steps` as your selected training-step list for an `mcjax` workload. The following submits that workload and sends SIGTERM to selected worker processes. The manager obtains isolated Kubernetes credentials through gcloud.

```python
from lilax.benchmarks.cluster import run_cluster_benchmarks
from lilax.benchmarks.cluster.benchmark_disruption import (
	DisruptionMethod, cluster_disruption_manager, construct_disruption_configs,
)

workload.disruption_configs = construct_disruption_configs(
	"mcjax", DisruptionMethod.SIGTERM, {"step": trigger_steps},
)
with cluster_disruption_manager(cluster) as manager:
	submissions = run_cluster_benchmarks(cluster, [workload], disruption_manager=manager)
	manager.start_disruptions_and_wait_for_completion()
```

Use the framework matching your workload. Recovery configuration fields are retained in the records, but the manager does not implement recovery actions.

## Aggregate existing metrics

```python
from lilax.benchmarks.cluster.benchmark_metrics import parse_metrics

jsonl_text = '{"perf/per_device_tflops_per_sec": 1.0, "perf/per_device_tokens_per_sec": 2.0, "perf/step_time_seconds": 3.0}'
summary = parse_metrics(jsonl_text, total_steps=1)
print(summary.avg_tflops_per_sec, summary.avg_tokens_per_sec)
print(summary.median_step_time, summary.e2e_step_time)
```

These input numbers are illustrative log records, not measured results. The returned `Metrics` record contains averages of the two per-device rates, median step seconds, and summed step seconds. This computes a summary from supplied logs; it does not run a benchmark. See [metric upload contracts](../../reference/benchmarks/cluster-metrics.md).
