# Run specialized benchmarks

Use separate benchmark APIs for matrix operations, sharding estimates, and cluster workload submission. Cluster submission launches external workloads.

## Run a matrix dtype benchmark

This allocates matrices and runs the configured dtype combinations. Its default workload uses 250 matrices of size 2048. Change the configuration before running if those allocations do not fit your device.

```python
from lilax.benchmarks.mlperf.matmul import MatmulDtypesConfig, run_matmul_dtypes

run_matmul_dtypes(MatmulDtypesConfig())
```

## Estimate sharding resources

Supply `ici_bytes_per_second` as your system's actual link bandwidth in bytes per second and `peak_flops_per_second` as its peak compute rate in FLOPs per second. The small shapes illustrate the interface, not a model workload or tuning recommendation.

```python
from lilax.benchmarks.sharding.analysis import calculate_matmul_resources

estimate = calculate_matmul_resources(
	(8, 16), (16, 32), ici_bandwidth=ici_bytes_per_second,
	peak_flops=peak_flops_per_second, sD=1, sK=1, sW=1, sF=1, sE=1, debug=False,
)
print(estimate["t_flops"], estimate["t_comms"], estimate["memory_per_TPU_bytes"])
```

The printed fields contain estimated compute seconds, communication seconds, and bytes per device. These are calculated estimates, not measured performance. See [shape and sharding restrictions](../../reference/benchmarks/benchmarks.md#sharding-estimates).

## Submit cluster workloads

Construct the cluster and workload with the public classes. Supply `cluster_name`, `project_id`, and `cluster_location` for your existing deployment, `device_type` for its accelerator hardware, `runner_image` as a deployed container image URI containing Lilax and its accelerator dependencies, and `output_uri` as writable output storage. Configure credentials for Cluster Toolkit, gcloud, and kubectl in your environment.

This example uses the `default-basic-1` synthetic recipe's model settings and a one-slice workload. These settings are recipe values, not tuning recommendations. `dataset_uri` is your deployment's dataset location retained in the recipe settings; this workload selects synthetic data.

```python
from lilax.benchmarks.cluster import BenchmarkModel, ClusterConfig, WorkloadConfig, run_cluster_benchmarks

cluster = ClusterConfig(
	cluster_name=cluster_name, project=project_id, location=cluster_location, device_type=device_type,
)
benchmark_model = BenchmarkModel(
	model_name="default-basic-1", model_type="default", xla_flags="",
	tuning_params={
		"per_device_batch_size": 1, "remat_policy": "full", "global_parameter_scale": 1,
		"attention": "flash", "dataset_path": dataset_uri, "dataset_type": "synthetic",
		"reuse_example_batch": 1, "enable_checkpointing": False, "profiler": "xplane",
	},
)
workload = WorkloadConfig(
	model=benchmark_model, device_type=cluster.device_type, num_slices=1,
	base_output_directory=output_uri, base_docker_image=runner_image,
)
workloads = [workload]
```

The following call submits external training jobs through Cluster Toolkit, obtains cluster credentials with gcloud, and waits for completion through kubectl. The jobs allocate accelerator resources, write to `output_uri`, and run the recipe's profiler.

```python
results = run_cluster_benchmarks(cluster, workloads, wait=True)
print(results)
```

`cluster` is a `ClusterConfig`. Each workload is a `WorkloadConfig` with a `BenchmarkModel`. Workload tuning belongs to the specific benchmark recipe. A benchmark model descriptor is separate from a loaded language model.

The result lists `(workload_name, status_code)` pairs. With `wait=True`, a successful submission returns its completion status, with zero for completed jobs. A failed submission returns its submission status. `wait=False` returns submission statuses without waiting. See [benchmark reference](../../reference/benchmarks/benchmarks.md).
