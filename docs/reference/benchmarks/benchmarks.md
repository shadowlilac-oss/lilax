# Specialized benchmarks

See [benchmark guide](../../guides/benchmarks/benchmarks.md).

`DisruptionManager` is available from `lilax.benchmarks.cluster.benchmark_disruption`.

## Cluster submission

Import from `lilax.benchmarks.cluster`.

### ClusterConfig

Import from `lilax.benchmarks.cluster`. Dataclass.

```python
ClusterConfig(*, cluster_name: str, project: str, device_type: str, zone: str = "",
	location: str | None = None, compute_type: str | None = None, topology: str | None = None)
```

Describes the cluster used for submission.

<details>
<summary>Parameters</summary>

- `cluster_name` (`str`). The cluster.
- `project` (`str`). The cloud project.
- `device_type` (`str`). Accelerator hardware.
- `zone` (`str`, optional, defaults to `""`). The fallback location.
- `location` (`str | None`, optional, defaults to `None`). Cluster location override for `zone`.
- `compute_type` (`str | None`, optional, defaults to `None`). Workload compute type override.
- `topology` (`str | None`, optional, defaults to `None`). Workload topology override.

</details>

### BenchmarkModel

Import from `lilax.benchmarks.cluster`. Dataclass.

```python
BenchmarkModel(model_name: str, model_type: str, tuning_params: dict, xla_flags: str,
	pathways_tuning_params: dict | None = None, pathways_xla_flag_options: dict | None = None)
```

Describes the submitted benchmark, not a loaded model.

<details>
<summary>Parameters</summary>

- `model_name` (`str`). Workload model name.
- `model_type` (`str`). Its model identity.
- `tuning_params` (`dict`). Training overrides.
- `xla_flags` (`str`). Compilation flags.
- `pathways_tuning_params` (`dict | None`, optional, defaults to `None`). Pathways overrides.
- `pathways_xla_flag_options` (`dict | None`, optional, defaults to `None`). Pathways flag modifications.

</details>

### WorkloadConfig

Import from `lilax.benchmarks.cluster`. Dataclass.

`class lilax.benchmarks.cluster.WorkloadConfig` is a dataclass with keyword-only fields below.
Device counts and hardware identity are derived. `num_devices_per_slice` is derived from selected hardware topology.
BigQuery uploads require a device type.

<details>
<summary>Parameters</summary>

- `model` (`BenchmarkModel | None`). Submitted benchmark model.
- `base_output_directory` (`str`). Output root.
- `device_type` (`str | None`, optional, defaults to `None`). Hardware.
- `num_slices` (`int | None`, optional, defaults to `None`). Slice count.
- `base_docker_image` (`str | None`, optional, defaults to `None`). The container.
- `num_steps` (`int`, optional, defaults to `20`). Training steps.
- `max_restarts` (`int`, optional, defaults to `0`). Restart allowance.
- `priority` (`str`, optional, defaults to `"medium"`). Job priority.
- `run_name` (`str | None`, optional, defaults to `None`). Workload name, or `None` for generated naming.
- `compute_type` (`str | None`, optional, defaults to `None`). Derived compute type override.
- `topology` (`str | None`, optional, defaults to `None`). Derived topology override.
- `mounts` (`list[str]`, optional, defaults to `[]`). Mount specifications. Each instance receives a fresh value.
- `pathways_config` (`PathwaysConfig | None`, optional, defaults to `None`). Pathways setup.
- `hlo_dump` (`bool`, optional, defaults to `False`). Flag for HLO output.
- `skip_validation` (`bool`, optional, defaults to `False`). Flag to skip submission prerequisite validation.
- `libtpu_type` (`str`, optional, defaults to `"maxtext-docker"`). TPU library setup.
- `libtpu_nightly_version` (`str`, optional, defaults to `""`). A nightly TPU library.
- `generate_metrics_and_upload_to_big_query` (`bool`, optional, defaults to `False`). Flag for metrics upload.
- `db_project` (`str`, optional, defaults to `""`). The BigQuery project.
- `db_dataset` (`str`, optional, defaults to `""`). The BigQuery dataset.
- `disruption_configs` (`list | None`, optional, defaults to `None`). Disruption schedules.

</details>

### PathwaysConfig

Import from `lilax.benchmarks.cluster`. Dataclass.

`class lilax.benchmarks.cluster.PathwaysConfig` is a dataclass with optional keyword-only fields below.

<details>
<summary>Parameters</summary>

- `server_image` (`str | None`, optional, defaults to `None`). The server image.
- `proxy_server_image` (`str | None`, optional, defaults to `None`). The proxy image.
- `runner_image` (`str | None`, optional, defaults to `None`). The runner image.
- `colocated_python_sidecar_image` (`str | None`, optional, defaults to `None`). The sidecar image.
- `server_flags` (`str`, optional, defaults to `""`). Server flags.
- `proxy_flags` (`str`, optional, defaults to `""`). Proxy flags.
- `worker_flags` (`str`, optional, defaults to `""`). Worker flags.
- `headless` (`bool`, optional, defaults to `False`). Flag for server and proxy startup without training.

</details>

### run_cluster_benchmarks

```python
run_cluster_benchmarks(cluster: ClusterConfig, workloads: list[WorkloadConfig], *, user: str | None = None,
	disruption_manager=None, wait: bool = False, timeout: str = "-1s") -> list[tuple[str, int]]
```

Submits workloads. Submission and optional disruption management have external side effects.

Returns `(name, status_code)` pairs, using completion status for successful submissions when `wait=True` and submission status otherwise.

<details>
<summary>Parameters</summary>

- `cluster` (`ClusterConfig`). The destination cluster.
- `workloads` (`list[WorkloadConfig]`). Job configurations.
- `user` (`str | None`, optional, defaults to `None`). Generated-name ownership.
- `disruption_manager` (`DisruptionManager | None`, optional, defaults to `None`). Manager for configured disruptions.
- `wait` (`bool`, optional, defaults to `False`). Flag to wait for each successful submission.
- `timeout` (`str`, optional, defaults to `"-1s"`). Completion wait timeout.

</details>

## Matrix operations

Import from `lilax.benchmarks.mlperf.matmul`.

### MatmulDtypesConfig

Import from `lilax.benchmarks.mlperf.matmul`. Dataclass.

```python
MatmulDtypesConfig(num_matmuls: int = 250, matrix_size: int = 2048, enable_profile: bool = False)
```

Selects the dtype benchmark.

<details>
<summary>Parameters</summary>

- `num_matmuls` (`int`, optional, defaults to `250`). Batched products.
- `matrix_size` (`int`, optional, defaults to `2048`). Square matrix dimensions.
- `enable_profile` (`bool`, optional, defaults to `False`). Flag for JAX traces.

</details>

### run_matmul_dtypes

```python
run_matmul_dtypes(config: MatmulDtypesConfig)
```

Runs dtype combinations and prints results.

Returns `None`.

<details>
<summary>Parameters</summary>

- `config` (`MatmulDtypesConfig`). Matrix and profiling settings.

</details>

### MatmulShardingConfig

Import from `lilax.benchmarks.mlperf.matmul`. Dataclass.

```python
MatmulShardingConfig(enable_visual: bool = False)
```

Selects sharding setup output.

<details>
<summary>Parameters</summary>

- `enable_visual` (`bool`, optional, defaults to `False`). Flag for array sharding visualizations.

</details>

### run_matmul_sharding

```python
run_matmul_sharding(config: MatmulShardingConfig)
```

Uses fixed eight-device layouts. The underlying `matmul` constructs arrays and a JIT function but does not execute that function or report timing.
Returns `None`.

<details>
<summary>Parameters</summary>

- `config` (`MatmulShardingConfig`). Visualization settings.

</details>

### matmul

```python
matmul(mesh, mesh_dim, dtype=jax.numpy.bfloat16, batch=1024, enable_visual=False)
```

Places specialized matrix-workload arrays on the mesh and defines the JIT computation. Does not execute it.

Returns `None`.

<details>
<summary>Parameters</summary>

- `mesh` (`jax.sharding.Mesh`). The existing JAX mesh with a `model` axis.
- `mesh_dim` (`tuple[int, int]`). Unused argument.
- `dtype` (`type`, optional, defaults to `jax.numpy.bfloat16`). Array dtype.
- `batch` (`int`, optional, defaults to `1024`). Activation rows.
- `enable_visual` (`bool`, optional, defaults to `False`). Flag for sharding visualizations.

</details>

### matmuls

```python
matmuls(mesh, mesh_dim, enable_visual=False)
```

Calls `matmul` for fixed dtype and prefill-length selections.

Returns `None`.

<details>
<summary>Parameters</summary>

- `mesh` (`jax.sharding.Mesh`). The existing JAX mesh.
- `mesh_dim` (`tuple[int, int]`). Layout argument forwarded to `matmul`.
- `enable_visual` (`bool`, optional, defaults to `False`). Flag for visualizations.

</details>

### simple_timeit

```python
simple_timeit(f, *args, tries=10, task=None, enable_profile=False)
```

Returns average elapsed seconds after a warmup call. A task name is required.
Creates a temporary trace directory and optionally records a JAX profile.

<details>
<summary>Parameters</summary>

- `f` (`Callable`). The callable to time.
- `*args` (`tuple[Any, ...]`). Positional arguments for the callable.
- `tries` (`int`, optional, defaults to `10`). Measured calls.
- `task` (`str | None`, optional, defaults to `None`). Required task name.
- `enable_profile` (`bool`, optional, defaults to `False`). Flag for JAX profiling.

</details>

## Sharding estimates

Import from `lilax.benchmarks.sharding.analysis`.

### calculate_matmul_resources

```python
calculate_matmul_resources(activations_shape: tuple[int, ...], weights_shape: tuple[int, ...],
	ici_bandwidth: float, peak_flops: float, sD: int = 1, sK: int = 1, sW: int = 1, sF: int = 1,
	sE: int = 1, activation_size_bytes: int = 2, weight_size_bytes: int = 2,
	ici_latency: float = 1e-6, all_gather_axes: Sequence[str] = tuple(), debug=True) -> dict[str, float]
```

Estimates compute, communication, and memory. The result is an estimate, not a measured benchmark.

`ici_bandwidth` is bytes per second, `peak_flops` is FLOPs per second, and `ici_latency` is seconds. The returned dictionary has `t_flops` and `t_comms` in seconds, and `memory_per_TPU_bytes` in bytes per device.

For a matrix product, use activation shape `(M, K)` and weight shape `(K, F)` or grouped weight shape `(G, K, F)`. The final activation dimension must match the weight's `K` dimension. Weight ranks other than two or three raise `ValueError`.

All shard factors must be at least one. `sK` and `sF` cannot both exceed one. If `sK` and `sW` both exceed one, they must match. Nondivisible dimensions produce warnings and approximate estimates. `all_gather_axes` accepts names `D`, `K`, `W`, `F`, and `E`.

<details>
<summary>Parameters</summary>

- `activations_shape` (`tuple[int, ...]`). `(M, K)` activation dimensions.
- `weights_shape` (`tuple[int, ...]`). `(K, F)` or `(G, K, F)` dimensions.
- `ici_bandwidth` (`float`). Hardware bandwidth in bytes per second.
- `peak_flops` (`float`). Hardware FLOPs per second.
- `sD` (`int`, optional, defaults to `1`). Data shard count.
- `sK` (`int`, optional, defaults to `1`). Contraction dimension shard count.
- `sW` (`int`, optional, defaults to `1`). Weight shard count.
- `sF` (`int`, optional, defaults to `1`). Output feature shard count.
- `sE` (`int`, optional, defaults to `1`). Expert shard count.
- `activation_size_bytes` (`int`, optional, defaults to `2`). Bytes per activation element.
- `weight_size_bytes` (`int`, optional, defaults to `2`). Bytes per weight element.
- `ici_latency` (`float`, optional, defaults to `1e-6`). Communication latency in seconds.
- `all_gather_axes` (`Sequence[str]`, optional, defaults to `()`). Gathered axis names among `D`, `K`, `W`, `F`, and `E`.
- `debug` (`bool`, optional, defaults to `True`). Flag for diagnostic output.

</details>

### latency_bound_comms

```python
latency_bound_comms(comm: float, latency=1e-6)
```

Returns the larger of communication time and latency.

<details>
<summary>Parameters</summary>

- `comm` (`float`). Communication seconds.
- `latency` (`float`, optional, defaults to `1e-6`). Minimum seconds.

</details>

### plot_sharding_scheme_comparison

```python
plot_sharding_scheme_comparison(calc_resource_func, activations_shape, weights_shape, sharding_schemes: list[dict])
```

Plots estimates for supplied sharding schemes.

<details>
<summary>Parameters</summary>

- `calc_resource_func` (`Callable`). The resource estimator.
- `activations_shape` (`tuple[int, ...]`). Activation dimensions.
- `weights_shape` (`tuple[int, ...]`). Weight dimensions.
- `sharding_schemes` (`list[dict]`). Scheme configurations to compare.

</details>
