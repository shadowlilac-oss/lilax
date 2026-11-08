# Debugging and profiling profile fields

These optional YAML sections belong to an [execution profile](execution-profile.md). Only supplied keys override the workflow.

## `debugging`

Controls JAX initialization, caching, and debugging. Skipping distributed initialization requires an already appropriate execution environment. Checkify adds execution overhead.

<details>
<summary>Parameters</summary>

- `constant_bound_config` (`list`, optional, defaults to `[]`). Six AQT FP8 bounds in forward LHS, forward RHS, input-gradient LHS, input-gradient RHS, weight-gradient LHS, weight-gradient RHS order. Other lengths do not enable static bounds. An empty string becomes `[]`. A comma-separated string becomes floats.
- `jax_cache_dir` (`str | None`, optional, defaults to `"~/jax_cache"`). Persistent compilation cache.
- `jax_distributed_initialization_timeout` (`int`, optional, defaults to `300`). Initialization timeout in seconds.
- `jax_debug_log_modules` (`str`, optional, defaults to `""`). Verbose modules. `"jax"` selects JAX logging.
- `skip_jax_distributed_system` (`bool`, optional, defaults to `false`). Skip distributed initialization.
- `enable_single_controller` (`bool`, optional, defaults to `false`). Pathways single-controller mode. `process.worker_arrangement: single_controller` also selects this mode.
- `subslice_shape` (`str`, optional, defaults to `""`). Pathways shape as `x,y,z`.
- `max_checkify` (`bool`, optional, defaults to `false`). Add JAX checkify checks.

</details>

## `hlo`

The default local directories use the process's system temporary directory, rather than a fixed `/tmp` path.

Enabled dumps write local files and can upload them to GCS, then delete the local copies.

<details>
<summary>Parameters</summary>

- `dump_hlo` (`bool`, optional, defaults to `false`). Enable HLO dumps.
- `dump_step` (`int`, optional, defaults to `-1`). Dump step. `-1` disables step-specific dumping.
- `dump_hlo_local_dir` (`str`, optional, defaults to `<system temp>/xla_dump/`). Local HLO destination.
- `dump_hlo_delete_local_after` (`bool`, optional, defaults to `true`). Delete local HLO after GCS upload.
- `dump_hlo_gcs_dir` (`str`, optional, defaults to `""`). HLO upload destination.
- `dump_hlo_module_name` (`str`, optional, defaults to `"jit_train_step"`). Dump module filter.
- `dump_hlo_local_module_name` (`str`, optional, defaults to `"jit_train_step"`). Local-save module filter.
- `dump_hlo_xla_flags` (`str`, optional, defaults to `""`). Additional dump XLA options.
- `dump_hlo_upload_all` (`bool`, optional, defaults to `false`). Upload from every host.
- `dump_jaxpr` (`bool`, optional, defaults to `false`). Enable JAXPR dumps.
- `dump_jaxpr_local_dir` (`str`, optional, defaults to `<system temp>/jaxpr_dump/`). Local JAXPR destination.
- `dump_jaxpr_delete_local_after` (`bool`, optional, defaults to `true`). Delete local JAXPR after GCS upload.
- `dump_jaxpr_gcs_dir` (`str`, optional, defaults to `""`). JAXPR upload destination.

</details>

## `profiling`

Controls step-window capture and the separate live JAX profiler server. Clean profiling adds device synchronization. TPU power events must remain disabled for GPU XPlane traces.

<details>
<summary>Parameters</summary>

- `profiler` (`ProfilerType`, optional, defaults to `""`). Empty disables step profiling. `xplane` selects XPlane, `nsys` selects NVIDIA profiling.
- `upload_all_profiler_results` (`bool`, optional, defaults to `false`). Upload all hosts' traces.
- `skip_first_n_steps_for_profiler` (`int`, optional, defaults to `1`). Steps before capture.
- `profiler_steps` (`int`, optional, defaults to `5`). Steps per window.
- `profile_cleanly` (`bool`, optional, defaults to `true`). Wait for device readiness at step boundaries.
- `profile_periodically_period` (`int`, optional, defaults to `-1`). Positive values repeat capture every N steps. Otherwise capture is not periodic.
- `hide_profiler_step_metric` (`bool`, optional, defaults to `false`). Hide metrics at capture activation and deactivation boundaries.
- `enable_continuous_profiling` (`bool`, optional, defaults to `false`). Support captures above 2 GB.
- `enable_jax_profiler` (`bool`, optional, defaults to `false`). Start the live JAX profiler server.
- `jax_profiler_port` (`int`, optional, defaults to `9999`). Live JAX profiler port.
- `enable_tpu_profiling_options` (`bool`, optional, defaults to `false`). Apply advanced TPU options.
- `tpu_num_chips_to_profile_per_task` (`int`, optional, defaults to `1`). Chips per task.
- `tpu_num_sparse_cores_to_trace` (`int`, optional, defaults to `2`). SparseCores to trace.
- `tpu_num_sparse_core_tiles_to_trace` (`int`, optional, defaults to `1`). Tiles per SparseCore.
- `xprof_tpu_power_trace_level` (`XProfTPUPowerTraceMode`, optional, defaults to `0`). `0` disables power tracing, `1` selects normal tracing, `2` selects SPI tracing.
- `xprof_e2e_enable_fw_throttle_event` (`bool`, optional, defaults to `false`). Capture firmware throttle events.
- `xprof_e2e_enable_fw_power_level_event` (`bool`, optional, defaults to `false`). Capture firmware power-level events.
- `xprof_e2e_enable_fw_thermal_event` (`bool`, optional, defaults to `false`). Capture firmware thermal events.
- `profile_power_events` (`bool`, optional, defaults to `false`). Enable TPU power and thermal options.

</details>

## `xprof` and `rl_profiling`

The analysis server, the live JAX profiler server, and the step-window capture are separate controls. Enabling one does not select the others.

RL host limits greater than one require XPlane and an active Pathways runtime. The XProf analysis server terminates when the training attempt ends.

The analysis server binds to `127.0.0.1` on process zero. Remote use requires port forwarding. The `rl_profiling` section contributes overrides only for RL runtimes.

<details>
<summary>Parameters</summary>

- `xprof.launch_server` (`bool`, optional, defaults to `false`). Start the TensorBoard analysis server for a training attempt.
- `xprof.port` (`int`, optional, defaults to `8791`). Analysis server port.
- `rl_profiling.profiler_max_num_hosts` (`int`, optional, defaults to `1`). Positive Pathways XPlane host limit across trainer and sampler. `1` traces worker zero. Larger values trace hosts across slices.

</details>
