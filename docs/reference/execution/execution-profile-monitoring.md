# Metrics and monitoring profile fields

These optional YAML sections belong to an [execution profile](execution-profile.md). Only supplied keys override the workflow.

## `metrics`

Controls metric collection and service startup.

<details>
<summary>Parameters</summary>

- `record_internal_nn_metrics` (`int`, optional, defaults to `0`). Collect internal neural-network metrics when nonzero.
- `prometheus_port` (`int`, optional, defaults to `0`). Prometheus port. Zero disables the server.
- `enable_checkpoint_cloud_logger` (`bool`, optional, defaults to `false`). Enable structured checkpoint logging.

</details>

## `goodput`

Recording writes job and work events. Monitoring computes and uploads time spent on productive work and overhead. They are separate switches. Process zero owns monitoring and recorder logging. Cloud logging, TensorBoard destinations, and cloud credentials must be available to the application.

The minimal recording profile is shown in [record diagnostics](../../guides/execution/diagnostics.md). It does not enable the uploader.

<details>
<summary>Parameters</summary>

- `enable_goodput_recording` (`bool`, optional, defaults to `false`). Create a recorder. Otherwise `create_goodput_recorder` returns `None`.
- `goodput_job_name` (`str`, optional, defaults to `""`). Job identity. Empty uses the run name.
- `monitor_goodput` (`bool`, optional, defaults to `false`). Start background upload in the owning context.
- `goodput_upload_interval_seconds` (`int`, optional, defaults to `30`). Seconds between cumulative uploads.
- `enable_pathways_goodput` (`bool`, optional, defaults to `false`). Pathways-aware monitoring.
- `monitor_step_time_deviation` (`bool`, optional, defaults to `true`). Include step-duration deviation.
- `step_deviation_interval_seconds` (`int`, optional, defaults to `30`). Seconds between deviation calculations.
- `enable_gcp_goodput_metrics` (`bool`, optional, defaults to `true`). Upload goodput to Cloud Monitoring.
- `enable_gcp_step_deviation_metrics` (`bool`, optional, defaults to `true`). Upload step deviation. Performance reporting through `monitoring` disables this duplicate stream.
- `rolling_windows_seconds` (`list[PositiveInt]`, optional, defaults to `[3600, 86400, 259200, 432000]`). Nonempty list of window durations in seconds.
- `enable_rolling_window_goodput` (`bool`, optional, defaults to `true`). Upload selected rolling windows while monitoring is active.

</details>

## `diagnostics`

Managed diagnostics requires a configured GCP environment and access to the selected region and storage destination.

On the RL path, `run_name` must be nonempty and either the diagnostics storage path or the base output directory must produce a `gs://` run directory. Programmatic XPlane profiling cannot run together with managed on-demand profiling there. Set `managed_mldiagnostics_on_demand_profiling: false` when selecting `profiler: xplane` for RL.

<details>
<summary>Parameters</summary>

- `managed_mldiagnostics` (`bool`, optional, defaults to `false`). Enable managed diagnostics.
- `managed_mldiagnostics_on_demand_profiling` (`bool`, optional, defaults to `true`). Enable the on-demand profile server with managed diagnostics.
- `managed_mldiagnostics_run_group` (`str`, optional, defaults to `""`). Group related runs.
- `managed_mldiagnostics_region` (`str`, optional, defaults to `""`). GCP region.
- `managed_mldiagnostics_storage_path` (`str`, optional, defaults to `""`). Profile and metric destination.
- `managed_mldiagnostics_metric_only_run` (`bool`, optional, defaults to `false`). Create a metrics-only run.

</details>

## `monitoring`

Controls GCP workload reports.

<details>
<summary>Parameters</summary>

- `report_heartbeat_metric_for_gcp_monitoring` (`bool`, optional, defaults to `false`). Report workload liveness.
- `heartbeat_reporting_interval_in_seconds` (`int`, optional, defaults to `5`). Seconds between heartbeats.
- `report_performance_metric_for_gcp_monitoring` (`bool`, optional, defaults to `false`). Report performance through the workload monitor. Disables goodput's separate GCP step-deviation stream.

</details>
