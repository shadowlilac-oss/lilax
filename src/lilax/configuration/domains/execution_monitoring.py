from pydantic import BaseModel, Field, PositiveInt


class Metrics(BaseModel):
  """General configuration for metrics and monitoring."""
  __module__ = "lilax.configuration.schema"

  metrics_file: str | None = Field(None, description="Local file to store scalar metrics for testing.")
  gcs_metrics: bool = Field(False, description="If True, save metrics to GCS.")
  save_config_to_gcs: bool = Field(False, description="If True, save config to GCS.")
  record_internal_nn_metrics: int = Field(0, description="Record internal neural network metrics.")
  prometheus_port: int = Field(0, description="Port for Prometheus metrics server. 0 disables it.")
  enable_checkpoint_cloud_logger: bool = Field(False, description="Enables structured logging for checkpointing.")
  enable_wandb: bool = Field(False, description="Enable Weights & Biases logging.")
  wandb_project_name: str = Field("maxtext", description="Weights & Biases project name.")
  wandb_run_name: str = Field("", description="Weights & Biases run name. If empty, a default name is generated.")


class ManagedMLDiagnostics(BaseModel):
  """Configuration for managed mldiagnostics."""
  __module__ = "lilax.configuration.schema"

  managed_mldiagnostics: bool = Field(False, description="Enable managed mldiagnostics.")
  managed_mldiagnostics_on_demand_profiling: bool = Field(True, description="Enable on-demand profiling server when managed mldiagnostics is enabled.")
  managed_mldiagnostics_run_group: str = Field("", description="Name used to group multiple runs.")
  managed_mldiagnostics_region: str = Field("", description="GCP region for managed mldiagnostics.")
  managed_mldiagnostics_storage_path: str = Field("", description="Storage path for mldiagnostics (profiles, metrics)")
  managed_mldiagnostics_metric_only_run: bool = Field(False, description="Whether to create a metric-only run.")


class Goodput(BaseModel):
  """Configuration for goodput monitoring."""
  __module__ = "lilax.configuration.schema"

  enable_goodput_recording: bool = Field(False, description="Enable goodput recording.")
  goodput_job_name: str = Field("", description="Optional job name override for goodput recording and monitoring. Defaults to run_name when empty.")
  monitor_goodput: bool = Field(False, description="Monitor goodput.")
  goodput_upload_interval_seconds: int = Field(30, description="Interval to upload goodput metrics.")
  enable_pathways_goodput: bool = Field(False, description="Enable goodput monitoring for Pathways.")
  monitor_step_time_deviation: bool = Field(True, description="Monitor step time deviation.")
  step_deviation_interval_seconds: int = Field(30, description="Interval to check step time deviation.")
  enable_gcp_goodput_metrics: bool = Field(True, description="Enable GCP goodput metrics.")
  enable_gcp_step_deviation_metrics: bool = Field(True, description="Enable GCP step deviation metrics.")
  rolling_windows_seconds: list[PositiveInt] = Field([3600, 86400, 259200, 432000], min_length=1, description="Rolling window intervals in seconds. Must contain at least one window.")
  enable_rolling_window_goodput: bool = Field(True, description="Enable rolling window goodput.")


class GcpMonitoring(BaseModel):
  """Configuration for GCP-specific workload monitoring."""
  __module__ = "lilax.configuration.schema"

  report_heartbeat_metric_for_gcp_monitoring: bool = Field(False, description="Report heartbeat metric for GCP monitoring.")
  heartbeat_reporting_interval_in_seconds: int = Field(5, description="Interval for heartbeat metric.")
  report_performance_metric_for_gcp_monitoring: bool = Field(False, description="Report performance metric for GCP monitoring.")


class Tensorboard(BaseModel):
  """Configuration for Tensorboard logging."""
  __module__ = "lilax.configuration.schema"

  enable_tensorboard: bool = Field(True, description="Enable Tensorboard logging.")
  use_vertex_tensorboard: bool = Field(False, description=(
      "Set to True to have MaxText create (or reuse) the Vertex AI Tensorboard instance and Experiment and upload"
      " logs. Leave False if they are created outside MaxText and UPLOAD_DATA_TO_TENSORBOARD is set in the"
      " environment."
  ))
  vertex_tensorboard_project: str | None = Field("", description="GCP project for Vertex AI Tensorboard.")
  vertex_tensorboard_region: str | None = Field("", description="Region for Vertex AI Tensorboard.")
