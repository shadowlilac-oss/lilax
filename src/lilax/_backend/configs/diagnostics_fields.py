"""Native managed diagnostics configuration fields."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class ManagedMLDiagnostics(BaseModel):
  """Configuration for managed mldiagnostics."""

  managed_mldiagnostics: bool = Field(False, description="Enable managed mldiagnostics.")
  managed_mldiagnostics_on_demand_profiling: bool = Field(
      True,
      description=("Enable on-demand profiling server when managed mldiagnostics is" " enabled."),
  )
  managed_mldiagnostics_run_group: str = Field("", description="Name used to group multiple runs.")
  managed_mldiagnostics_region: str = Field("", description="GCP region for managed mldiagnostics.")
  managed_mldiagnostics_storage_path: str = Field("", description="Storage path for mldiagnostics (profiles, metrics)")
  managed_mldiagnostics_metric_only_run: bool = Field(
      False,
      description="Whether to create a metric-only run.",
  )
