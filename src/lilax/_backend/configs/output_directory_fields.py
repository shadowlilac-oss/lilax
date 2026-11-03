"""Native run output directory fields derived from the base output directory and run name."""

from pydantic.fields import Field
from pydantic.main import BaseModel


class RunOutputDirectories(BaseModel):
  """Checkpoint, metrics, Tensorboard, and managed diagnostics directories of a run."""

  checkpoint_dir: None | str = Field(
      None,
      description="The full path to the checkpoint directory, derived from `run_name`.",
  )
  metrics_dir: None | str = Field(
      None,
      description="The full path to the metrics directory, derived from `run_name`.",
  )
  tensorboard_dir: None | str = Field(
      None,
      description="The full path to the tensorboard directory, derived from `run_name`.",
  )
  managed_mldiagnostics_dir: None | str = Field(
      None,
      description="The full path to the managed mldiagnostics directory, derived from `run_name`.",
  )
