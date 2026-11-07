"""Select the existing MLPerf logging implementation for a Trainer."""

from transformers import TrainerCallback


class MLPerfCallback(TrainerCallback):
  def __init__(self, *, mllog_file=""):
    self.mllog_file = mllog_file

  def _runtime_overrides(self):
    return {"enable_mllog": True, "mllog_file": self.mllog_file}
