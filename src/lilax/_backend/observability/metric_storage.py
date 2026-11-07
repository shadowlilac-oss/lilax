"""Persist scalar metrics to local files and the existing GCS batches."""

import json
import os

from lilax._backend.storage import gcs
from lilax._backend.observability import max_logging


def metrics_for_json(metrics, step, run_name):
  values = {name: float(value) for name, value in metrics["scalar"].items()}
  values["step"] = float(step)
  values["run_name"] = run_name
  return values


class ScalarMetricStorage:
  def __init__(self, config):
    self.config = config
    self.running_gcs_metrics = [] if config.gcs_metrics else None

  def write_locally(self, metrics, step):
    with open(self.config.metrics_file, "a", encoding="utf8") as output:
      if step == 0:
        output.truncate(0)
      output.write(json.dumps(metrics_for_json(metrics, step, self.config.run_name)) + "\n")

  def write_for_gcs(self, metrics, step, metric_type):
    self.running_gcs_metrics.append(metrics_for_json(metrics, step, self.config.run_name))
    if metric_type == "train" and (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      start_step = (step // self.config.log_period) * self.config.log_period
      filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
      with open(filename, "wt", encoding="utf8") as output:
        for values in self.running_gcs_metrics:
          output.write(json.dumps(values) + "\n")
      destination = os.path.join(self.config.metrics_dir, filename)
      max_logging.log(f"Moving file {filename} to GCS...")
      gcs.upload_blob(destination, filename)
      max_logging.log(f"File {filename} moved successfully!")
      self.running_gcs_metrics = []
